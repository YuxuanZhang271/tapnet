# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Offline Demo for TensorRT Online TAPIR on a video file."""

import os
import sys

# When run as a script, Python prepends this file's directory to sys.path.
# That directory also contains a `torch/` package (the TAPIR model code), which
# would shadow the real PyTorch library. Drop the script directory and add the
# repo root instead, so `import torch` resolves to the installed PyTorch and
# `import tapnet` resolves to the local package.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR in sys.path:
  sys.path.remove(_SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

import argparse
import math
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
  import tensorrt as trt
except ImportError:
  raise ImportError(
      "TensorRT Python bindings not found. Install them for your TensorRT "
      "version (e.g. 'pip install tensorrt'), then re-run trt_video.py."
  )

# Model-input resize globals, set in main() after the first frame is read.
# The model input is a square R x R tensor whose size is fixed by the compiled
# TensorRT engine, so model_h and model_w always equal the engine resolution.
resize_enabled = False
model_h = None
model_w = None
scale_x = 1.0
scale_y = 1.0

# Grid-mode globals, set when --mode grid is active.
grid_spacing = 64
grid_min_brightness = 0.0


def sigmoid(x):
  """Element-wise sigmoid on numpy arrays (matches torch's F.sigmoid)."""
  return 1.0 / (1.0 + np.exp(-x))


def preprocess_frames(frames):
  """Preprocess frames to model inputs.

  Args:
    frames: [height, width, 3] (or batched), [0, 255], np.uint8

  Returns:
    frames: same shape, [-1, 1], np.float32
  """
  frames = frames.astype(np.float32)
  frames = frames / 255 * 2 - 1
  return frames


def get_frame(video_capture):
  """Read a frame, keeping the original size and aspect ratio (no crop)."""
  r_val, image = video_capture.read()
  return r_val, image


def to_model_frame(frame):
  """Resize a frame to the model input size, preserving the aspect ratio."""
  if resize_enabled:
    return cv2.resize(frame, (model_w, model_h), interpolation=cv2.INTER_AREA)
  return frame


def sample_points_from_mask(mask, num_points):
  """Sample points evenly across the masked (available) area.

  Uses deterministic farthest-point sampling (no RNG): the first point is the
  masked pixel closest to the mask centroid, then each next point is the
  candidate that maximizes the minimum Euclidean distance to the already
  selected points.

  Args:
    mask: Grayscale uint8 array, nonzero pixels are the available area.
    num_points: Number of points to sample.

  Returns:
    List of float32 (t=0.0, y, x) tuples in original pixel coordinates.
  """
  coords = np.argwhere(mask > 127)  # rows are [y, x]
  if coords.size == 0:
    raise ValueError("Mask has no available (nonzero) area.")
  if len(coords) > 50000:
    coords = coords[:: len(coords) // 50000 + 1]

  coords = coords.astype(np.float32)
  # Seed with the candidate pixel nearest to the mask centroid.
  centroid = coords.mean(axis=0)
  seed_idx = int(np.argmin(np.sum((coords - centroid) ** 2, axis=1)))
  min_dist = np.sum((coords - coords[seed_idx]) ** 2, axis=1)
  selected_idx = [seed_idx]

  while len(selected_idx) < num_points:
    if min_dist.max() == 0:
      break  # candidates exhausted
    nxt = int(np.argmax(min_dist))
    selected_idx.append(nxt)
    min_dist = np.minimum(
        min_dist, np.sum((coords - coords[nxt]) ** 2, axis=1)
    )

  return [(0.0, float(y), float(x)) for y, x in coords[selected_idx]]


def build_grid(frame, mask_eroded):
  """Build a fixed, evenly-spaced grid of points on the given frame.

  Args:
    frame: numpy frame in original coordinates (also used for the current
      frame's local-brightness filter).
    mask_eroded: eroded grayscale mask (nonzero = available area) or None.

  Returns:
    (full_grid, filtered_grid): lists of (t=0.0, y, x) tuples in original
    pixel coordinates. filtered_grid is the subset passing the mask and
    current-frame brightness filters.
  """
  h, w = frame.shape[:2]
  spacing = grid_spacing
  full_grid = []
  y = spacing / 2.0
  while y < h:
    x = spacing / 2.0
    while x < w:
      full_grid.append((0.0, float(y), float(x)))
      x += spacing
    y += spacing

  if mask_eroded is None and grid_min_brightness <= 0:
    return full_grid, list(full_grid)

  bright = None
  if grid_min_brightness > 0:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright = cv2.blur(gray, (31, 31))

  filtered = []
  for t, y, x in full_grid:
    yi, xi = int(y), int(x)
    if mask_eroded is not None and mask_eroded[yi, xi] <= 127:
      continue
    if bright is not None and bright[yi, xi] <= grid_min_brightness:
      continue
    filtered.append((t, y, x))
  return full_grid, filtered


class OnlineTapirTRT:
  """Runs online-TAPIR inference on a TensorRT engine.

  The engine carries the anchor's query features as state (q_lowres/q_hires
  input buffers), mirroring pytorch_video.py, where online_model_init() is
  called once per anchor and its features are reused frozen for every frame.

  Protocol: call init_from_frame() once per anchor (query points + anchor
  frame); it extracts the anchor's query features and stores them in the
  q_lowres/q_hires input buffers. Then call step() on each frame; step uses
  the frozen anchor query features and rolls the causal state forward.
  """

  def __init__(self, engine_path):
    self.logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
      with trt.Runtime(self.logger) as runtime:
        self.engine = runtime.deserialize_cuda_engine(f.read())
    self.context = self.engine.create_execution_context()
    self.stream = torch.cuda.current_stream()
    # Persistent buffers for every engine I/O tensor, allocated from the
    # engine's static shapes at load time (self-describing engine).
    self.buffers = {}
    self.io_names = []
    for i in range(self.engine.num_io_tensors):
      name = self.engine.get_tensor_name(i)
      self.io_names.append(name)
      shape = tuple(int(d) for d in self.engine.get_tensor_shape(name))
      self.buffers[name] = torch.zeros(shape, dtype=torch.float32, device='cuda')
    self.resolution = int(self.buffers['video'].shape[-2])  # square R
    self.num_points = int(self.buffers['query_points'].shape[1])
    self.num_positions = int(self.buffers['causal_1'].shape[0])
    self.num_query_levels = int(self.buffers['q_lowres'].shape[0])  # L

  def reset_causal(self):
    self.buffers['causal_1'].zero_()
    self.buffers['causal_2'].zero_()

  def reset_query_features(self):
    self.buffers['q_lowres'].zero_()
    self.buffers['q_hires'].zero_()

  def set_query_points(self, points):
    """points: np.float32 [n, 3] (t, y, x) in MODEL pixel space, n <= num_points.
    Pads to capacity by repeating the last point (duplicates are harmless:
    all per-point computations are independent)."""
    n = len(points)
    if n == 0:
      raise ValueError("No query points provided (n must be >= 1).")
    if n > self.num_points:
      raise ValueError(
          f"{n} points exceed engine capacity {self.num_points}; re-export "
          "the ONNX with a larger --num_points."
      )
    padded = np.repeat(points[-1:], self.num_points, axis=0)
    padded[:n] = points
    self.buffers['query_points'][0].copy_(
        torch.from_numpy(np.ascontiguousarray(padded))
    )

  def init_from_frame(self, points, anchor_frame):
    """Initialize per anchor: extract query features from the anchor frame.

    points: np.float32 [n, 3] (t, y, x) in MODEL pixel space, n <= num_points.
    anchor_frame: np.float32 [R, R, 3] preprocessed ([-1,1], resized) frame.

    Runs one engine call with zeroed q features and zeroed causal state; its
    tracks/causal outputs are garbage (computed from zero features) and are
    discarded. The returned q_*_out are the anchor's query features (the
    equivalent of pytorch_video.py's online_model_init); they are saved into
    the q input buffers, and the causal state is left zeroed for stepping.
    """
    self.set_query_points(points)
    self.reset_query_features()
    self.reset_causal()
    self._execute(anchor_frame, roll_causal=False)
    self.buffers['q_lowres'].copy_(self.buffers['q_lowres_out'])
    self.buffers['q_hires'].copy_(self.buffers['q_hires_out'])
    self.reset_causal()  # discard the garbage causal from the zero-feature call

  def step(self, frame):
    """frame: np.float32 [R, R, 3] already preprocessed to [-1,1] and resized.
    Uses the anchor query features saved by init_from_frame (the q buffers are
    NOT touched here, so the anchor features persist across steps).
    Returns (tracks [N,2], occlusion [N], expected_dist [N]) numpy arrays."""
    return self._execute(frame, roll_causal=True)

  def _execute(self, frame, roll_causal):
    """Copy frame into the video buffer, run the engine, optionally roll the
    causal state (stream-ordered D2D). Returns (tracks[N,2], occ[N], expd[N])."""
    self.buffers['video'][0, 0].copy_(
        torch.from_numpy(np.ascontiguousarray(frame)))
    for name in self.io_names:
      self.context.set_tensor_address(name, self.buffers[name].data_ptr())
    ok = self.context.execute_async_v3(self.stream.cuda_stream)
    if not ok:
      raise RuntimeError("TensorRT execution failed")
    if roll_causal:
      self.buffers['causal_1'].copy_(self.buffers['causal_1_out'])
      self.buffers['causal_2'].copy_(self.buffers['causal_2_out'])
    tracks = self.buffers['tracks'][0, :, 0].cpu().numpy()      # [N, 2]
    occ = self.buffers['occlusion'][0, :, 0].cpu().numpy()      # [N]
    expd = self.buffers['expected_dist'][0, :, 0].cpu().numpy()  # [N]
    return tracks, occ, expd


def run_grid_mode(args, runner, vc, first_frame, fps, effective_total):
  """Run --mode grid: fixed-grid motion-vector field, re-anchored every k
  frames. Owns vc/writer cleanup and prints its own summary."""
  global grid_spacing, grid_min_brightness
  grid_spacing = args.grid_spacing
  grid_min_brightness = args.min_brightness

  if args.points is not None:
    raise ValueError("--points applies to --mode track only.")
  print("Note: --num_points is ignored in grid mode.")

  # Prepare the eroded mask (same --mask_margin erosion as track mode).
  mask_eroded = None
  if args.mask is not None:
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
      raise ValueError(f"Unable to read mask: {args.mask}")
    if mask.shape[:2] != first_frame.shape[:2]:
      print(
          f"Warning: mask size {mask.shape[:2]} differs from frame "
          f"{first_frame.shape[:2]}, resizing mask to match the frame."
      )
      mask = cv2.resize(
          mask, (first_frame.shape[1], first_frame.shape[0]),
          interpolation=cv2.INTER_NEAREST,
      )
    mask_eroded = mask
    if args.mask_margin > 0:
      print(f"Eroding mask by {args.mask_margin}px before sampling.")
      kernel = cv2.getStructuringElement(
          cv2.MORPH_ELLIPSE,
          (2 * args.mask_margin + 1, 2 * args.mask_margin + 1),
      )
      eroded = cv2.erode(mask, kernel)
      if np.any(eroded > 127):
        mask_eroded = eroded
      else:
        print(
            "mask_margin too large; no available area left after erosion. "
            "Falling back to no margin."
        )

  frame_h, frame_w = first_frame.shape[:2]
  frame_diag = np.hypot(frame_w, frame_h)
  k = args.vector_window

  # --------------------
  # Prepare the output writer if requested.
  writer = None
  if args.output:
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        args.output, fourcc, fps, (frame_w, frame_h)
    )
    if not writer.isOpened():
      vc.release()
      raise ValueError(f"Unable to open output video for writing: {args.output}")

  # --------------------
  # Grid windows: anchor frames t = 0, k, 2k, ... Window t owns frames
  # [t, t+k); frame t+k is read for the endpoint track but carried over as the
  # next anchor (written by the next window with the new arrows).
  processed = 0
  start_time = time.time()
  last_postfix = time.time()
  first_anchor = True
  fallback_printed = False

  pbar = tqdm(
      total=effective_total, desc="Grid", unit="frame", dynamic_ncols=True
  )
  anchor = first_frame
  try:
    while True:
      full_grid, filtered_grid = build_grid(anchor, mask_eroded)
      if filtered_grid:
        grid_points = filtered_grid
      else:
        grid_points = full_grid
        if not fallback_printed:
          print(
              "Note: mask/brightness filtering removed all grid points; "
              "using unfiltered grid for this anchor."
          )
          fallback_printed = True

      # If the grid exceeds the engine's point capacity, evenly subsample it.
      n_grid = len(grid_points)
      if n_grid > runner.num_points:
        stride = max(1, math.ceil(n_grid / runner.num_points))
        grid_points = grid_points[::stride]
        print(
            f"Warning: grid yielded {n_grid} points, exceeding engine capacity "
            f"{runner.num_points}; evenly subsampled with stride {stride} to "
            f"{len(grid_points)} points."
        )

      if first_anchor:
        print(
            f"Grid: {len(grid_points)} points (from {len(full_grid)} before "
            f"mask/brightness filtering), spacing {grid_spacing}, window {k}."
        )
        first_anchor = False

      # Initialize query features and causal state for this anchor. The
      # graph carries query features as state, so extracting them from the
      # anchor frame here is the online_model_init equivalent; the window
      # steps then run on frames t+1..t+k with zero-start causal state.
      qp = np.array(grid_points, dtype=np.float32)
      qp[:, 1] *= scale_y
      qp[:, 2] *= scale_x
      runner.init_from_frame(qp, preprocess_frames(to_model_frame(anchor)))

      # Window: read frames t+1..t+k, predict causally, buffer for writing.
      window_frames = [anchor]
      next_anchor = None
      track = None
      vis_full = None
      vis_occ = None
      for j in range(1, k + 1):
        rval, f = get_frame(vc)
        if not rval:
          break
        track, occ, expd = runner.step(
            preprocess_frames(to_model_frame(f))
        )
        vis_full = (1 - sigmoid(occ)) * (1 - sigmoid(expd)) > 0.5
        vis_occ = (1 - sigmoid(occ)) > 0.5
        window_frames.append(f)
        if j == k:
          next_anchor = f
          window_frames.pop()  # frame t+k belongs to the next window
          break

      # Build vectors from the endpoint track (last read frame).
      vectors = []
      if track is not None:
        track_np = track
        track_np[..., 0] = track_np[..., 0] / scale_x
        track_np[..., 1] = track_np[..., 1] / scale_y
        if args.visibility == "occlusion":
          visible = vis_occ
        elif args.visibility == "all":
          visible = np.ones(vis_full.shape, dtype=bool)
        else:
          visible = vis_full
        for i, (t, y, x) in enumerate(grid_points):
          if not visible[i]:
            continue
          x2 = float(track_np[i, 0])
          y2 = float(track_np[i, 1])
          if np.hypot(x2 - x, y2 - y) > 0.4 * frame_diag:
            continue  # bad-track guard
          vectors.append((int(x), int(y), int(x2), int(y2)))

      # Draw the same vector set on every buffered frame of the window.
      for f_img in window_frames:
        for x, y, x2, y2 in vectors:
          cv2.arrowedLine(
              f_img,
              (x, y),
              (x2, y2),
              (0, 255, 0),
              max(1, int(round(args.marker_scale))),
              cv2.LINE_8,
              0,
              0.25,
          )
        if writer is not None:
          writer.write(f_img)
        if not args.no_display:
          cv2.imshow("Point Tracking", f_img)
          key = cv2.waitKey(1)
          if key == 27:  # exit on ESC
            next_anchor = None
            break
        processed += 1
        pbar.update(1)
        if time.time() - last_postfix > 1.0:
          pbar.set_postfix_str(f"{processed / (time.time() - start_time):.2f} fps")
          last_postfix = time.time()
        if args.max_frames is not None and processed >= args.max_frames:
          next_anchor = None
          break

      if next_anchor is None:
        break
      anchor = next_anchor
  finally:
    pbar.close()

  elapsed = time.time() - start_time
  avg_fps = processed / elapsed if elapsed > 0 else 0.0

  # --------------------
  # Clean up.
  if not args.no_display:
    cv2.destroyAllWindows()
  vc.release()
  if writer is not None:
    writer.release()

  summary = f"Processed {processed} frame(s), average {avg_fps:.2f} fps"
  if args.output:
    summary += f", output written to {args.output}"
  summary += f" (grid spacing {args.grid_spacing}, window {args.vector_window})"
  print(summary)


def parse_args():
  parser = argparse.ArgumentParser(
      description="Offline point tracking with TensorRT Online TAPIR."
  )
  parser.add_argument(
      "--video",
      required=True,
      help="Path to the input mp4 video file.",
  )
  parser.add_argument(
      "--engine",
      default="checkpoints/causal_bootstapir_checkpoint_fp16.engine",
      help="Path to the compiled TensorRT engine (.engine) file.",
  )
  parser.add_argument(
      "--num_points",
      type=int,
      default=16,
      help="Maximum number of query points to track.",
  )
  parser.add_argument(
      "--points",
      default=None,
      help=(
          "Optional space-separated 'x,y' pixel coordinates in the first frame, "
          "e.g. \"100,150 200,150\". If given, interactive selection is skipped."
      ),
  )
  parser.add_argument(
      "--mask",
      default=None,
      help=(
          "Path to a grayscale mask PNG (nonzero = available area). Points are "
          "distributed evenly across the masked region instead of interactive "
          "selection."
      ),
  )
  parser.add_argument(
      "--mask_margin",
      type=int,
      default=15,
      help=(
          "Erode the mask by this many pixels before sampling points, keeping "
          "them away from the soft/unreliable boundary of the available area. "
          "Use 0 to disable."
      ),
  )
  parser.add_argument(
      "--min_brightness",
      type=float,
      default=0.0,
      help=(
          "Only sample points where the average brightness of the first frame "
          "(31px local window) exceeds this value (0-255). Excludes dark, "
          "untrackable voids inside the mask. Use 0 to disable. Try 40 for "
          "endoscopy-style footage."
      ),
  )
  parser.add_argument(
      "--output",
      default=None,
      help="Optional path to write an annotated output mp4.",
  )
  parser.add_argument(
      "--max_frames",
      type=int,
      default=None,
      help="Optional maximum number of frames to process.",
  )
  parser.add_argument(
      "--no_display",
      action="store_true",
      help="Headless mode: do not open any cv2 window.",
  )
  parser.add_argument(
      "--resize_height",
      type=int,
      default=None,
      help=(
          "Optional square size (height AND width) in pixels for the model "
          "input. Must match the engine's fixed resolution; default None uses "
          "the engine resolution. Frames are stretched to square for the "
          "model (aspect ratio NOT preserved for the model); display and "
          "output stay at original resolution."
      ),
  )
  parser.add_argument(
      "--visibility",
      choices=["full", "occlusion", "all"],
      default="full",
      help=(
          "How to decide whether a tracked point is drawn. 'full': visible "
          "only if not occluded AND uncertainty is low (strict; the "
          "uncertainty head can over-flag on out-of-distribution footage like "
          "endoscopy). 'occlusion': ignore the uncertainty head, use occlusion "
          "only. 'all': always draw all tracks."
      ),
  )
  parser.add_argument(
      "--mode",
      choices=["track", "grid"],
      default="track",
      help=(
          "'track': persistent point tracking (default). 'grid': fixed, "
          "evenly-distributed grid of points rendered as motion vectors, "
          "re-anchored every --vector_window frames."
      ),
  )
  parser.add_argument(
      "--grid_spacing",
      type=int,
      default=64,
      help="Grid spacing in pixels (original video coordinates) for --mode grid.",
  )
  parser.add_argument(
      "--vector_window",
      type=int,
      default=1,
      help=(
          "Vector window k: each vector shows motion from its anchor frame t "
          "to frame t+k. Default 1 (frame-to-frame). The grid is re-anchored "
          "every k frames."
      ),
  )
  parser.add_argument(
      "--marker_scale",
      type=float,
      default=2.0,
      help=(
          "Scale factor for drawn markers: point radius in track mode, arrow "
          "thickness in grid mode, and interactive selection dots. 2.0 makes "
          "markers twice as large."
      ),
  )
  return parser.parse_args()


def main():
  global resize_enabled, model_h, model_w, scale_x, scale_y

  args = parse_args()

  print("Welcome to the TAPIR TensorRT video demo.")
  print("Tracking points in an offline video file.")
  print("Press ESC to exit (when display is enabled).")

  # --------------------
  # Load the TensorRT engine. Its static I/O shapes are self-describing:
  # resolution R and point capacity N are read from the engine.
  print("Creating model...")
  runner = OnlineTapirTRT(args.engine)
  print(
      f"Loaded TensorRT engine from {args.engine}: resolution "
      f"{runner.resolution} (square), point capacity {runner.num_points}, "
      f"query-feature levels {runner.num_query_levels}."
  )

  # --------------------
  # Open the input video
  print("Opening video...")
  vc = cv2.VideoCapture(args.video)
  if not vc.isOpened():
    raise ValueError(f"Unable to open video: {args.video}")

  fps = vc.get(cv2.CAP_PROP_FPS)
  if not fps or fps <= 0:
    fps = 30
  frame_w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

  total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
  if total_frames <= 0:
    effective_total = None  # unknown frame count -> indeterminate bar
  elif args.max_frames is not None:
    effective_total = min(args.max_frames, total_frames)
  else:
    effective_total = total_frames

  rval, frame = get_frame(vc)
  if not rval:
    vc.release()
    raise ValueError(f"Unable to read the first frame of video: {args.video}")

  # --------------------
  # The model-input size is fixed by the engine: a square R x R tensor. The
  # input is stretched to square (aspect ratio NOT preserved for the model,
  # matching the official torch causal colab); display, query points and the
  # output video all stay in original pixel coordinates.
  resolution = runner.resolution
  if args.resize_height is not None and args.resize_height != resolution:
    vc.release()
    raise ValueError(
        f"--resize_height {args.resize_height} does not match the engine "
        f"resolution {resolution}. Re-export the ONNX at the desired "
        f"resolution, or omit --resize_height to use the engine resolution."
    )
  model_h = resolution
  model_w = resolution
  if frame.shape[0] != resolution or frame.shape[1] != resolution:
    resize_enabled = True
    scale_y = model_h / frame.shape[0]
    scale_x = model_w / frame.shape[1]
    print(
        f"Model input resized to {model_w}x{model_h} "
        f"(square, aspect stretched; engine resolution {resolution})."
    )
  else:
    resize_enabled = False
    scale_y = 1.0
    scale_x = 1.0
    print(f"Model input running at engine resolution {resolution}.")

  if args.mode == "grid":
    run_grid_mode(args, runner, vc, frame, fps, effective_total)
    return

  # --------------------
  # Select query points on the first frame.
  selected_points = []  # list of (t, y, x) tuples.

  if args.points is not None:
    if args.mask is not None:
      print("--points given, ignoring --mask.")
    try:
      for pair in args.points.split():
        x_str, y_str = pair.split(",")
        selected_points.append((0.0, float(y_str), float(x_str)))
    except ValueError as exc:
      raise ValueError(
          f"Invalid --points value {args.points!r}; expected space-separated "
          "'x,y' pairs, e.g. \"100,150 200,150\"."
      ) from exc
    print(f"Using {len(selected_points)} point(s) from --points.")
  elif args.mask is not None:
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
      raise ValueError(f"Unable to read mask: {args.mask}")
    if mask.shape[:2] != frame.shape[:2]:
      print(
          f"Warning: mask size {mask.shape[:2]} differs from frame "
          f"{frame.shape[:2]}, resizing mask to match the frame."
      )
      mask = cv2.resize(
          mask, (frame.shape[1], frame.shape[0]),
          interpolation=cv2.INTER_NEAREST,
      )
    sample_mask = mask
    if args.mask_margin > 0:
      print(f"Eroding mask by {args.mask_margin}px before sampling.")
      kernel = cv2.getStructuringElement(
          cv2.MORPH_ELLIPSE,
          (2 * args.mask_margin + 1, 2 * args.mask_margin + 1),
      )
      eroded = cv2.erode(mask, kernel)
      if np.any(eroded > 127):
        sample_mask = eroded
      else:
        print(
            "mask_margin too large; no available area left after erosion. "
            "Falling back to no margin."
        )
    if args.min_brightness > 0:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      brightness = cv2.blur(gray, (31, 31))
      bright_mask = (
          (brightness > args.min_brightness) & (sample_mask > 127)
      ).astype(np.uint8) * 255
      if np.any(bright_mask > 127):
        print(
            f"Excluding first-frame areas with local brightness below "
            f"{args.min_brightness}."
        )
        sample_mask = bright_mask
      else:
        print(
            "min_brightness too high; no available area left after filtering. "
            "Falling back to brightness-unfiltered mask."
        )
    selected_points = sample_points_from_mask(sample_mask, args.num_points)
    print(f"Sampled {len(selected_points)} point(s) from mask {args.mask}.")
    print("Points: " + " ".join(f"{int(x)},{int(y)}" for _, y, x in selected_points))
  elif not args.no_display:
    display_frame = frame.copy()
    window_name = "Select Points"

    def mouse_click(event, x, y, flags, param):
      del flags, param
      if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_points) < args.num_points:
          selected_points.append((0.0, float(y), float(x)))
          cv2.circle(display_frame, (x, y), int(5 * args.marker_scale), (0, 255, 0), -1)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click)

    print("Left click to add points, 'r' to reset, SPACE/ENTER to finish, ESC to quit.")
    while True:
      cv2.imshow(window_name, display_frame)
      key = cv2.waitKey(20) & 0xFF
      if key == 27:  # ESC
        print("Point selection cancelled.")
        cv2.destroyWindow(window_name)
        vc.release()
        return
      elif key == ord("r"):
        selected_points.clear()
        display_frame = frame.copy()
      elif key in (13, 32):  # ENTER or SPACE
        if not selected_points:
          print("Please select at least one point before finishing.")
          continue
        print(f"Selected {len(selected_points)} point(s).")
        break

    cv2.destroyWindow(window_name)
  else:
    vc.release()
    raise ValueError(
        "No query points available: pass --points (e.g. \"100,150 200,150\") "
        "or --mask (path to a mask PNG) when using --no_display."
    )

  if not selected_points:
    vc.release()
    raise ValueError("No query points selected.")

  num_selected = len(selected_points)
  query_points = np.array(selected_points, dtype=np.float32)
  # Scale (t, y, x) rows into the resized model frame's pixel space.
  query_points[:, 1] *= scale_y
  query_points[:, 2] *= scale_x

  # The engine carries the anchor's query features as state: extract them
  # from the first (anchor) frame here, exactly like pytorch_video.py's
  # online_model_init. The tracking loop then steps on every frame starting
  # with frame 0, whose step (anchor features + zero causal state) reproduces
  # pytorch_video.py's frame-0 predict.
  runner.init_from_frame(
      query_points, preprocess_frames(to_model_frame(frame))
  )

  # --------------------
  # Prepare the output writer if requested.
  writer = None
  if args.output:
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        args.output, fourcc, fps, (frame.shape[1], frame.shape[0])
    )
    if not writer.isOpened():
      vc.release()
      raise ValueError(f"Unable to open output video for writing: {args.output}")

  # --------------------
  # Track points across the remaining frames. The first frame is processed
  # through the same predict/draw path so the output starts at frame 0.
  processed = 0
  start_time = time.time()
  last_postfix = time.time()

  pbar = tqdm(
      total=effective_total, desc="Tracking", unit="frame", dynamic_ncols=True
  )
  try:
    while True:
      track, occlusion, expected_dist = runner.step(
          preprocess_frames(to_model_frame(frame))
      )
      # Scale tracks from model pixel space back to original pixel coordinates,
      # then round.
      track[..., 0] = track[..., 0] / scale_x
      track[..., 1] = track[..., 1] / scale_y
      track = np.round(track)
      visibles_full = (
          (1 - sigmoid(occlusion)) * (1 - sigmoid(expected_dist)) > 0.5
      )
      visibles_occ = (1 - sigmoid(occlusion)) > 0.5
      if args.visibility == "occlusion":
        visible = visibles_occ
      elif args.visibility == "all":
        visible = np.ones(visibles_full.shape, dtype=bool)
      else:
        visible = visibles_full

      numpy_frame = frame
      for i in range(num_selected):
        if visible[i]:
          cv2.circle(
              numpy_frame,
              (int(track[i, 0]), int(track[i, 1])),
              int(5 * args.marker_scale),
              (255, 0, 0),
              -1,
          )

      if writer is not None:
        writer.write(numpy_frame)

      if not args.no_display:
        cv2.imshow("Point Tracking", numpy_frame)
        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
          break

      processed += 1
      pbar.update(1)
      if time.time() - last_postfix > 1.0:
        pbar.set_postfix_str(f"{processed / (time.time() - start_time):.2f} fps")
        last_postfix = time.time()

      if args.max_frames is not None and processed >= args.max_frames:
        break

      rval, frame = get_frame(vc)
      if not rval:
        break
  finally:
    pbar.close()

  elapsed = time.time() - start_time
  avg_fps = processed / elapsed if elapsed > 0 else 0.0

  # --------------------
  # Clean up.
  if not args.no_display:
    cv2.destroyAllWindows()
  vc.release()
  if writer is not None:
    writer.release()

  summary = f"Processed {processed} frame(s), average {avg_fps:.2f} fps"
  if args.output:
    summary += f", output written to {args.output}"
  print(summary)


if __name__ == "__main__":
  main()