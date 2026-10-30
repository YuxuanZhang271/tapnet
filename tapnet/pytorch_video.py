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

"""Offline Demo for PyTorch Online TAPIR on a video file."""

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
import contextlib
import io
import time
import warnings

import cv2
import numpy as np
from tapnet.torch import tapir_model
import torch
import torch.nn.functional as F
from tqdm import tqdm
import tree

# Global model handle, set in main() so the helper functions below can use it,
# matching the style of the live demo.
model = None

# Model-input resize globals, set in main() after the first frame is read.
resize_enabled = False
model_h = None
model_w = None
scale_x = 1.0
scale_y = 1.0

# Grid-mode globals, set when --mode grid is active.
grid_spacing = 64
grid_min_brightness = 0.0


def preprocess_frames(frames):
  """Preprocess frames to model inputs.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

  Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
  """
  frames = frames.float()
  frames = frames / 255 * 2 - 1
  return frames


def online_model_init(frames, points):
  """Initialize query features for the query points."""
  frames = preprocess_frames(frames)
  feature_grids = model.get_feature_grids(frames, is_training=False)
  features = model.get_query_features(
      frames,
      is_training=False,
      query_points=points,
      feature_grids=feature_grids,
  )
  return features


def postprocess_occlusions(occlusions, expected_dist):
  visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
  return visibles


def online_model_predict(frames, features, causal_context):
  """Compute point tracks and occlusions given frames and query points."""
  frames = preprocess_frames(frames)
  feature_grids = model.get_feature_grids(frames, is_training=False)
  trajectories = model.estimate_trajectories(
      frames.shape[-3:-1],
      is_training=False,
      feature_grids=feature_grids,
      query_features=features,
      query_points_in_video=None,
      query_chunk_size=64,
      causal_context=causal_context,
      get_causal_context=True,
  )
  causal_context = trajectories["causal_context"]
  del trajectories["causal_context"]
  # Take only the predictions for the final resolution.
  # For running on higher resolution, it's typically better to average across
  # resolutions.
  tracks = trajectories["tracks"][-1]
  occlusions = trajectories["occlusion"][-1]
  uncertainty = trajectories["expected_dist"][-1]
  # Two visibility variants so the drawing mode can ignore the (sometimes
  # over-flagging) uncertainty head without touching causal bookkeeping.
  visibles_full = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(uncertainty)) > 0.5
  visibles_occ = (1 - F.sigmoid(occlusions)) > 0.5
  return tracks, visibles_full, visibles_occ, causal_context


def get_frame(video_capture):
  """Read a frame, keeping the original size and aspect ratio (no crop)."""
  r_val, image = video_capture.read()
  return r_val, image


def to_model_frame(frame):
  """Resize a frame to the model input size, preserving the aspect ratio."""
  if resize_enabled:
    return cv2.resize(frame, (model_w, model_h), interpolation=cv2.INTER_AREA)
  return frame


@contextlib.contextmanager
def _suppress_stdout():
  """Suppress stdout writes (torch prints its per-frame spam instead of
  warning)."""
  old_stdout = sys.stdout
  sys.stdout = io.StringIO()
  try:
    yield
  finally:
    sys.stdout = old_stdout


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


def run_grid_mode(args, device, vc, first_frame, fps, effective_total):
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
      if first_anchor:
        print(
            f"Grid: {len(grid_points)} points (from {len(full_grid)} before "
            f"mask/brightness filtering), spacing {grid_spacing}, window {k}."
        )
        first_anchor = False

      # Initialize query features and causal state for this anchor.
      qp = np.array(grid_points, dtype=np.float32)
      qp[:, 1] *= scale_y
      qp[:, 2] *= scale_x
      qp_t = torch.tensor(qp).to(device)
      anchor_t = torch.tensor(to_model_frame(anchor)).to(device)
      with _suppress_stdout():
        query_features = online_model_init(
            frames=anchor_t[None, None], points=qp_t[None, :]
        )
      causal_state = model.construct_initial_causal_state(
          len(qp), len(query_features.resolutions) - 1
      )
      causal_state = tree.map_structure(
          lambda x: x.to(device), causal_state
      )

      # Window: read frames t+1..t+k, predict causally, buffer for writing.
      window_frames = [anchor]
      next_anchor = None
      track = None
      vis_full = None
      vis_occ = None
      completed = False
      for j in range(1, k + 1):
        rval, f = get_frame(vc)
        if not rval:
          break
        f_tensor = torch.tensor(to_model_frame(f)).to(device)
        with _suppress_stdout():
          track, vis_full, vis_occ, causal_state = online_model_predict(
              frames=f_tensor[None, None],
              features=query_features,
              causal_context=causal_state,
          )
        window_frames.append(f)
        if j == k:
          completed = True
          next_anchor = f
          window_frames.pop()  # frame t+k belongs to the next window
          break

      # Build vectors from the endpoint track (last read frame).
      vectors = []
      if track is not None:
        track_np = track.cpu().numpy()
        track_np[..., 0] = track_np[..., 0] / scale_x
        track_np[..., 1] = track_np[..., 1] / scale_y
        vis_full_np = vis_full.cpu().numpy()
        vis_occ_np = vis_occ.cpu().numpy()
        if args.visibility == "occlusion":
          visible = vis_occ_np
        elif args.visibility == "all":
          visible = np.ones(vis_full_np.shape, dtype=bool)
        else:
          visible = vis_full_np
        for i, (t, y, x) in enumerate(grid_points):
          if not visible[0, i, 0]:
            continue
          x2 = float(track_np[0, i, 0, 0])
          y2 = float(track_np[0, i, 0, 1])
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
      description="Offline point tracking with PyTorch Online TAPIR."
  )
  parser.add_argument(
      "--video",
      required=True,
      help="Path to the input mp4 video file.",
  )
  parser.add_argument(
      "--checkpoint",
      default="checkpoints/causal_bootstapir_checkpoint.pt",
      help="Path to the TAPIR checkpoint.",
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
      "--device",
      default=None,
      help=(
          "Device to run on ('cuda' or 'cpu'). Defaults to cuda if available, "
          "otherwise cpu."
      ),
  )
  parser.add_argument(
      "--resize_height",
      type=int,
      default=256,
      help=(
          "Square size (height AND width) in pixels for the model input. The "
          "input is stretched to square (aspect ratio NOT preserved for the "
          "model — display and output stay at original resolution), matching "
          "the official torch causal colab. Set to 0 to run at the original "
          "resolution (not recommended: non-square inputs break the causal "
          "model's multi-resolution output). Default 256; 512 often improves "
          "BootsTAPIR quality at ~4x slower speed."
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
  global model, resize_enabled, model_h, model_w, scale_x, scale_y

  args = parse_args()

  # torch spam-floods this warning per frame per resolution and it buries
  # everything else. The filter covers warning-API paths; _suppress_stdout()
  # additionally silences the print()-based copies emitted by utils.py.
  warnings.filterwarnings(
      "ignore", message=".*output size is not a multiple of 8.*"
  )

  if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
    device = torch.device(args.device)

  print("Welcome to the TAPIR PyTorch video demo.")
  print("Tracking points in an offline video file.")
  print("Press ESC to exit (when display is enabled).")

  # --------------------
  # Load checkpoint and initialize
  print("Creating model...")
  model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
  print("Loading checkpoint...")
  model.load_state_dict(torch.load(args.checkpoint, map_location=device))
  model = model.to(device)
  model = model.eval()
  torch.set_grad_enabled(False)

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
  # Compute the model-input size and scaling factors. The model input is
  # stretched to a square (aspect ratio NOT preserved for the model, matching
  # the official torch causal colab); display, query points and the output
  # video all stay in original pixel coordinates. Non-square inputs break the
  # causal model's multi-resolution output, so a square input is required.
  if args.resize_height and frame.shape[0] != args.resize_height:
    resize_enabled = True
    model_h = args.resize_height
    model_w = args.resize_height  # square, aspect stretched (intentional)
    scale_y = model_h / frame.shape[0]
    scale_x = model_w / frame.shape[1]
    print(
        f"Model input resized to {model_w}x{model_h} "
        f"(square, aspect stretched; resize_height={args.resize_height})."
    )
  else:
    resize_enabled = False
    model_h = frame.shape[0]
    model_w = frame.shape[1]
    scale_y = 1.0
    scale_x = 1.0
    print("Model input running at original resolution.")

  if args.mode == "grid":
    run_grid_mode(args, device, vc, frame, fps, effective_total)
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
  query_points = torch.tensor(query_points).to(device)

  frame_tensor = torch.tensor(to_model_frame(frame)).to(device)
  with _suppress_stdout():
    query_features = online_model_init(
        frames=frame_tensor[None, None], points=query_points[None, :]
    )

  causal_state = model.construct_initial_causal_state(
      num_selected, len(query_features.resolutions) - 1
  )
  causal_state = tree.map_structure(lambda x: x.to(device), causal_state)

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
      frame_tensor = torch.tensor(to_model_frame(frame)).to(device)
      with _suppress_stdout():
        track, visibles_full, visibles_occ, causal_state = online_model_predict(
            frames=frame_tensor[None, None],
            features=query_features,
            causal_context=causal_state,
        )
      track = track.cpu().numpy()
      # Scale tracks from model pixel space back to original pixel coordinates,
      # then round.
      track[..., 0] = track[..., 0] / scale_x
      track[..., 1] = track[..., 1] / scale_y
      track = np.round(track)
      visibles_full = visibles_full.cpu().numpy()
      visibles_occ = visibles_occ.cpu().numpy()
      if args.visibility == "occlusion":
        visible = visibles_occ
      elif args.visibility == "all":
        visible = np.ones(visibles_full.shape, dtype=bool)
      else:
        visible = visibles_full

      numpy_frame = frame
      for i in range(num_selected):
        if visible[0, i, 0]:
          cv2.circle(
              numpy_frame,
              (int(track[0, i, 0, 0]), int(track[0, i, 0, 1])),
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
