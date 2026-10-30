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

"""Export the causal (Boots)TAPIR online model to a single ONNX graph.

The online/causal TAPIR model keeps state between frames: the query features
extracted once from the ANCHOR frame (pytorch_video.py's online_model_init),
plus a causal conv state (a list of dicts: `num_pips_iter *
num_refinement_levels` positions, each with `num_mixer_blocks` x 2 tensors).
ONNX and TensorRT need flat tensor I/O, so this exporter flattens all of that
state into tensors and exports ONE unified static-shape graph that performs a
single online step:

  inputs:
    video        [1, 1, R, R, 3]      float32  one frame, values in [-1, 1]
    query_points [1, N, 3]            float32  (t, y, x) in model pixel space
    q_lowres     [L, N, 256]          float32  anchor query features (lowres)
    q_hires      [L, N, 128]          float32  anchor query features (hires)
    causal_1     [P, 12, N, 2, 512]   float32  causal conv state (small)
    causal_2     [P, 12, N, 2, 2048]  float32  causal conv state (large)
  outputs:
    tracks        [1, N, 1, 2]  float32  (x, y) in model pixel space
    occlusion     [1, N, 1]     float32  occlusion logits
    expected_dist [1, N, 1]     float32  uncertainty logits
    causal_1_out  [P, 12, N, 2, 512]   float32  updated causal state
    causal_2_out  [P, 12, N, 2, 2048]  float32  updated causal state
    q_lowres_out  [L, N, 256]   float32  query features from THIS frame
    q_hires_out   [L, N, 128]   float32  query features from THIS frame

where R = --resolution (square), N = --num_points (capacity; pad at inference
time), L = 1 + num_refinement_levels (2 for R=256, 3 for R=512), and
P = 4 * num_refinement_levels (4 for R=256, 8 for R=512).

CALLER PROTOCOL (matches pytorch_video.py's numerics exactly):

  1. Init (once per anchor: video start in track mode, or every re-anchor in
     grid mode): call the graph with the ANCHOR frame, the query points,
     q_lowres/q_hires = ZEROS and causal = ZEROS. This call's tracks/causal
     outputs are garbage (they were computed from zero features) and must be
     discarded. Save q_lowres_out/q_hires_out; these are the anchor's query
     features (the equivalent of online_model_init). Re-zero the causal state.

  2. Track mode: then call the graph once per frame starting WITH the anchor
     frame, feeding the SAVED anchor features and the rolling causal state.
     This reproduces pytorch_video.py's loop, which predicts frame 0 with the
     frame-0 features and a zero causal state.

  3. Grid mode: after init, call the graph on frames t+1..t+k with the saved
     anchor features and a causal state that starts at zero.

On every call the graph also recomputes the current frame's query features
(cheap: interpolation over the feature grids it already computed) and returns
them as q_*_out; regular steps must IGNORE them and keep feeding the anchor
features saved during init.

Shapes are STATIC on purpose: the query permutation and chunking inside
estimate_trajectories are baked at trace time, so N and R must be fixed when
the ONNX file is created (and therefore when the TensorRT engine is built).

Typical usage:
  python tapnet/export_onnx.py \
      --checkpoint checkpoints/causal_bootstapir_checkpoint.pt \
      --output checkpoints/causal_bootstapir_checkpoint.onnx

Then compile with trtexec:
  trtexec --onnx=checkpoints/causal_bootstapir_checkpoint.onnx \
      --saveEngine=checkpoints/causal_bootstapir_checkpoint_fp16.engine --fp16
"""

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

import numpy as np
from tapnet.torch import tapir_model
from tapnet.torch import utils as model_utils
import torch
import torch.nn.functional as F

# Exact I/O names; tapnet/trt_video.py relies on these verbatim.
INPUT_NAMES = ['video', 'query_points', 'q_lowres', 'q_hires',
               'causal_1', 'causal_2']
OUTPUT_NAMES = [
    'tracks',
    'occlusion',
    'expected_dist',
    'causal_1_out',
    'causal_2_out',
    'q_lowres_out',
    'q_hires_out',
]


def _map_coordinates_3d_single_frame(feats, coordinates):
  """Single-frame (T==1) rewrite of utils.map_coordinates_3d.

  The stock implementation uses 5-D volumetric F.grid_sample, which the ONNX
  exporter (and TensorRT) do not support. For T==1 the time axis is degenerate
  (it always samples its only voxel with weight 1.0), so a 4-D grid_sample is
  numerically identical. Patched into tapnet.torch.utils for the export
  process only; the model library itself is untouched.
  """
  if feats.shape[1] != 1:
    raise ValueError(
        'This ONNX export only supports single-frame (T=1) online steps, got '
        f'T={feats.shape[1]}.'
    )
  b, _, h, w, c = feats.shape
  # feats is [B, T, H, W, C]; select the single frame -> [B, H, W, C] ->
  # [B, C, H, W].
  x = feats[:, 0].permute(0, 3, 1, 2)
  # coordinates are (t, y, x); drop t and normalize (y, x) by (H, W).
  y = coordinates[:, :, None, None, 1:].float()  # [B, N, 1, 1, 2]
  y = 2 * (y / torch.tensor((h, w), device=y.device)) - 1
  y = torch.flip(y, dims=(-1,)).squeeze(3)  # [B, N, 1, 2] as (x, y)
  out = F.grid_sample(
      x, y, mode='bilinear', align_corners=False, padding_mode='border'
  )  # [B, C, N, 1]
  return out.squeeze(dim=3).permute(0, 2, 1)  # [B, N, C]


# tapir_model resolves utils.map_coordinates_3d at call time through the module
# object, so this attribute patch covers the traced path.
model_utils.map_coordinates_3d = _map_coordinates_3d_single_frame


class OnlineTAPIRWrapper(torch.nn.Module):
  """Wraps the causal TAPIR model with flat-tensor I/O for ONNX export.

  forward() performs one online step with the caller-supplied anchor query
  features and causal state, mirroring exactly what pytorch_video.py does per
  frame. It also recomputes the current frame's query features and returns
  them; the caller saves them on the init call (when the q_* inputs are zeros)
  and ignores them on regular steps. See the module docstring for the full
  caller protocol.
  """

  def __init__(self, model, num_points):
    super().__init__()
    self._model = model
    self._num_points = num_points

  def forward(self, video, query_points, q_lowres, q_hires, causal_1,
              causal_2):
    model = self._model
    num_positions = causal_1.shape[0]  # P, static
    num_blocks = model.num_mixer_blocks  # 12
    num_levels = q_lowres.shape[0]  # L, static

    feature_grids = model.get_feature_grids(video, is_training=False)

    # Anchor query features carried in from the init call. On the init call
    # itself these are zeros, and only the q_*_out outputs below matter.
    query_features = tapir_model.QueryFeatures(
        lowres=tuple(
            q_lowres[l].unsqueeze(0) for l in range(num_levels)
        ),
        hires=tuple(q_hires[l].unsqueeze(0) for l in range(num_levels)),
        resolutions=feature_grids.resolutions,
    )

    # Fresh query features from the CURRENT frame at the query positions
    # (cheap: interpolation over feature_grids). On the init call these are
    # the anchor features the caller saves (= online_model_init).
    computed = model.get_query_features(
        video,
        is_training=False,
        query_points=query_points,
        feature_grids=feature_grids,
    )
    q_lowres_out = torch.stack([f[0] for f in computed.lowres], dim=0)
    q_hires_out = torch.stack([f[0] for f in computed.hires], dim=0)

    # Unflatten [P, 12, N, 2, C] into the list-of-dicts the model expects.
    # Each entry value gets a leading batch dim of 1, matching
    # construct_initial_causal_state's (1, num_points, 2, C) layout.
    causal_context = []
    for p in range(num_positions):
      state = {}
      for b in range(num_blocks):
        state[f'block_{b}_causal_1'] = causal_1[p, b].unsqueeze(0)
        state[f'block_{b}_causal_2'] = causal_2[p, b].unsqueeze(0)
      causal_context.append(state)

    trajectories = model.estimate_trajectories(
        video.shape[-3:-1],
        is_training=False,
        feature_grids=feature_grids,
        query_features=query_features,
        query_points_in_video=None,
        # Single chunk: with static shapes the chunk loop is baked, and one
        # chunk keeps the graph simplest.
        query_chunk_size=self._num_points,
        causal_context=causal_context,
        get_causal_context=True,
    )

    # Re-flatten the new causal state: list of P dicts of (1, N, 2, C) ->
    # [P, 12, N, 2, C] (drop the leading batch dim of 1).
    new_causal = trajectories['causal_context']
    causal_1_out = torch.stack(
        [
            torch.stack(
                [new_causal[p][f'block_{b}_causal_1'][0] for b in range(num_blocks)],
                dim=0,
            )
            for p in range(num_positions)
        ],
        dim=0,
    )
    causal_2_out = torch.stack(
        [
            torch.stack(
                [new_causal[p][f'block_{b}_causal_2'][0] for b in range(num_blocks)],
                dim=0,
            )
            for p in range(num_positions)
        ],
        dim=0,
    )

    # Final-resolution predictions only (matches pytorch_video.py, which takes
    # trajectories[...][-1]).
    return (
        trajectories['tracks'][-1],
        trajectories['occlusion'][-1],
        trajectories['expected_dist'][-1],
        causal_1_out,
        causal_2_out,
        q_lowres_out,
        q_hires_out,
    )


def parse_args():
  parser = argparse.ArgumentParser(
      description='Export the causal (Boots)TAPIR online model to ONNX.'
  )
  parser.add_argument(
      '--checkpoint',
      default='checkpoints/causal_bootstapir_checkpoint.pt',
      help='Path to the TAPIR checkpoint.',
  )
  parser.add_argument(
      '--output',
      default='checkpoints/causal_bootstapir_checkpoint.onnx',
      help='Path for the exported ONNX file.',
  )
  parser.add_argument(
      '--resolution',
      type=int,
      default=256,
      help=(
          'Square model input resolution R in pixels (must be a multiple of 8; '
          'must match the --resize_height used at inference). 256 matches the '
          'pytorch_video.py default; 512 gives higher-quality BootsTAPIR '
          'tracks at ~4x the compute.'
      ),
  )
  parser.add_argument(
      '--num_points',
      type=int,
      default=64,
      help=(
          'Point capacity N baked into the graph. At inference, query points '
          'are padded up to N (duplicates are harmless), so pick a value >= '
          'the largest point count you will use (e.g. --num_points for track '
          'mode, or the grid point count for --mode grid). The engine must be '
          're-exported to change N.'
      ),
  )
  parser.add_argument(
      '--opset',
      type=int,
      default=17,
      help='ONNX opset version (>=16 required for grid_sample).',
  )
  parser.add_argument(
      '--no_verify',
      action='store_true',
      help='Skip the onnxruntime numeric check after export.',
  )
  parser.add_argument(
      '--device',
      default=None,
      help="Device to trace on ('cuda' or 'cpu'). Defaults to cuda if available.",
  )
  return parser.parse_args()


def build_model(checkpoint_path, device):
  """Builds the causal BootsTAPIR model and loads the checkpoint."""
  model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
  state = torch.load(checkpoint_path, map_location=device)
  model.load_state_dict(state)
  model = model.to(device)
  model = model.eval()
  return model


def make_dummy_inputs(resolution, num_points, num_positions, num_levels,
                      num_blocks, device, rng):
  """Creates dummy inputs for tracing (values matter only for exportability).

  The q_* and causal_* tensors are zeros so that the traced graph is the init
  call; the caller protocol then exercises the same graph with the saved
  features and rolling state.
  """
  video = rng.uniform(-1, 1, size=(1, 1, resolution, resolution, 3))
  query_points = np.zeros((1, num_points, 3), dtype=np.float32)
  query_points[:, :, 1] = rng.uniform(0, resolution, size=num_points)  # y
  query_points[:, :, 2] = rng.uniform(0, resolution, size=num_points)  # x
  q_lowres = np.zeros((num_levels, num_points, 256), dtype=np.float32)
  q_hires = np.zeros((num_levels, num_points, 128), dtype=np.float32)
  causal_1 = np.zeros(
      (num_positions, num_blocks, num_points, 2, 512), dtype=np.float32
  )
  causal_2 = np.zeros(
      (num_positions, num_blocks, num_points, 2, 2048), dtype=np.float32
  )
  return tuple(
      torch.tensor(x, dtype=torch.float32, device=device)
      for x in (video, query_points, q_lowres, q_hires, causal_1, causal_2)
  )


def _structured_frame(rng, resolution, device):
  """A smooth, structured random frame (low-frequency blobs + mild detail).

  Uniform white-noise frames put the model in a degenerate regime (the
  cost-volume softmax is nearly flat, so the soft-argmax track output is
  hypersensitive to ~1e-4 numeric noise and cross-backend comparisons look
  far worse than they do on real imagery). Structured frames keep the
  verification in the model's normal operating regime.
  """
  low = rng.uniform(-1, 1, size=(1, 3, 8, 8)).astype(np.float32)
  t = torch.from_numpy(low)
  t = F.interpolate(t, size=(resolution, resolution), mode='bicubic',
                    align_corners=False)
  frame = t.permute(0, 2, 3, 1)[None]  # [1, 1, R, R, 3]
  frame = frame + 0.1 * torch.from_numpy(
      rng.uniform(-1, 1, size=frame.shape).astype(np.float32)
  )
  return frame.clamp(-1, 1).to(device)


def verify(onnx_path, wrapper, dummy_inputs, device):
  """Compares ONNX (onnxruntime) vs PyTorch over the full caller protocol.

  Runs the init call (zero q features, zero causal state), then chains two
  regular steps feeding back the saved features and rolling causal state, and
  checks every output at each stage. Frames are structured (see
  _structured_frame) to stay in the model's normal operating regime.
  """
  try:
    import onnxruntime as rt
  except ImportError:
    print('onnxruntime not installed; skipping verification.')
    return True

  session = rt.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

  def run_onnx(video, qp, ql, qh, c1, c2):
    feeds = dict(
        zip(INPUT_NAMES, [t.cpu().numpy() for t in (video, qp, ql, qh, c1, c2)])
    )
    outs = session.run(OUTPUT_NAMES, feeds)
    return [torch.tensor(o, device=device) for o in outs]

  def compare(stage, ref, got):
    ok = True
    for name, r, g in zip(OUTPUT_NAMES, ref, got):
      diff = (r.float() - g.float()).abs().max().item()
      status = 'ok' if diff < 0.05 else ('DRIFT' if diff < 1.0 else 'FAIL')
      if status != 'ok':
        ok = False
      print(f'  {stage} {name:>15}: max|diff| = {diff:.3e} [{status}]')
    return ok

  ok = True
  resolution = dummy_inputs[0].shape[-2]
  rng = np.random.default_rng(1234)
  with torch.no_grad():
    video0 = _structured_frame(rng, resolution, device)
    # Init call: zero q features, zero causal.
    ref = wrapper(video0, *dummy_inputs[1:])
    got = run_onnx(video0, *dummy_inputs[1:])
    ok &= compare('init', ref, got)
    # Save features (torch ref / onnx), reset causal, chain two steps.
    ql_t, qh_t = ref[5], ref[6]
    ql_o, qh_o = got[5], got[6]
    c1_t, c2_t = torch.zeros_like(dummy_inputs[4]), torch.zeros_like(
        dummy_inputs[5]
    )
    c1_o, c2_o = torch.zeros_like(dummy_inputs[4]), torch.zeros_like(
        dummy_inputs[5]
    )
    for step in range(2):
      video = _structured_frame(rng, resolution, device)
      ref = wrapper(video, dummy_inputs[1], ql_t, qh_t, c1_t, c2_t)
      got = run_onnx(video, dummy_inputs[1], ql_o, qh_o, c1_o, c2_o)
      ok &= compare(f'step {step}', ref, got)
      c1_t, c2_t = ref[3], ref[4]
      c1_o, c2_o = got[3], got[4]
  return ok


def main():
  args = parse_args()
  if args.resolution % 8 != 0:
    raise ValueError('--resolution must be a multiple of 8.')
  if args.device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  else:
    device = torch.device(args.device)
  torch.set_grad_enabled(False)

  resolution = args.resolution
  num_points = args.num_points

  # Resolution bookkeeping matching pytorch_video.py:
  #   refinement_levels = generate_default_resolutions(video, (256, 256))
  #   query-feature levels L = 1 + len(refinement_levels)
  #   causal positions P = 4 * len(refinement_levels)
  # (construct_initial_causal_state is called with num_resolutions =
  # len(query_features.resolutions) - 1 = len(refinement_levels).)
  refinement_levels = model_utils.generate_default_resolutions(
      (resolution, resolution), (256, 256)
  )
  num_levels = 1 + len(refinement_levels)
  num_positions = 4 * len(refinement_levels)

  print(f'Exporting online TAPIR to ONNX:')
  print(f'  checkpoint     : {args.checkpoint}')
  print(f'  output         : {args.output}')
  print(f'  resolution (R) : {resolution}x{resolution}')
  print(f'  point capacity : {num_points}')
  print(f'  refinement     : {refinement_levels}')
  print(f'  query levels L : {num_levels}, causal positions P: {num_positions}')

  model = build_model(args.checkpoint, device)
  wrapper = OnlineTAPIRWrapper(model, num_points).to(device).eval()

  rng = np.random.default_rng(0)
  dummy_inputs = make_dummy_inputs(
      resolution, num_points, num_positions, num_levels,
      model.num_mixer_blocks, device, rng,
  )

  output_dir = os.path.dirname(os.path.abspath(args.output))
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)

  print('Tracing and exporting (this may take a minute)...')
  try:
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        args.output,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
  except TypeError:
    # Older torch without the `dynamo` kwarg.
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        args.output,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=args.opset,
        do_constant_folding=True,
    )

  # Attach self-describing metadata and validate the graph.
  import onnx

  onnx_model = onnx.load(args.output)
  metadata = {
      'num_points': str(num_points),
      'resolution': str(resolution),
      'num_query_levels': str(num_levels),
      'causal_positions': str(num_positions),
      'num_mixer_blocks': str(model.num_mixer_blocks),
      'io_names': ','.join(INPUT_NAMES + OUTPUT_NAMES),
  }
  for key, value in metadata.items():
    entry = onnx_model.metadata_props.add()
    entry.key = key
    entry.value = value
  onnx.save(onnx_model, args.output)
  onnx.checker.check_model(onnx_model)
  size_mb = os.path.getsize(args.output) / 1e6
  print(f'Exported {args.output} ({size_mb:.1f} MB), graph check passed.')

  if args.no_verify:
    print('Verification skipped (--no_verify).')
    return

  print('Verifying against PyTorch with onnxruntime (init + 2 steps)...')
  if verify(args.output, wrapper, dummy_inputs, device):
    print('Verification passed: ONNX matches PyTorch.')
  else:
    print(
        'WARNING: verification found differences > 0.05. Small drift can come '
        'from op reordering; inspect the [DRIFT]/[FAIL] entries above.'
    )


if __name__ == '__main__':
  main()
