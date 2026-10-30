#!/usr/bin/env python3
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

"""Full benchmark for Online TAPIR.

Methods:
- torch: PyTorch causal online TAPIR, run from a .pt checkpoint.
- tensorrt: TensorRT online TAPIR engine (OnlineTapirTRT from trt_video.py).

The model is STATEFUL: each run anchors once on query points (init is excluded
from timing), then steps frames through the online loop. This measures the
steady-state per-frame latency (avg ms per step, FPS) and the peak GPU memory
delta.

Discovers checkpoints / engines dynamically from ./checkpoints by filename
convention:
- torch checkpoints:  <model>_checkpoint.pt
- TRT engines:        <model>_checkpoint[_<variant>]_<dtype>.engine
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
import contextlib
import glob
import io
import re
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import tree

from tapnet.torch import tapir_model
from tapnet.trt_video import OnlineTapirTRT

# ---------- Configuration ----------
FRAME_H = 480
FRAME_W = 640
WARMUP = 10
TEST = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = './checkpoints'

# Only models with a torch configuration can be benchmarked in torch mode.
MODEL_CONFIGS = {
    'causal_bootstapir': dict(pyramid_level=1, use_casual_conv=True),
}

# Filename conventions:
#   torch:  <model>_checkpoint.pt                (e.g. causal_bootstapir_checkpoint.pt)
#   engine: <model>_checkpoint[_<variant>]_<dtype>.engine
#           (e.g. causal_bootstapir_checkpoint_fp16.engine -> variant=default;
#            causal_bootstapir_checkpoint_orin_fp16.engine  -> variant=orin)
_ENGINE_RE = re.compile(
    r'^(?P<model>[a-z0-9_]+?)_checkpoint'
    r'(?:_(?P<variant>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*?))?'
    r'_(?P<dtype>fp16|fp32|int8)\.engine$'
)
_CHECKPOINT_RE = re.compile(r'^(?P<model>[a-z0-9_]+?)_checkpoint\.pt$')


# ---------- Dynamic discovery ----------
def discover_engines(checkpoint_dir, dtypes=None, variants=None, models=None):
    """Return list of dicts: {path, model, variant, dtype}. Only existing files."""
    found = []
    for path in sorted(glob.glob(os.path.join(checkpoint_dir, '*.engine'))):
        m = _ENGINE_RE.match(Path(path).name)
        if not m:
            print(f"  [skip] Unrecognised engine name: {Path(path).name}")
            continue
        info = m.groupdict()
        info['variant'] = info['variant'] or 'default'
        info['path'] = path
        if dtypes and info['dtype'] not in dtypes:
            continue
        if variants and info['variant'] not in variants:
            continue
        if models and info['model'] not in models:
            continue
        found.append(info)
    return found


def discover_checkpoints(checkpoint_dir, models=None):
    """Return dict {model: path} for available .pt files."""
    found = {}
    for path in sorted(glob.glob(os.path.join(checkpoint_dir, '*.pt'))):
        m = _CHECKPOINT_RE.match(Path(path).name)
        if not m:
            print(f"  [skip] Unrecognised checkpoint name: {Path(path).name}")
            continue
        model_name = m.group('model')
        if models and model_name not in models:
            continue
        found[model_name] = path
    return found


# ---------- Preprocessing / helpers ----------
def get_gpu_memory():
    free, total = torch.cuda.mem_get_info()
    return (total - free) // (1024 * 1024)


def generate_frame(height=FRAME_H, width=FRAME_W):
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def preprocess(frame, resolution):
    """Resize to R x R (INTER_AREA) and scale to [-1, 1], np.float32."""
    resized = cv2.resize(frame, (resolution, resolution),
                         interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255 * 2 - 1


def sample_query_points(num_points, resolution):
    """Random (t=0, y, x) query points, uniform in [0, resolution)."""
    qp = np.random.uniform(0.0, resolution, (num_points, 3)).astype(np.float32)
    qp[:, 0] = 0.0
    return qp


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


# ---------- Benchmarking ----------
def benchmark_torch(model_name, checkpoint_path, num_points, resolution,
                    warmup=WARMUP, test=TEST):
    print(f"Loading torch model {model_name} from {checkpoint_path}...")
    model = tapir_model.TAPIR(**MODEL_CONFIGS[model_name])
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    torch.set_grad_enabled(False)

    baseline = get_gpu_memory()  # after model load, like the reference
    print(f"  N={num_points} points, R={resolution} resolution")

    # Anchor once: extract query features from a random frame (excluded from
    # timing), then roll the causal state forward per step.
    qp = torch.tensor(sample_query_points(num_points, resolution)).to(DEVICE)
    frame0 = torch.tensor(preprocess(generate_frame(), resolution))[None, None].to(DEVICE)
    with _suppress_stdout():
        feature_grids = model.get_feature_grids(frame0, is_training=False)
        features = model.get_query_features(
            frame0, is_training=False, query_points=qp[None, :],
            feature_grids=feature_grids,
        )
    causal_state = model.construct_initial_causal_state(
        num_points, len(features.resolutions) - 1)
    causal_state = tree.map_structure(lambda x: x.to(DEVICE), causal_state)

    for _ in range(warmup):
        frame_t = torch.tensor(preprocess(generate_frame(), resolution))[None, None].to(DEVICE)
        with _suppress_stdout():
            feature_grids = model.get_feature_grids(frame_t, is_training=False)
            trajs = model.estimate_trajectories(
                frame_t.shape[-3:-1], is_training=False,
                feature_grids=feature_grids, query_features=features,
                query_points_in_video=None, query_chunk_size=64,
                causal_context=causal_state, get_causal_context=True)
            causal_state = trajs['causal_context']

    times, mems = [], []
    for _ in range(test):
        frame_t = torch.tensor(preprocess(generate_frame(), resolution))[None, None].to(DEVICE)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with _suppress_stdout():
            feature_grids = model.get_feature_grids(frame_t, is_training=False)
            trajs = model.estimate_trajectories(
                frame_t.shape[-3:-1], is_training=False,
                feature_grids=feature_grids, query_features=features,
                query_points_in_video=None, query_chunk_size=64,
                causal_context=causal_state, get_causal_context=True)
            causal_state = trajs['causal_context']
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        mems.append(get_gpu_memory())

    avg_ms = float(np.mean(times))
    return avg_ms, 1000.0 / avg_ms, float(max(mems) - baseline)


def benchmark_trt(engine_path, num_points_torch, warmup=WARMUP, test=TEST):
    print(f"Loading TRT engine {engine_path}...")
    runner = OnlineTapirTRT(engine_path)
    N = runner.num_points  # point capacity is fixed by the engine
    R = runner.resolution

    if N != num_points_torch:
        print(f"  Note: engine point capacity {N} != torch {num_points_torch} "
              f"query points; the runs are not strictly comparable.")
    print(f"  N={N} points, R={R} resolution")

    baseline = get_gpu_memory()  # after engine load, like the reference

    qp = sample_query_points(N, R)
    runner.init_from_frame(qp, preprocess(generate_frame(), R))

    for _ in range(warmup):
        runner.step(preprocess(generate_frame(), R))

    times, mems = [], []
    for _ in range(test):
        frame = preprocess(generate_frame(), R)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        runner.step(frame)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        mems.append(get_gpu_memory())

    avg_ms = float(np.mean(times))
    return avg_ms, 1000.0 / avg_ms, float(max(mems) - baseline)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', default=CHECKPOINT_DIR)
    parser.add_argument('--models', nargs='+', default=None,
                        help='Filter models (torch checkpoints and TRT engines)')
    parser.add_argument('--dtypes', nargs='+', default=None,
                        choices=['fp16', 'fp32', 'int8'],
                        help='Filter engine dtypes')
    parser.add_argument('--variants', nargs='+', default=None,
                        help='Filter engine variants, e.g. default orin')
    parser.add_argument('--skip-torch', action='store_true')
    parser.add_argument('--skip-trt', action='store_true')
    parser.add_argument('--num_points', type=int, default=64,
                        help='Query points for torch runs '
                             '(TRT runs use the engine point capacity)')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Square input resolution for torch runs '
                             '(must be a multiple of 8)')
    parser.add_argument('--warmup', type=int, default=WARMUP)
    parser.add_argument('--test', type=int, default=TEST)
    args = parser.parse_args()

    if not args.skip_torch and args.resolution % 8 != 0:
        raise ValueError("--resolution must be a multiple of 8")

    # torch spam-floods this warning per frame per resolution and it buries
    # everything else; see pytorch_video.py's main().
    warnings.filterwarnings(
        "ignore", message=".*output size is not a multiple of 8.*"
    )

    # Discover
    print(f"Scanning {args.checkpoint_dir} ...")
    checkpoints = discover_checkpoints(args.checkpoint_dir, args.models)
    engines = discover_engines(args.checkpoint_dir, args.dtypes, args.variants,
                               args.models)

    print(f"  Found {len(checkpoints)} torch checkpoint(s): {sorted(checkpoints)}")
    print(f"  Found {len(engines)} TRT engine(s):")
    for e in engines:
        print(f"    - {Path(e['path']).name}  (model={e['model']}, "
              f"variant={e['variant']}, dtype={e['dtype']})")

    results, skipped = [], []

    # 1. Torch
    if not args.skip_torch:
        for model_name, path in sorted(checkpoints.items()):
            if model_name not in MODEL_CONFIGS:
                skipped.append((model_name, 'torch', '-', '-', 'no torch config'))
                continue
            print(f"\n=== Benchmarking torch + {model_name} ===")
            try:
                t, fps, m = benchmark_torch(model_name, path, args.num_points,
                                            args.resolution, args.warmup, args.test)
                results.append(dict(model=model_name, method='torch', dtype='-',
                                    variant='-', avg_time_ms=t, fps=fps,
                                    peak_mem_mb=m))
            except Exception as ex:
                skipped.append((model_name, 'torch', '-', '-', f'error: {ex}'))

    # 2. TensorRT — one row per discovered engine
    if not args.skip_trt:
        for e in engines:
            print(f"\n=== Benchmarking tensorrt + {e['model']} + "
                  f"{e['variant']} + {e['dtype']} ===")
            try:
                t, fps, m = benchmark_trt(e['path'], args.num_points,
                                          args.warmup, args.test)
                results.append(dict(model=e['model'], method='tensorrt',
                                    dtype=e['dtype'], variant=e['variant'],
                                    avg_time_ms=t, fps=fps, peak_mem_mb=m))
            except Exception as ex:
                skipped.append((e['model'], 'tensorrt', e['dtype'],
                                e['variant'], f'error: {ex}'))

    # Summary
    print("\n\n========== Summary ==========")
    hdr = (f"{'Model':<18} {'Method':<10} {'Dtype':<6} {'Variant':<10} "
           f"{'Avg Time (ms)':<15} {'FPS':<8} {'Peak Mem (MB)':<15}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['model']:<18} {r['method']:<10} {r['dtype']:<6} "
              f"{r['variant']:<10} {r['avg_time_ms']:<15.2f} "
              f"{r['fps']:<8.1f} {r['peak_mem_mb']:<15.2f}")

    if skipped:
        print("\nSkipped:")
        for s in skipped:
            print(f"  {s[0]:<18} {s[1]:<10} dtype={s[2]:<6} "
                  f"variant={s[3]:<10} ({s[4]})")
    print("==============================\n")


if __name__ == '__main__':
    main()