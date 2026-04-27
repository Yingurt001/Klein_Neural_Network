"""Isolated BN microbenchmark (Table D).

Measures forward + backward wall-clock of four BN variants at varying
batch sizes and dimensions. Runs 1000 iterations per config (100 warmup +
1000 measurement) on a single GPU with torch.cuda.synchronize() before
each timer.

Output: CSV + markdown table written to `bn_microbench_results.{csv,md}`.

Usage:
    python bn_microbench.py [--cuda 0]
    sbatch scripts/bn_ablation/submit_microbench.sh   # on Ada
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# ensure local klein_nn is importable
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from klein_nn.manifold import KleinManifold
from klein_nn.layers.klein_bn import (
    KleinEinsteinBN, KleinTangentBN, KleinFrechetBNViaPoincare,
    KleinLorentzStyleBN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_klein_points(B: int, D: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(0)
    radius = 0.9 / 1.0  # K=-1 → 1/sqrt(-K) = 1
    x = torch.randn(B, D, dtype=dtype, device=device) * 0.25
    x = KleinManifold.projx(x, K=-1.0)
    n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    x = torch.where(n > radius, x / n * radius, x)
    return x


def bench_one(bn: nn.Module, x: torch.Tensor, n_warm: int, n_iter: int,
              measure_backward: bool = True) -> tuple[float, float, float]:
    """Return (fwd_ms, bwd_ms, peak_mem_mb) per iteration."""
    bn.train()
    x_req = x.clone().detach().requires_grad_(measure_backward)

    # warmup
    for _ in range(n_warm):
        y = bn(x_req)
        if measure_backward:
            loss = y.pow(2).sum()
            loss.backward()
            x_req.grad = None
            for p in bn.parameters():
                if p.grad is not None:
                    p.grad = None

    # reset CUDA memory stats before measurement
    if x.is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device=x.device)

    # measure forward
    t0 = time.perf_counter()
    ys = []
    for _ in range(n_iter):
        y = bn(x_req)
        ys.append(y)
    if x.is_cuda:
        torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1000 / n_iter

    # measure backward
    if measure_backward:
        t0 = time.perf_counter()
        for y in ys:
            loss = y.pow(2).sum()
            loss.backward(retain_graph=False)
            x_req.grad = None
            for p in bn.parameters():
                if p.grad is not None:
                    p.grad = None
        if x.is_cuda:
            torch.cuda.synchronize()
        bwd_ms = (time.perf_counter() - t0) * 1000 / n_iter
    else:
        bwd_ms = 0.0

    # peak memory (MB) during the benchmark
    if x.is_cuda:
        peak_mem_mb = torch.cuda.max_memory_allocated(device=x.device) / (1024 * 1024)
    else:
        peak_mem_mb = 0.0

    return fwd_ms, bwd_ms, peak_mem_mb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0, help='GPU index, -1 for CPU')
    parser.add_argument('--n_warm', type=int, default=50)
    parser.add_argument('--n_iter', type=int, default=200,
                        help='fewer iters for Fréchet to keep total time bounded')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float64'])
    parser.add_argument('--out_csv', type=str, default='bn_microbench_results.csv')
    parser.add_argument('--out_md',  type=str, default='bn_microbench_results.md')
    args = parser.parse_args()

    if args.cuda >= 0 and torch.cuda.is_available():
        device = f'cuda:{args.cuda}'
    else:
        device = 'cpu'
    dtype = torch.float32 if args.dtype == 'float32' else torch.float64

    # grid (extended to larger batches to show Fréchet's scaling)
    Bs = [64, 512, 4096, 8192, 16384]
    Ds = [64, 128, 256]

    # BN variants to benchmark
    def make_variants(D: int) -> dict[str, nn.Module]:
        return {
            'TangentBN':     KleinTangentBN(D, K=-1.0).to(device).to(dtype),
            'KleinFrechet':  KleinFrechetBNViaPoincare(D, K=-1.0).to(device).to(dtype),
            'LorentzCentroid': KleinLorentzStyleBN(D, K=-1.0).to(device).to(dtype),
            'KleinEinstein': KleinEinsteinBN(D, K=-1.0).to(device).to(dtype),
        }

    print(f"device={device}  dtype={dtype}  n_warm={args.n_warm}  n_iter={args.n_iter}")
    print(f"{'B':>6} {'D':>6} {'Method':<18} {'Fwd (ms)':>10} {'Bwd (ms)':>10} {'Total':>10} {'Mem (MB)':>10}")
    print('-' * 76)

    rows = []
    for B in Bs:
        for D in Ds:
            x = random_klein_points(B, D, device, dtype)
            variants = make_variants(D)
            for name, bn in variants.items():
                try:
                    fwd, bwd, mem = bench_one(bn, x, args.n_warm, args.n_iter)
                    total = fwd + bwd
                    print(f"{B:>6} {D:>6} {name:<18} {fwd:>10.3f} {bwd:>10.3f} {total:>10.3f} {mem:>10.1f}")
                    rows.append({
                        'B': B, 'D': D, 'method': name,
                        'fwd_ms': fwd, 'bwd_ms': bwd, 'total_ms': total,
                        'peak_mem_mb': mem,
                    })
                except Exception as e:
                    print(f"{B:>6} {D:>6} {name:<18} FAILED: {str(e)[:60]}")
                    rows.append({
                        'B': B, 'D': D, 'method': name,
                        'fwd_ms': float('nan'), 'bwd_ms': float('nan'),
                        'total_ms': float('nan'), 'peak_mem_mb': float('nan'),
                    })
            del variants
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

    # write CSV
    out_csv = Path(args.out_csv)
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['B', 'D', 'method', 'fwd_ms', 'bwd_ms',
                                          'total_ms', 'peak_mem_mb'])
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV written to {out_csv.resolve()}")

    # write markdown table (total_ms + memory)
    def _get(B, D, method):
        return next((r for r in rows if r['B']==B and r['D']==D and r['method']==method), None)

    out_md = Path(args.out_md)
    with out_md.open('w') as f:
        f.write(f"# BN Microbench Results\n\n")
        f.write(f"- Device: `{device}` · dtype `{args.dtype}` · {args.n_iter} iters × {args.n_warm} warmup\n")
        f.write(f"- Metric: total = forward + backward, average ms per iteration; peak CUDA memory in MB\n\n")

        f.write(f"## Total wall-clock (fwd + bwd) ms\n\n")
        f.write(f"| B | D | TgBN | V3 Fréchet | V4 Lorentz | **V5 Einstein (ours)** | V3/V5 speedup |\n")
        f.write(f"|---|---|:----:|:----------:|:----------:|:---------------------:|:-------------:|\n")
        for B in Bs:
            for D in Ds:
                tg, fr, lz, ei = _get(B,D,'TangentBN'), _get(B,D,'KleinFrechet'), _get(B,D,'LorentzCentroid'), _get(B,D,'KleinEinstein')
                if all(x is not None for x in [tg, fr, lz, ei]):
                    ratio = fr['total_ms'] / ei['total_ms'] if ei['total_ms'] > 0 else float('nan')
                    f.write(f"| {B} | {D} | {tg['total_ms']:.3f} | {fr['total_ms']:.3f} | {lz['total_ms']:.3f} | **{ei['total_ms']:.3f}** | {ratio:.2f}× |\n")

        f.write(f"\n## Peak GPU memory (MB)\n\n")
        f.write(f"| B | D | TgBN | V3 Fréchet | V4 Lorentz | **V5 Einstein (ours)** |\n")
        f.write(f"|---|---|:----:|:----------:|:----------:|:---------------------:|\n")
        for B in Bs:
            for D in Ds:
                tg, fr, lz, ei = _get(B,D,'TangentBN'), _get(B,D,'KleinFrechet'), _get(B,D,'LorentzCentroid'), _get(B,D,'KleinEinstein')
                if all(x is not None for x in [tg, fr, lz, ei]):
                    f.write(f"| {B} | {D} | {tg['peak_mem_mb']:.1f} | {fr['peak_mem_mb']:.1f} | {lz['peak_mem_mb']:.1f} | **{ei['peak_mem_mb']:.1f}** |\n")

    print(f"Markdown written to {out_md.resolve()}")


if __name__ == '__main__':
    main()
