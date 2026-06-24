#!/usr/bin/env python3
"""
Reproducible benchmark harness for the indexed_logits CUDA extension.

This script is designed for paper-facing experiments. It benchmarks the
implementation that actually exists in this repository:
  - fused indexed logits CUDA kernel
  - dense full-logits baseline (H @ W.T followed by gather)
  - naive subset baseline (W[idx] materialization followed by reduction)

It measures:
  - forward correctness against the naive subset reference
  - backward correctness against autograd on the naive subset reference
  - forward latency
  - end-to-end forward+backward latency
  - peak allocated / reserved CUDA memory
  - estimated throughput in TFLOP/s
  - sensitivity to index collisions

Outputs:
  - human-readable tables on stdout
  - optional JSON and CSV files for paper tables/plots
"""

import argparse
import csv
import gc
import json
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from indexed_logits import indexed_logits, indexed_logits_backward, indexed_logits_forward

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass
class Config:
    name: str
    N: int
    d: int
    V: int
    k: int
    dtype: str


@dataclass
class Result:
    config: str
    method: str
    phase: str
    status: str
    seed: int
    N: int
    d: int
    V: int
    k: int
    dtype: str
    collision_vocab: Optional[int]
    time_ms: Optional[float]
    peak_alloc_mb: Optional[float]
    peak_reserved_mb: Optional[float]
    tflops_est: Optional[float]
    max_abs_err: Optional[float]
    mean_abs_err: Optional[float]


@dataclass
class SummaryResult:
    config: str
    method: str
    phase: str
    status: str
    N: int
    d: int
    V: int
    k: int
    dtype: str
    collision_vocab: Optional[int]
    num_seeds: int
    time_ms_mean: Optional[float]
    time_ms_std: Optional[float]
    peak_alloc_mb_mean: Optional[float]
    peak_alloc_mb_std: Optional[float]
    peak_reserved_mb_mean: Optional[float]
    peak_reserved_mb_std: Optional[float]
    tflops_est_mean: Optional[float]
    tflops_est_std: Optional[float]
    max_abs_err_mean: Optional[float]
    max_abs_err_std: Optional[float]
    mean_abs_err_mean: Optional[float]
    mean_abs_err_std: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark indexed_logits for paper experiments.")
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP), default="fp16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds. Overrides --seed if provided.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-summary-json", type=Path, default=None)
    parser.add_argument("--output-summary-csv", type=Path, default=None)
    parser.add_argument(
        "--suite",
        choices=["quick", "paper", "collisions", "extended"],
        default="paper",
        help="Preset benchmark suite.",
    )
    parser.add_argument(
        "--allow-dense-oom",
        action="store_true",
        help="Attempt dense baseline even when the memory estimate looks too large.",
    )
    return parser.parse_args()


def get_presets(dtype: str, suite: str) -> list[Config]:
    if suite == "quick":
        return [
            Config("quick_small", N=1024, d=512, V=50257, k=32, dtype=dtype),
            Config("quick_medium", N=4096, d=1024, V=50257, k=64, dtype=dtype),
        ]
    if suite == "collisions":
        return [
            Config("collision_probe", N=4096, d=1024, V=50257, k=64, dtype=dtype),
        ]
    if suite == "extended":
        return [
            Config("sweep_k16", N=4096, d=768, V=50257, k=16, dtype=dtype),
            Config("sweep_k32", N=4096, d=768, V=50257, k=32, dtype=dtype),
            Config("sweep_k64", N=4096, d=768, V=50257, k=64, dtype=dtype),
            Config("sweep_k128", N=4096, d=768, V=50257, k=128, dtype=dtype),
            Config("sweep_k256", N=4096, d=768, V=50257, k=256, dtype=dtype),
            Config("sweep_k512", N=4096, d=768, V=50257, k=512, dtype=dtype),
            Config("sweep_N1024", N=1024, d=768, V=50257, k=128, dtype=dtype),
            Config("sweep_N2048", N=2048, d=768, V=50257, k=128, dtype=dtype),
            Config("sweep_N4096", N=4096, d=768, V=50257, k=128, dtype=dtype),
            Config("sweep_N8192", N=8192, d=768, V=50257, k=128, dtype=dtype),
            Config("sweep_d768", N=4096, d=768, V=50257, k=128, dtype=dtype),
            Config("sweep_d1024", N=4096, d=1024, V=50257, k=128, dtype=dtype),
            Config("sweep_d1536", N=4096, d=1536, V=50257, k=128, dtype=dtype),
            Config("sweep_d3072", N=4096, d=3072, V=50257, k=128, dtype=dtype),
        ]
    return [
        Config("gpt2_like_smallk", N=4096, d=768, V=50257, k=32, dtype=dtype),
        Config("gpt2_like_midk", N=4096, d=768, V=50257, k=128, dtype=dtype),
        Config("gpt2_like_largek", N=4096, d=768, V=50257, k=256, dtype=dtype),
        Config("scaling_hidden", N=4096, d=1536, V=50257, k=128, dtype=dtype),
        Config("scaling_batch", N=8192, d=768, V=50257, k=128, dtype=dtype),
    ]


def dtype_size_bytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"Unsupported dtype: {dtype}")


def theoretical_naive_subset_bytes(cfg: Config, dtype: torch.dtype) -> int:
    return cfg.N * cfg.k * cfg.d * dtype_size_bytes(dtype)


def theoretical_dense_logits_bytes(cfg: Config, dtype: torch.dtype) -> int:
    return cfg.N * cfg.V * dtype_size_bytes(dtype)


def estimate_dense_feasible(cfg: Config, dtype: torch.dtype) -> bool:
    total_mem = torch.cuda.get_device_properties(0).total_memory
    dense_logits = theoretical_dense_logits_bytes(cfg, dtype)
    # Conservative multiplier for forward+backward activations and gradients.
    estimated = dense_logits * 3 + cfg.N * cfg.d * dtype_size_bytes(dtype) + cfg.V * cfg.d * dtype_size_bytes(dtype)
    return estimated < total_mem * 0.80


def reset_cuda_state() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def make_inputs(cfg: Config, collision_vocab: Optional[int], requires_grad: bool) -> dict[str, torch.Tensor]:
    dtype = DTYPE_MAP[cfg.dtype]
    device = "cuda"

    H = torch.randn(cfg.N, cfg.d, dtype=dtype, device=device, requires_grad=requires_grad)
    W = torch.randn(cfg.V, cfg.d, dtype=dtype, device=device, requires_grad=requires_grad)

    vocab = cfg.V if collision_vocab is None else min(collision_vocab, cfg.V)
    idx = torch.randint(0, vocab, (cfg.N, cfg.k), dtype=torch.int32, device=device)
    return {"H": H, "W": W, "idx": idx}


def reference_forward(H: torch.Tensor, W: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return (H.unsqueeze(1) * W[idx]).sum(dim=-1)


def max_and_mean_abs_err(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    return {
        "max_abs_err": diff.max().item(),
        "mean_abs_err": diff.mean().item(),
    }


def benchmark_forward(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        out = fn()
        del out
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
        del out
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / iters
    return {
        "time_ms": elapsed_ms,
        "peak_alloc_mb": torch.cuda.max_memory_allocated() / 1e6,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
    }


def benchmark_fwd_bwd(
    fn: Callable[[], torch.Tensor],
    params: Iterable[torch.Tensor],
    warmup: int,
    iters: int,
) -> dict[str, float]:
    params = list(params)

    def zero_grads() -> None:
        for p in params:
            if p.grad is not None:
                p.grad = None

    for _ in range(warmup):
        zero_grads()
        out = fn()
        loss = out.float().sum()
        loss.backward()
        del out, loss
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        zero_grads()
        out = fn()
        loss = out.float().sum()
        loss.backward()
        del out, loss
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / iters
    return {
        "time_ms": elapsed_ms,
        "peak_alloc_mb": torch.cuda.max_memory_allocated() / 1e6,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
    }


def benchmark_backward_only(
    H: torch.Tensor,
    W: torch.Tensor,
    idx: torch.Tensor,
    grad_out: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup):
        grad_h, grad_w = indexed_logits_backward(H, W, idx, grad_out)
        del grad_h, grad_w
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        grad_h, grad_w = indexed_logits_backward(H, W, idx, grad_out)
        del grad_h, grad_w
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / iters
    return {
        "time_ms": elapsed_ms,
        "peak_alloc_mb": torch.cuda.max_memory_allocated() / 1e6,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
    }


def forward_tflops(cfg: Config, time_ms: float) -> float:
    ops = 2.0 * cfg.N * cfg.k * cfg.d
    return ops / (time_ms * 1e-3) / 1e12


def fwd_bwd_tflops(cfg: Config, time_ms: float) -> float:
    # Forward + grad_H + grad_W are each approximately one dot/outer-product pass.
    ops = 6.0 * cfg.N * cfg.k * cfg.d
    return ops / (time_ms * 1e-3) / 1e12


def fused_forward(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return indexed_logits_forward(inputs["H"], inputs["W"], inputs["idx"])


def fused_autograd(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return indexed_logits(inputs["H"], inputs["W"], inputs["idx"])


def dense_forward(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = inputs["H"] @ inputs["W"].T
    return torch.gather(logits, 1, inputs["idx"].to(torch.int64))


def naive_subset_forward(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return reference_forward(inputs["H"], inputs["W"], inputs["idx"].to(torch.int64))


def correctness_check(cfg: Config, collision_vocab: Optional[int], seed: int) -> list[Result]:
    reset_cuda_state()
    torch.manual_seed(seed)
    inputs = make_inputs(cfg, collision_vocab=collision_vocab, requires_grad=True)

    with torch.no_grad():
        fused = fused_forward(inputs)
        ref = naive_subset_forward(inputs)
        errs = max_and_mean_abs_err(fused, ref)

    H_ref = inputs["H"].detach().clone().requires_grad_(True)
    W_ref = inputs["W"].detach().clone().requires_grad_(True)
    idx_ref = inputs["idx"].detach()
    out_ref = reference_forward(H_ref, W_ref, idx_ref.to(torch.int64))
    loss_ref = out_ref.float().sum()
    loss_ref.backward()

    out = fused_autograd(inputs)
    loss = out.float().sum()
    loss.backward()

    grad_h_err = max_and_mean_abs_err(inputs["H"].grad, H_ref.grad)
    grad_w_err = max_and_mean_abs_err(inputs["W"].grad, W_ref.grad)

    return [
        Result(
            config=cfg.name,
            method="fused",
            phase="correctness_forward",
            status="ok",
            seed=seed,
            N=cfg.N,
            d=cfg.d,
            V=cfg.V,
            k=cfg.k,
            dtype=cfg.dtype,
            collision_vocab=collision_vocab,
            time_ms=None,
            peak_alloc_mb=None,
            peak_reserved_mb=None,
            tflops_est=None,
            max_abs_err=errs["max_abs_err"],
            mean_abs_err=errs["mean_abs_err"],
        ),
        Result(
            config=cfg.name,
            method="fused",
            phase="correctness_grad_H",
            status="ok",
            seed=seed,
            N=cfg.N,
            d=cfg.d,
            V=cfg.V,
            k=cfg.k,
            dtype=cfg.dtype,
            collision_vocab=collision_vocab,
            time_ms=None,
            peak_alloc_mb=None,
            peak_reserved_mb=None,
            tflops_est=None,
            max_abs_err=grad_h_err["max_abs_err"],
            mean_abs_err=grad_h_err["mean_abs_err"],
        ),
        Result(
            config=cfg.name,
            method="fused",
            phase="correctness_grad_W",
            status="ok",
            seed=seed,
            N=cfg.N,
            d=cfg.d,
            V=cfg.V,
            k=cfg.k,
            dtype=cfg.dtype,
            collision_vocab=collision_vocab,
            time_ms=None,
            peak_alloc_mb=None,
            peak_reserved_mb=None,
            tflops_est=None,
            max_abs_err=grad_w_err["max_abs_err"],
            mean_abs_err=grad_w_err["mean_abs_err"],
        ),
    ]


def run_one_method(
    cfg: Config,
    method: str,
    phase: str,
    fn_builder: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    warmup: int,
    iters: int,
    collision_vocab: Optional[int],
    seed: int,
) -> Result:
    reset_cuda_state()
    torch.manual_seed(seed)
    requires_grad = phase == "fwd_bwd"
    inputs = make_inputs(cfg, collision_vocab=collision_vocab, requires_grad=requires_grad)

    try:
        if phase == "forward":
            metrics = benchmark_forward(lambda: fn_builder(inputs), warmup=warmup, iters=iters)
            tflops = forward_tflops(cfg, metrics["time_ms"])
        else:
            metrics = benchmark_fwd_bwd(
                lambda: fn_builder(inputs),
                params=(inputs["H"], inputs["W"]),
                warmup=warmup,
                iters=iters,
            )
            tflops = fwd_bwd_tflops(cfg, metrics["time_ms"])

        return Result(
            config=cfg.name,
            method=method,
            phase=phase,
            status="ok",
            seed=seed,
            N=cfg.N,
            d=cfg.d,
            V=cfg.V,
            k=cfg.k,
            dtype=cfg.dtype,
            collision_vocab=collision_vocab,
            time_ms=metrics["time_ms"],
            peak_alloc_mb=metrics["peak_alloc_mb"],
            peak_reserved_mb=metrics["peak_reserved_mb"],
            tflops_est=tflops,
            max_abs_err=None,
            mean_abs_err=None,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            status = "oom"
        else:
            status = f"error:{type(exc).__name__}"
        return Result(
            config=cfg.name,
            method=method,
            phase=phase,
            status=status,
            seed=seed,
            N=cfg.N,
            d=cfg.d,
            V=cfg.V,
            k=cfg.k,
            dtype=cfg.dtype,
            collision_vocab=collision_vocab,
            time_ms=None,
            peak_alloc_mb=None,
            peak_reserved_mb=None,
            tflops_est=None,
            max_abs_err=None,
            mean_abs_err=None,
        )


def run_backward_only(
    cfg: Config,
    warmup: int,
    iters: int,
    collision_vocab: Optional[int],
    seed: int,
) -> Result:
    reset_cuda_state()
    torch.manual_seed(seed)
    inputs = make_inputs(cfg, collision_vocab=collision_vocab, requires_grad=False)
    grad_out = torch.randn(cfg.N, cfg.k, dtype=DTYPE_MAP[cfg.dtype], device="cuda")

    try:
        metrics = benchmark_backward_only(
            inputs["H"],
            inputs["W"],
            inputs["idx"],
            grad_out,
            warmup=warmup,
            iters=iters,
        )
        tflops = fwd_bwd_tflops(cfg, metrics["time_ms"]) / 2.0
        return Result(
            config=cfg.name,
            method="fused",
            phase="backward_only",
            status="ok",
            seed=seed,
            N=cfg.N,
            d=cfg.d,
            V=cfg.V,
            k=cfg.k,
            dtype=cfg.dtype,
            collision_vocab=collision_vocab,
            time_ms=metrics["time_ms"],
            peak_alloc_mb=metrics["peak_alloc_mb"],
            peak_reserved_mb=metrics["peak_reserved_mb"],
            tflops_est=tflops,
            max_abs_err=None,
            mean_abs_err=None,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            status = "oom"
        else:
            status = f"error:{type(exc).__name__}"
        return Result(
            config=cfg.name,
            method="fused",
            phase="backward_only",
            status=status,
            seed=seed,
            N=cfg.N,
            d=cfg.d,
            V=cfg.V,
            k=cfg.k,
            dtype=cfg.dtype,
            collision_vocab=collision_vocab,
            time_ms=None,
            peak_alloc_mb=None,
            peak_reserved_mb=None,
            tflops_est=None,
            max_abs_err=None,
            mean_abs_err=None,
        )


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def print_results(results: list[Result]) -> None:
    current_key = None
    for result in results:
        key = (result.config, result.collision_vocab)
        if key != current_key:
            current_key = key
            suffix = "" if result.collision_vocab is None else f", collision_vocab={result.collision_vocab}"
            print_section(
                f"{result.config}: N={result.N}, d={result.d}, V={result.V}, k={result.k}, dtype={result.dtype}{suffix}"
            )
            print(f"{'seed':>4} {'method':<14} {'phase':<18} {'status':<12} {'time_ms':>10} {'alloc_mb':>10} {'tflops':>10} {'max_abs_err':>12}")

        time_ms = "-" if result.time_ms is None else f"{result.time_ms:10.3f}"
        alloc_mb = "-" if result.peak_alloc_mb is None else f"{result.peak_alloc_mb:10.1f}"
        tflops = "-" if result.tflops_est is None else f"{result.tflops_est:10.3f}"
        max_abs_err = "-" if result.max_abs_err is None else f"{result.max_abs_err:12.3e}"
        print(
            f"{result.seed:4d} {result.method:<14} {result.phase:<18} {result.status:<12} {time_ms} {alloc_mb} {tflops} {max_abs_err}"
        )


def metric_stats(values: list[Optional[float]]) -> tuple[Optional[float], Optional[float]]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def summarize_results(results: list[Result]) -> list[SummaryResult]:
    grouped: dict[tuple[str, str, str, str, int, int, int, int, str, Optional[int]], list[Result]] = {}
    for result in results:
        key = (
            result.config,
            result.method,
            result.phase,
            result.status,
            result.N,
            result.d,
            result.V,
            result.k,
            result.dtype,
            result.collision_vocab,
        )
        grouped.setdefault(key, []).append(result)

    summaries: list[SummaryResult] = []
    for key, group in grouped.items():
        time_mean, time_std = metric_stats([g.time_ms for g in group])
        alloc_mean, alloc_std = metric_stats([g.peak_alloc_mb for g in group])
        reserved_mean, reserved_std = metric_stats([g.peak_reserved_mb for g in group])
        tflops_mean, tflops_std = metric_stats([g.tflops_est for g in group])
        max_err_mean, max_err_std = metric_stats([g.max_abs_err for g in group])
        mean_err_mean, mean_err_std = metric_stats([g.mean_abs_err for g in group])
        summaries.append(
            SummaryResult(
                config=key[0],
                method=key[1],
                phase=key[2],
                status=key[3],
                N=key[4],
                d=key[5],
                V=key[6],
                k=key[7],
                dtype=key[8],
                collision_vocab=key[9],
                num_seeds=len(group),
                time_ms_mean=time_mean,
                time_ms_std=time_std,
                peak_alloc_mb_mean=alloc_mean,
                peak_alloc_mb_std=alloc_std,
                peak_reserved_mb_mean=reserved_mean,
                peak_reserved_mb_std=reserved_std,
                tflops_est_mean=tflops_mean,
                tflops_est_std=tflops_std,
                max_abs_err_mean=max_err_mean,
                max_abs_err_std=max_err_std,
                mean_abs_err_mean=mean_err_mean,
                mean_abs_err_std=mean_err_std,
            )
        )
    summaries.sort(key=lambda x: (x.config, -1 if x.collision_vocab is None else x.collision_vocab, x.method, x.phase, x.status))
    return summaries


def save_outputs(
    results: list[Result],
    summaries: list[SummaryResult],
    json_path: Optional[Path],
    csv_path: Optional[Path],
    summary_json_path: Optional[Path],
    summary_csv_path: Optional[Path],
) -> None:
    rows = [asdict(result) for result in results]
    if json_path is not None:
        json_path.write_text(json.dumps(rows, indent=2))
    if csv_path is not None:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary_rows = [asdict(summary) for summary in summaries]
    if summary_json_path is not None:
        summary_json_path.write_text(json.dumps(summary_rows, indent=2))
    if summary_csv_path is not None:
        with summary_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    dtype = DTYPE_MAP[args.dtype]
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("Current GPU does not support bfloat16.")

    seeds = [args.seed] if args.seeds is None else [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    presets = get_presets(args.dtype, args.suite)
    results: list[Result] = []

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Suite: {args.suite}")
    print(f"Warmup: {args.warmup}, Iters: {args.iters}")
    print(f"Seeds: {seeds}")

    for cfg in presets:
        dense_ok = args.allow_dense_oom or estimate_dense_feasible(cfg, dtype)
        for seed in seeds:
            results.extend(correctness_check(cfg, collision_vocab=None, seed=seed))

            results.append(
                run_one_method(cfg, "fused", "forward", fused_forward, args.warmup, args.iters, collision_vocab=None, seed=seed)
            )
            results.append(
                run_one_method(cfg, "fused", "fwd_bwd", fused_autograd, args.warmup, args.iters, collision_vocab=None, seed=seed)
            )

            results.append(
                run_one_method(
                    cfg,
                    "naive_subset",
                    "forward",
                    naive_subset_forward,
                    args.warmup,
                    args.iters,
                    collision_vocab=None,
                    seed=seed,
                )
            )
            results.append(
                run_one_method(
                    cfg,
                    "naive_subset",
                    "fwd_bwd",
                    naive_subset_forward,
                    args.warmup,
                    args.iters,
                    collision_vocab=None,
                    seed=seed,
                )
            )

            if dense_ok:
                results.append(
                    run_one_method(cfg, "dense_gemm", "forward", dense_forward, args.warmup, args.iters, collision_vocab=None, seed=seed)
                )
                results.append(
                    run_one_method(cfg, "dense_gemm", "fwd_bwd", dense_forward, args.warmup, args.iters, collision_vocab=None, seed=seed)
                )
            else:
                results.append(
                    Result(
                        config=cfg.name,
                        method="dense_gemm",
                        phase="forward",
                        status="skipped_est_oom",
                        seed=seed,
                        N=cfg.N,
                        d=cfg.d,
                        V=cfg.V,
                        k=cfg.k,
                        dtype=cfg.dtype,
                        collision_vocab=None,
                        time_ms=None,
                        peak_alloc_mb=None,
                        peak_reserved_mb=None,
                        tflops_est=None,
                        max_abs_err=None,
                        mean_abs_err=None,
                    )
                )
                results.append(
                    Result(
                        config=cfg.name,
                        method="dense_gemm",
                        phase="fwd_bwd",
                        status="skipped_est_oom",
                        seed=seed,
                        N=cfg.N,
                        d=cfg.d,
                        V=cfg.V,
                        k=cfg.k,
                        dtype=cfg.dtype,
                        collision_vocab=None,
                        time_ms=None,
                        peak_alloc_mb=None,
                        peak_reserved_mb=None,
                        tflops_est=None,
                        max_abs_err=None,
                        mean_abs_err=None,
                    )
                )

            if args.suite in ("paper", "collisions", "extended"):
                for collision_vocab in (cfg.V, 4096, 512, 64):
                    results.append(
                        run_one_method(
                            cfg,
                            "fused",
                            "forward",
                            fused_forward,
                            args.warmup,
                            args.iters,
                            collision_vocab=collision_vocab,
                            seed=seed,
                        )
                    )
                    results.append(
                        run_backward_only(
                            cfg,
                            args.warmup,
                            args.iters,
                            collision_vocab=collision_vocab,
                            seed=seed,
                        )
                    )
                    results.append(
                        run_one_method(
                            cfg,
                            "fused",
                            "fwd_bwd",
                            fused_autograd,
                            args.warmup,
                            args.iters,
                            collision_vocab=collision_vocab,
                            seed=seed,
                        )
                    )

    print_results(results)
    summaries = summarize_results(results)
    save_outputs(
        results,
        summaries,
        args.output_json,
        args.output_csv,
        args.output_summary_json,
        args.output_summary_csv,
    )


if __name__ == "__main__":
    main()
