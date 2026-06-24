# Indexed Logits Paper Experiment Plan

This document is the paper-facing experiment design for the code currently in this repository.

## 1. What the current repo can and cannot support

The appendix text overstates the implementation relative to the code in this repo.

Claims the current code can support:
- Fused forward indexed-logit computation that avoids materializing `W[idx]` as `[N, k, d]`
- Custom backward for `grad_H` and `grad_W`
- FP32 accumulation in forward and in `grad_W`
- Memory comparisons against naive subset materialization and dense full-logit baselines
- Runtime measurements for forward and forward+backward
- Collision-sensitivity measurements for the atomic `grad_W` path
- Correctness comparisons against PyTorch reference implementations

Claims the current code cannot support without more implementation work:
- LoRA forward/backward support
- A fused LoRA kernel
- Triton fallback
- Pure PyTorch fallback achieving a fixed fraction of CUDA-kernel performance
- Optimized-forward claims about tiling, shared-memory staging, or high-efficiency coalescing beyond the current straightforward loop kernel
- A100-specific numbers unless the experiments are run on an A100

## 2. Minimum paper corrections before reporting results

The appendix should be revised to match the implementation:
- Replace “LoRA extension” with “future work” unless a LoRA kernel is added and benchmarked.
- Replace “performance optimization techniques” with “correctness-first CUDA implementation”.
- Remove claims about shared memory, register-resident LoRA intermediates, Triton fallbacks, and occupancy saturation unless those paths are implemented and measured.
- Reframe the benchmark section around empirical comparison of the current kernel versus PyTorch baselines.

## 3. Core experiments to run

Use `benchmark_paper.py` for the main experiments. It benchmarks:
- `fused`: this repo’s CUDA extension
- `naive_subset`: `W[idx]` materialization plus elementwise multiply and reduction
- `dense_gemm`: `H @ W.T` followed by gather

### Experiment A: Correctness

Question:
- Does the extension match PyTorch reference outputs and gradients closely enough for training use?

Measurements:
- Forward max/mean absolute error versus naive subset reference
- `grad_H` max/mean absolute error versus autograd reference
- `grad_W` max/mean absolute error versus autograd reference

Configs:
- `quick_small`
- `gpt2_like_smallk`
- `gpt2_like_midk`
- `gpt2_like_largek`
- Run in `fp16`
- Run in `bf16` if the GPU supports it

Paper output:
- One small table with max and mean absolute errors for forward, `grad_H`, and `grad_W`

### Experiment B: Forward and End-to-End Runtime

Question:
- When is the fused kernel faster or slower than baseline PyTorch alternatives?

Measurements:
- Forward-only latency
- Forward+backward latency
- Backward-only latency for the fused kernel
- Estimated TFLOP/s using the script’s simple operation model

Configs:
- Sweep `k` at fixed `N`, `d`, `V`
- Sweep `d` at fixed `N`, `k`, `V`
- Sweep `N` at fixed `d`, `k`, `V`

Use the `paper` suite first, then extend if needed.

Paper output:
- Table: latency and TFLOP/s by method and configuration
- Plot: speedup of `fused` over `naive_subset`
- Plot: speedup of `fused` over `dense_gemm` when dense fits in memory

### Experiment C: Memory

Question:
- How much memory does the fused kernel save relative to baselines?

Measurements:
- Peak allocated CUDA memory
- Peak reserved CUDA memory
- Theoretical size of the avoided `[N, k, d]` tensor
- Whether dense full logits OOM or are skipped by feasibility estimate

Configs:
- Same as Experiment B

Paper output:
- Table with peak memory for forward and forward+backward
- Plot: peak allocated memory versus `k`
- Plot: peak allocated memory versus `N`

### Experiment D: Atomic-Collision Sensitivity

Question:
- How much does repeated indexing hurt the `grad_W` kernel?

Measurements:
- Forward-only latency under each collision regime
- Backward-only latency under each collision regime
- Forward+backward latency for the fused kernel only
- Collision regimes induced by drawing indices from progressively smaller vocabularies

Collision regimes in the script:
- `collision_vocab = V`
- `collision_vocab = 4096`
- `collision_vocab = 512`
- `collision_vocab = 64`

Paper output:
- Plot: fused backward latency versus collision regime
- Plot: fused forward latency versus collision regime
- Text: whether the trend is driven primarily by atomic contention or by improved weight-row locality

This experiment is important because the current backward implementation uses atomics directly over `grad_W`, so the contention story should be established empirically rather than asserted.
The first-pass run already suggests that shrinking the sampled vocabulary can make the end-to-end kernel faster, which means the benchmark must separate forward locality effects from backward atomic effects before the paper draws conclusions.

## 4. Suggested command lines

Build the extension in the environment that already has PyTorch and CUDA:

```bash
/home/pettyjohnjn/.conda/envs/tuned_lens_env/bin/python3.9 setup.py build_ext --inplace
```

Quick smoke test:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/pettyjohnjn/.conda/envs/tuned_lens_env/bin/python3.9 benchmark_paper.py \
  --suite quick \
  --dtype fp16 \
  --warmup 10 \
  --iters 25
```

Main paper run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/pettyjohnjn/.conda/envs/tuned_lens_env/bin/python3.9 benchmark_paper.py \
  --suite paper \
  --dtype fp16 \
  --warmup 25 \
  --iters 100 \
  --output-json results_fp16.json \
  --output-csv results_fp16.csv
```

BF16 run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/pettyjohnjn/.conda/envs/tuned_lens_env/bin/python3.9 benchmark_paper.py \
  --suite paper \
  --dtype bf16 \
  --warmup 25 \
  --iters 100 \
  --output-json results_bf16.json \
  --output-csv results_bf16.csv
```

Collision study:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/pettyjohnjn/.conda/envs/tuned_lens_env/bin/python3.9 benchmark_paper.py \
  --suite collisions \
  --dtype fp16 \
  --seeds 0,1,2 \
  --warmup 25 \
  --iters 100 \
  --output-json results_collisions.json \
  --output-csv results_collisions.csv \
  --output-summary-json results_collisions_summary.json \
  --output-summary-csv results_collisions_summary.csv
```

Extended scaling sweep:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/pettyjohnjn/.conda/envs/tuned_lens_env/bin/python3.9 benchmark_paper.py \
  --suite extended \
  --dtype fp16 \
  --seeds 0,1,2 \
  --warmup 25 \
  --iters 100 \
  --output-json results_extended.json \
  --output-csv results_extended.csv \
  --output-summary-json results_extended_summary.json \
  --output-summary-csv results_extended_summary.csv
```

## 5. Tables and figures to include in the paper

Recommended replacement for the current benchmark table:

### Table 1: Correctness
- Config
- Dtype
- Forward max abs err
- `grad_H` max abs err
- `grad_W` max abs err

### Table 2: Runtime and Memory
- Config
- Method
- Forward ms
- Forward+backward ms
- Peak allocated MB
- Peak reserved MB
- TFLOP/s estimate
- Status

### Figure 1: Speedup versus subset size
- X-axis: `k`
- Y-axis: speedup over naive subset

### Figure 2: Peak memory versus subset size
- X-axis: `k`
- Y-axis: peak allocated MB
- Curves: fused, naive subset, dense GEMM

### Figure 3: Backward sensitivity to repeated indices
- X-axis: collision vocabulary size
- Y-axis: fused forward+backward ms

## 6. Interpreting likely outcomes honestly

Given the current implementation, expect:
- Very strong memory wins over naive subset materialization
- Dense full-logits baseline to become infeasible quickly as `N` and `V` grow
- Fused forward runtime to be competitive in memory-bound regimes, but not necessarily faster than highly optimized GEMM when dense fits comfortably
- Backward performance to degrade as collisions increase because of direct atomic accumulation into `grad_W`

That means the strongest supported paper claim is:
- the kernel is a correctness-first, memory-efficient implementation that avoids subset-materialization blowups

The weakest claim, unless new results prove otherwise, is:
- that this specific kernel is broadly faster than dense GEMM on modern GPUs

## 7. Next implementation steps if you want the appendix to stay as written

To support the appendix without softening the claims, the repo would need:
- A real LoRA path in C++/CUDA and corresponding tests
- A benchmarked Triton fallback
- More optimized forward and backward kernels
- Profiling evidence for memory coalescing, occupancy, and register behavior
- A dedicated script that emits exactly the final paper tables
