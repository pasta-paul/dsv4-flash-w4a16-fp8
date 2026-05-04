# DGX Spark TP=2 deployment validation

**Date**: 2026-05-04 · **Hardware**: 2× DGX Spark GB10 (SM 12.1a, 121 GB UMA each) · **Topology**: TP=2 over QSFP RDMA · **Quant**: this model (`pastapaul/DeepSeek-V4-Flash-W4A16-FP8`)

End-to-end validation of this W4A16-FP8 quant on dual DGX Spark Grace Blackwell hardware. Same harness as the H200 reference run, different topology, one operational constraint surfaced (CUDA-graph workspace lock).

## TL;DR

Runs **stably** with two configuration constraints on Spark UMA:

1. **`--enforce-eager` is required**. Without it, vLLM's per-rank attention workspace gets locked at the post-profile size and crashes on prompts >~1K tokens with `Workspace is locked but allocation requires X MB, current size is Y MB` originating in `deepseek_v4_attention.py:1454:_forward_prefill`. Eager mode costs ~4× decode throughput (~3–4 tok/s vs ~14–15 tok/s) but lets every harness prompt size complete.
2. **`--gpu-memory-utilization 0.92`** — running at 0.85 left no headroom for the attention workspace; 0.92 is the sweet spot on Spark UMA without OOM-ing the OS.

With those, **103 / 108 evaluated cases pass** across the public jasl `vllm-ds4-sm120-harness` plus the original Spark validation harness and 5 B200 oracle-alignment cases.

## Build provenance

Built via the [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) toolchain, targeting SM 12.1a:

| Component | Pin |
|---|---|
| vLLM | `jasl/vllm@428e08e` + cherry-pick `f910a73a93` + packed_modules patch |
| Resulting commit | `2467baff` (in image) |
| transformers | `5.8.0.dev0` (HF main; PR #45643 `add-deepseek-v4` was merged 2026-05-02 and the branch deleted) |
| compressed-tensors | `0.15.1.a20260428` (pre-release) |
| PyTorch | `2.11.0+cu130` (aarch64) |
| FlashInfer | `0.6.9` (commit `68d2b66a`) |
| Triton | `3.6.0` |
| Base image | `nvidia/cuda:13.2.0-devel-ubuntu24.04` |
| Image size | 20.35 GB |

`transformers 5.8.0.dev0` was layered on top of the base build because the original `add-deepseek-v4` branch was deleted post-merge — installing from the branch fails. Installing from `main` after 2026-05-02 picks up the merged DSV4 layer-types, which is what's required.

## Serve invocation

```bash
vllm serve pastapaul/DeepSeek-V4-Flash-W4A16-FP8 \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --enforce-eager \
  --max-model-len 16384 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.92 \
  --host 0.0.0.0 --port 8888 \
  -tp 2 --nnodes 2 \
  --master-addr <HEAD_IP> --master-port 29501 \
  --node-rank 0   # rank 1 on the worker node also gets --headless
```

Worker (rank 1) launches with the same image and additional `--headless`. Without `--headless`, the worker tries to initialize its own engine and hits `AssertionError: collective_rpc should not be called on follower node` in `multiproc_executor.py:351`.

Cold start (eager): weight load 2:18, no torch.compile, KV profiling ~30s, server up at ~3 min total. Loading 73 GiB resident per rank.

## Validation results

### jasl `run_acceptance.sh` — full gate run

| Gate | Result |
|---|---|
| `compileall` | ✅ |
| `health` | ✅ |
| `pytest` (harness self-tests) | ⚠️ 3 env-specific failures unrelated to model |
| `ruff` | ✅ |
| `smoke_quick` | ✅ **4 / 4** |
| `generation` | ✅ **54 / 54** (18 prompts × 3 thinking modes: non-thinking / think-high / think-max) |
| `toolcall15` | ✅ **39 / 45**, 80 / 90 points (89%) |
| `oracle_compare` (vs B200 TP=2 baseline) | ✅ 5 / 5 ran, alignment numbers below |

#### toolcall15 partial-failures (6 / 45)

| Scenario | Mode(s) | Reason |
|---|---|---|
| TC-06 Multi-Value Extraction | all 3 thinking modes | Doesn't split a multi-language translation request into two separate tool calls |
| TC-15 Conflicting Information | think-high | Partial credit — got right answer but lost points on intermediate trace |
| TC-05 Date/Time Parsing | think-max | Relative-date parsing wrong only when thinking-max engaged |
| TC-11 Simple Math | think-max | Used calculator unnecessarily for trivial arithmetic |

TC-06 failure pattern matches the H200 baseline (also fails there).

### B200 token-level alignment

This-model TP=2 (W4A16 GPTQ + FP8_BLOCK + BF16 shared) vs B200 TP=2 native FP4/FP8 reference. Both share the kylesayrs PR #41276 cherry-pick + packed_modules_mapping patch but use different underlying expert quants. Tested via `oracle-compare` against `baselines/20260502_b200_tp2_main_5737770c6/oracle/nomtp/`.

| Case | Top-1 match | Top-K overlap | Matching prefix | Mean chosen-token logprob err |
|---|---|---|---|---|
| `completion_short_math` | **87.5%** | 69.7% | 10 / 16 | 0.094 |
| `completion_translation` | 22.7% | 27.5% | 4 / 22 | 0.041 |
| `completion_long_prefill_2048` | 20.0% | 18.5% | 8 / 50 | 0.139 |
| `completion_raw_intro` | 7.3% | 12.7% | 7 / 96 | 0.256 |
| `completion_code_probe` | 1.9% | 3.7% | 0 / 160 | 0.238 |

**Reading the table**:
- Math (87.5%) shows the two quants converge cleanly on deterministic-computation tokens.
- Translation, prose, and code show greedy-path divergence after the first few tokens — both quants pick valid-but-different next tokens (`提升` vs `改善`, `refers` vs `means`, etc).
- Mean chosen-token logprob error 0.04–0.26 across cases — the *probabilities* the model assigns to its chosen tokens are close to B200 even when the picked token differs. Expected fingerprint of two different quantizations of the same base model.

#### Note on the oracle replay

The B200 oracle JSONs have `"model": "deepseek-ai/DeepSeek-V4-Flash"` baked into each request body. vLLM `--served-model-name` does not accumulate cleanly across repeated flags or multi-positional values (only the first or last wins depending on form). To make the oracle replay work, the JSON files were patched locally to substitute the served name. No data semantics change.

### Original Spark validation harness (13 prompts) — 13 / 13 PASS

This earlier run was on `--gpu-memory-utilization 0.92` *without* `--enforce-eager` (CUDA graphs on, decode peaked at ~15 tok/s). All prompts in this set stayed under the workspace-lock threshold.

| Tier | Tests | Result |
|---|---|---|
| P1 smoke (3) | math / geo / colors | 3 / 3 stop |
| P2 tool calls (3) | weather / read_file / multi | 3 / 3 tool_calls |
| P3 long-form code (3) | clock_html (2,845 tok) / aquarium_html (3,578 tok) / python_fib (1,966 tok) | 3 / 3 stop @ 14.4–15.0 tok/s |
| P4 creative (3) | LOTR / Ming / haiku-format | 3 / 3 stop |
| P5 agentic codemod (1) | tools-mediated file modify | 1 / 1 tool_calls |

## Operational constraints

1. **TP=2 only.** TP=1 OOMs even on 141 GB H200 (per model card); TP≥4 hits upstream `compressed-tensors W4A16 MoE scale-sharding` bug ([vllm-project/vllm#41511](https://github.com/vllm-project/vllm/issues/41511)).
2. **`--enforce-eager` required** until vLLM's `vllm/v1/worker/workspace.py:_ensure_workspace_size` allows growth after lock, OR `jasl/vllm@843fe9e` (53 commits ahead of the build base) includes a relevant fix. The newer commits include `[DSV4] Add knob to enable pre-attn gemm` (#41443), `[Perf] Integrate Tile Kernels head_compute_mix_kernel for DSV4` (#41255), and `[Attention] Abstract MLA prefill backends` (#32623) — at least one may move workspace sizing.
3. **Memory tight at 0.92 gpu-mem-util.** Spark UMA shows ~118 / 121 GiB used while serving; no headroom for co-tenants.
4. **Decode ~3–4 tok/s eager** vs ~14–15 tok/s with CUDA graphs. ~4× penalty.
5. **Worker (rank 1) needs `--headless`** so it waits for head-side RPC broadcasts instead of trying to init its own engine.

## Recommended next iterations

1. Rebuild against `jasl/vllm@843fe9e` (current branch tip) to test if the 53-commit gap closes the workspace-lock issue.
2. If not, patch `workspace.py:_ensure_workspace_size` to allow growth after lock (single-line fix candidate).
3. Run `scripts/run_long_context_probe.sh` after raising `--max-model-len` (currently capped at 16384) once eager is stable, to validate long-context behavior on Spark.
4. Run `scripts/run_bench_matrix.sh` for throughput-vs-concurrency curves.
5. Run `scripts/run_lm_eval.sh` (gsm8k/mrcr/etc.) to add Spark-side standardized scores alongside the existing H200 numbers.
