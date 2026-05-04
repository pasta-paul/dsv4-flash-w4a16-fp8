# DeepSeek-V4-Flash → W4A16-FP8 (compressed-tensors, vLLM-deployable)

End-to-end integration work to produce a **W4A16 + FP8_BLOCK quantization of
DeepSeek-V4-Flash that actually serves in vLLM**, on Hopper-class GPUs (H200
SM 9.0). Built by stacking three in-flight upstream draft PRs and patching
the gaps between them.

> **Status (2026-05-04):** Phase 3b complete. Calibrated on 768 samples,
> serves cleanly at TP=2 on 8× H200, full harness suite passes:
> chat-smoke quick 4/4, quality 4/4, coding 2/2, **toolcall15 26/30 (87%)** —
> beating the FP4/FP8 native baseline by **+3 points** on toolcall15.
> Live model: **https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8** (public, Apache-2.0).

## Why this exists

DeepSeek-V4-Flash dropped on April 24, 2026 (284B total / 13B active MoE,
hybrid CSA + HCA attention with mHC hyperconnections, hash-routed experts).
As of May 2, 2026:

- The model code is in **no released `transformers`** — only an open PR (#45643).
- vLLM has **no merged compressed-tensors path** for V4 — only an open Draft PR (#41276).
- LLM Compressor has **no merged V4 quantization path** — only an open PR (#2647).
- The published reference quant ([RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8))
  uses NVFP4 expert weights, which require SM 10.0+ tcgen05 instructions —
  unavailable on Hopper SM 9.0 and Spark SM 12.1.
- Intel published [`Intel/DeepSeek-V4-Flash-W4A16-AutoRound`](https://huggingface.co/Intel/DeepSeek-V4-Flash-W4A16-AutoRound)
  (5 days old, 23k+ downloads), but their model card explicitly states *"vLLM
  and SGLang is not supported currently."*

This project produces a W4A16 GPTQ V4-Flash that **does serve in vLLM at TP=2
on Hopper today**, with attention quantized to FP8_BLOCK (mirroring RedHat's
recipe topology, swapping NVFP4 → W4A16 for SM 9.x / 12.x compatibility).

## Validation (Phase 3b on 8× H200, TP=2)

| Tag | Native FP4/FP8 baseline | **W4A16-FP8** |
|---|---|---|
| `chat-smoke quick` | 4/4 | **4/4** |
| `chat-smoke quality` | 4/4 | **4/4** |
| `chat-smoke coding` (8192 max_tokens, t=1.0) | 2/2 | **2/2** |
| `toolcall15` (15 cases × 2 pts) | 23/30 (77%) | **26/30 (87%)** |
| `toolcall15` PASS / PARTIAL / FAIL | 11 / 1 / 4 | **13 / 0 / 2** |

Apples-to-apples on [`jasl/vllm-ds4-sm120-harness`](https://github.com/jasl/vllm-ds4-sm120-harness) HEAD `85aca32`.
TC-11 (Simple Math) was PARTIAL on baseline AND the 16-sample dryrun; **PASS in Phase 3b** — likely from the larger calibration tightening math-reasoning weight quantization.
Remaining 2 toolcall15 fails (TC-06 Multi-Value Extraction, TC-08 Conditional Branching) **also fail on the native FP4/FP8 baseline** — these are V4-Flash model-architecture limits, not quantization defects.

## What's in this repo

| Path | What |
|---|---|
| `REPORT.md` | Full phase-by-phase mission log: setup → native baseline → dequant → calibration → vLLM serve attempts → harness results → decisions and pivots. |
| `model-card-draft.md` | Pre-publish draft of the HF model card. The published version lives at https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8. |
| `findings/upstream-issue-marlin-tp-sharding.md` | Root-cause for a Marlin MoE kernel TP scale-sharding bug discovered during integration. Empirical TP=1/2/8 table, suggested fix location. **Filed upstream as [vllm-project/vllm#41511](https://github.com/vllm-project/vllm/issues/41511) on 2026-05-02.** Blocks all compressed-tensors W4A16 MoE deployments under TP > 2 on every GPU architecture. |
| `findings/kylesayrs-pr-41276-integration.md` | Detailed integration notes for the [neuralmagic/vllm](https://github.com/neuralmagic/vllm) `kylesayrs/deepseek-ct` branch (PR #41276) — 5 documented upstream gaps with our patches. |
| `findings/phase3b-recovery.md` | The Phase 3b OOM + NCCL-timeout journey: what failed, what worked, and the exact env+recipe combination required to GPTQ-calibrate V4-Flash at scale on 8× H200 without crashes. |
| `patches/` | Static patches against upstream (see [`patches/VERSIONS.md`](patches/VERSIONS.md)). Includes calibration patches (`helpers.py.diff`, `modeling_deepseek_v4.py.diff`) and the `packed_modules_mapping.diff` for vLLM serving. |
| `scripts/` | Working scripts in two categories: **calibration** (run on AWS H200 box) and **serve** (run on inference target — H200 here, applicable to DGX Spark with kernel updates). |

## Build for AWS calibration

Calibration runs against the BF16-dequantized base model on 8× H200 with `kylesayrs/transformers-v5` llm-compressor branch.

```bash
# In a clean venv (do NOT share the vLLM serve venv — pip cascades break vLLM's torch+cu13 pin)
pip install git+https://github.com/huggingface/transformers.git@add-deepseek-v4
pip install git+https://github.com/vllm-project/llm-compressor.git@kylesayrs/transformers-v5
pip install --pre 'compressed-tensors>=0.15.1a2'

# Apply calibration-time patches:
patch -p1 -d "$(python -c 'import llmcompressor; print(llmcompressor.__path__[0])')" < patches/helpers.py.diff
patch -p1 -d "$(python -c 'import transformers; print(transformers.__path__[0])')" < patches/modeling_deepseek_v4.py.diff

# /dev/shm must be ≥ 1.8 TiB for 8-rank torchrun on a 543 GB BF16 model
sudo mount -o remount,size=1800G /dev/shm

# Run calibration (recipe: FP8_BLOCK attn + W4A16 GPTQ routed experts)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_CUDA_ARCH_LIST=9.0a
torchrun --nproc-per-node 8 scripts/quantize_v4_w4a16.py \
    --samples 768 --batch-size 4 \
    --input  /path/to/DeepSeek-V4-Flash-bf16 \
    --output /path/to/DeepSeek-V4-Flash-W4A16-FP8
```

See `findings/phase3b-recovery.md` for **why** each env var is required (every one of them is the fix to a specific failure mode hit during this work).

## Build for vLLM serving

Serving requires:
- [`jasl/vllm@ds4-sm120`](https://github.com/jasl/vllm/tree/ds4-sm120) (PR #40991) base
- Cherry-pick the `kylesayrs/deepseek-ct` branch from [`neuralmagic/vllm`](https://github.com/neuralmagic/vllm/tree/kylesayrs/deepseek-ct) — vLLM PR #41276, branch `kylesayrs/deepseek-ct` in the `neuralmagic/vllm` fork (commit `f910a73a`). **Note**: this is a branch in the `neuralmagic` organization's vLLM fork; not in `kylesayrs`'s personal fork.
- Apply `patches/packed_modules_mapping.diff` to add the `packed_modules_mapping` class attribute that PR #41276 references but does not define.

```bash
git clone https://github.com/jasl/vllm.git -b ds4-sm120 vllm
cd vllm
git remote add neuralmagic https://github.com/neuralmagic/vllm.git
git fetch neuralmagic kylesayrs/deepseek-ct
git cherry-pick f910a73a   # the "support ct quantization" commit
patch -p1 < ../patches/packed_modules_mapping.diff
pip install -e . --no-build-isolation
```

Then:
```bash
vllm serve pastapaul/DeepSeek-V4-Flash-W4A16-FP8 \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 --block-size 256 --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --trust-remote-code
```

**Important — TP limit:** TP=1 OOMs on a single 141 GB H200. **TP=2 works.** TP ≥ 4 hits the upstream Marlin MoE TP scale-sharding bug ([vllm-project/vllm#41511](https://github.com/vllm-project/vllm/issues/41511)) — until that's fixed, this model is TP=2-only.

## Roadmap status

- [x] **Phase 0** — Setup & verification on AWS p5en.48xlarge
- [x] **Phase 1** — Native FP4/FP8 V4-Flash baseline (jasl harness reference scores)
- [x] **Phase 2** — Dequantize FP4/FP8 → BF16 (flagos)
- [x] **Phase 3a** — Dry-run W4A16-FP8 calibration (16 samples) — toolcall15 25/30
- [x] **Phase 3b** — Full W4A16-FP8 calibration (768 samples) — toolcall15 26/30
- [x] **Phase 4** — Harness verify on H200 (TP=2): chat-smoke 10/10, toolcall15 26/30
- [x] **Phase 5** — Public HF release at [`pastapaul/DeepSeek-V4-Flash-W4A16-FP8`](https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8)
- [x] **Upstream contribution** — vllm-project/vllm#41511 filed; cross-link comment on PR #40991

### Standard benchmarks (live)

| Benchmark | Setting | Score |
|---|---|---|
| GSM8K | 5-shot, chat-template, flexible-extract | **92.87% ±0.71%** |
| MMLU | 5-shot | **87.27% ±0.27%** |
| HumanEval | 0-shot (instruct), pass@1 | **54.27% ±3.9%** |

Results updated to the [HF model card](https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8) as each lands.

## Credits

- [@jasl](https://github.com/jasl) — DeepSeek-V4 vLLM SM12x base support (PR #40991)
- [@kylesayrs](https://github.com/kylesayrs) — compressed-tensors V4 attention path (PR #41276)
- [@aabbccddwasd](https://github.com/aabbccddwasd) — indexer KV cache layout fix
- [@bbbearxyz](https://github.com/bbbearxyz) — SM12x Triton fallback kernels
- [@wuwenthink](https://github.com/wuwenthink) — SM12x harness validation
- [`RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8`](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8) — published reference for V4 mixed-precision attention topology

Apache-2.0, inherited from the base model.
