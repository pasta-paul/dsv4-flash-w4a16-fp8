# DeepSeek-V4-Flash → W4A16 (compressed-tensors, vLLM-deployable)

End-to-end integration work to produce a **W4A16 + FP8_BLOCK quantization of
DeepSeek-V4-Flash that actually serves in vLLM**, on Hopper-class GPUs (H200
SM 9.0). Built by stacking three in-flight upstream draft PRs and patching
the gaps between them.

> **Status (May 2, 2026):** Phase 3a (16-sample dryrun) calibration complete,
> serves cleanly at TP=2, full harness suite passes (chat-smoke quick 4/4,
> quality 4/4, coding 2/2, toolcall15 25/30 = 83%) — beating the FP4/FP8
> native baseline by +2 points on toolcall15. Phase 3b (1024-sample full
> calibration) launching for the production artifact. HF upload pending
> Phase 4 verify.

## Why this exists

DeepSeek-V4-Flash dropped on April 24, 2026 (284B total / 13B active MoE,
hybrid CSA + HCA attention with mHC hyperconnections, hash-routed experts).
As of May 2, 2026:

- The model code is in **no released `transformers`** — only an open PR.
- vLLM has **no merged compressed-tensors path** for V4 — only an open Draft PR.
- LLM Compressor has **no merged V4 quantization path** — only an open PR.
- The published reference quant ([RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8))
  uses NVFP4 expert weights, which require SM 10.0+ tcgen05 instructions —
  unavailable on Hopper SM 9.0 and Spark SM 12.1.
- Intel published [`Intel/DeepSeek-V4-Flash-W4A16-AutoRound`](https://huggingface.co/Intel/DeepSeek-V4-Flash-W4A16-AutoRound)
  (5 days old, 23k+ downloads), but their model card explicitly states *"vLLM
  and SGLang is not supported currently."*

This project produces a W4A16 GPTQ V4-Flash that **does serve in vLLM at TP=2
on Hopper** today, with attention quantized to FP8_BLOCK (mirroring RedHat's
recipe topology, swapping NVFP4 → W4A16 for SM 9.x / 12.x compatibility).

## What's in this repo

| Path | What |
|---|---|
| `REPORT.md` | Full phase-by-phase mission log: setup, native baseline, dequant, calibration, vLLM serve attempts, harness results, decisions and pivots. The narrative of how the integration came together. |
| `findings/upstream-issue-marlin-tp-sharding.md` | Root-cause analysis of a Marlin MoE kernel TP scale-sharding bug discovered during integration. Empirical TP=1/2/8 table, suggested fix location, paste-ready issue body for upstream filing. **Affects all compressed-tensors W4A16 MoE deployments under TP > 2 on every GPU architecture.** |
| `findings/kylesayrs-pr-41276-integration.md` | Five distinct integration gaps in vLLM PR #41276 and LLM Compressor PR #2647 documented end-to-end: weight-name remapping, `shared_experts.down_proj` save decomposition, `quantization_config.ignore` rewrite, BF16 attention NotImplementedError, and the `packed_modules_mapping` class attribute fix. |
| `model-card-draft.md` | Working draft of the HF model card. Will become the README on `pastapaul/DeepSeek-V4-Flash-W4A16-FP8` once Phase 3b lands. |
| `patches/` | Source patches with provenance: `helpers.py.diff` (llm-compressor Cache-class tracing fix), `modeling_deepseek_v4.py.diff` (transformers DynamicCache skip), `quantize_v4_w4a16.py.snapshot` (the recipe), `VERSIONS.md` (full reproduction checklist with package versions + repo HEADs). |
| `scripts/` | All the integration glue: build script, calibration recipe, weight-name rewriter (`rewrite_for_vllm.py`), config patchers, diagnostics, harness runner. |

## Upstream branches built on

| Project | Branch / PR | Commit | Purpose |
|---|---|---|---|
| vLLM | [jasl/vllm@ds4-sm120](https://github.com/jasl/vllm/tree/ds4-sm120) (PR [#40991](https://github.com/vllm-project/vllm/pull/40991)) | `428e08e` (build base) → `68901da` (current HEAD) | Hopper + SM12x V4 inference support |
| vLLM | [neuralmagic:kylesayrs/deepseek-ct](https://github.com/neuralmagic/vllm/tree/kylesayrs/deepseek-ct) (PR [#41276](https://github.com/vllm-project/vllm/pull/41276)) | `f910a73a` | compressed-tensors V4 attention path |
| LLM Compressor | [kylesayrs/transformers-v5](https://github.com/vllm-project/llm-compressor/tree/kylesayrs/transformers-v5) (PR [#2647](https://github.com/vllm-project/llm-compressor/pull/2647)) | `a308bc0e` | V4 calibration + `linearize_moe_model()` |
| transformers | [huggingface/transformers@add-deepseek-v4](https://github.com/huggingface/transformers/tree/add-deepseek-v4) (PR [#45643](https://github.com/huggingface/transformers/pull/45643)) | `5.8.0.dev0` | V4 model architecture in transformers |
| FlagOS | [flagos-ai/DeepSeek-V4-FlagOS](https://github.com/flagos-ai/DeepSeek-V4-FlagOS) | `f9846dc` | FP4/FP8 → BF16 dequantization tool |

## Quick reference: harness results at TP=2

H200 SM 9.0, full V4 harness from [jasl/vllm-ds4-sm120-harness](https://github.com/jasl/vllm-ds4-sm120-harness) (HEAD `85aca32`).

| Suite | TP=2 W4A16 dryrun (this work) | Native FP4/FP8 baseline | wuwenthink RTX PRO 6000 SM120 TP=2 |
|---|---|---|---|
| chat-smoke quick | **4/4 PASS** | 4/4 PASS | 2/4 |
| chat-smoke quality | **4/4 PASS** | 4/4 PASS | 3/4 |
| chat-smoke coding | **2/2 PASS** ✅ | 2/2 PASS | 0/2 |
| toolcall15 | **25/30 (83%)** | 23/30 (77%) | 26/30 |

Notable: a 16-sample-calibration W4A16 dryrun outperforms native FP4/FP8 on
toolcall15 (failure-set is a strict subset; TC-14 *Malformed Response* passes
under W4A16 but fails under native).

The SM120 coding 0/2 result was originally hypothesized to be reasoning-token
exhaustion. The H200 result with the same harness defaults disproves that:
**SM12x coding 0/2 is reproducibly an SM12x kernel correctness issue, not a
reasoning-loop**. See `findings/upstream-issue-marlin-tp-sharding.md` and the
PR comment template at the bottom of that doc.

## Reproduction

If you have a Hopper-class box (H100/H200 ideally) with sufficient VRAM
(8× 80GB+ minimum for calibration; 2× 80GB for inference at TP=2):

1. Read `REPORT.md` for the full narrative.
2. Read `findings/kylesayrs-pr-41276-integration.md` for the integration gaps.
3. Follow `patches/VERSIONS.md` "How to reproduce on a fresh box" — pinned
   commits, exact package versions, build steps.
4. Apply the patches in `patches/`.
5. Run the calibration via `scripts/quantize_v4_w4a16.py` + the recipe in
   `patches/quantize_v4_w4a16.py.snapshot`.
6. Post-process with `scripts/rewrite_for_vllm.py` + the config patchers.
7. Serve at TP=2 using `scripts/serve_quant.sh`.
8. Validate with `scripts/run_harness.sh`.

The published model will live at
[`pastapaul/DeepSeek-V4-Flash-W4A16-FP8`](https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8)
once Phase 3b verification completes.

## Status / Roadmap

- [x] Phase 0 — vLLM build, environment setup
- [x] Phase 1 — Native FP4/FP8 baseline + harness
- [x] Phase 2 — FP4/FP8 → BF16 dequantization
- [x] Phase 3a — 16-sample dryrun calibration + vLLM serve at TP=2 + full harness
- [ ] Phase 3b — 1024-sample full calibration *(in progress)*
- [ ] Phase 4 — Verify Phase 3b artifact at TP=2 + delta vs dryrun
- [ ] Phase 5 — HF upload + model card publication
- [ ] Spark deployment — TP=2 serve on DGX Spark (SM 12.1 / GB10)

## Upstream contributions

- **vLLM Marlin MoE TP scale-sharding bug** — *filed* (issue link will appear
  here once posted). Affects all compressed-tensors W4A16 MoE under TP > 2.
- **PR #40991 reasoning-loop hypothesis disambiguation** — empirical
  comparison data (TP=2 same harness defaults: H200 W4A16 coding 2/2 PASS
  vs SM120 coding 0/2) posted as a comment on PR #40991.
- **kylesayrs PR #41276 integration findings** — five distinct gaps
  documented in this repo for the maintainers' reference.

## Credits

Built on work by [@jasl](https://github.com/jasl) (PR #40991 V4 + SM12x
support), [@kylesayrs](https://github.com/kylesayrs) and the Red Hat / Neural
Magic team (PR #41276 compressed-tensors, PR #2647 LLM Compressor V4),
[@bbbearxyz](https://github.com/bbbearxyz) (SM12x Triton fallbacks),
[@aabbccddwasd](https://github.com/aabbccddwasd) (indexer KV cache layout
fix), [@wuwenthink](https://github.com/wuwenthink) (SM12x harness
validation), [@huggingface/transformers](https://github.com/huggingface/transformers)
contributors on PR #45643, and [@flagos-ai](https://github.com/flagos-ai) for
the BF16 dequant tool. This project is integration glue on top of all of that
work.

## License

Apache 2.0 (see `LICENSE`). The base model `deepseek-ai/DeepSeek-V4-Flash` is
under its own license — see the model's HF page.
