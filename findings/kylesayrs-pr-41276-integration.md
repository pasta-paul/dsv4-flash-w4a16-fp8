# Integrating kylesayrs PR #41276 with W4A16 quantization on jasl/ds4-sm120

Notes on bringing up DeepSeek-V4-Flash AWQ-W4A16 end-to-end on 8× H200 by stacking
the upstream draft PRs that exist for V4 today, plus the gaps those PRs leave.

This is from a single quant + serve session 2026-05-01/02, integrating:

- vLLM PR [#41276](https://github.com/vllm-project/vllm/pull/41276)
  (`neuralmagic/vllm@kylesayrs/deepseek-ct`, originally commit `f910a73a93`;
   rebased successor `d09eeb498` is vendored locally as
   `scripts/kylesayrs-deepseek-ct.patch` — see "SHA rebase recovery" below)
- llm-compressor PR [#2647](https://github.com/vllm-project/llm-compressor/pull/2647)
  (branch `kylesayrs/transformers-v5`, commit `a308bc0e`)
- transformers PR [#45643](https://github.com/huggingface/transformers/pull/45643)
  (branch `add-deepseek-v4`)
- vLLM jasl/ds4-sm120 (commit `428e08ec`) for Hopper SM12x adaptations

Hardware: AWS p5en.48xlarge, 8× H200, DLAMI Ubuntu 24.04 + PyTorch 2.10. We bumped
torch to 2.11.0+cu130 because vLLM jasl/ds4-sm120 hard-pins it.

## TL;DR

vLLM PR #41276 fixes attention-side compressed-tensors integration. It does NOT
cover four other gaps you will hit if you actually try to quantize V4-Flash to
W4A16 with the kylesayrs llm-compressor branch and serve in vLLM. Those gaps:

1. Weight name remapping (transformers-v5 nested → vLLM expected flat).
2. `shared_experts.down_proj` save-side row decomposition (refusion required).
3. `quantization_config.ignore` post-rename path rewrite.
4. PR #41276 explicitly raises `NotImplementedError("DeepSeekV4 requires FP8
   attention quantization")` if attn weights are unquantized — the recipe must
   include attention.

Each is well-scoped and tractable. The end-to-end path takes one calibration
run plus a header-rewrite pass plus a config patch, and the model loads.

## What PR #41276 does and does not cover

It covers:

- `scale_fmt` defensive fallback in `DeepseekV4Attention.__init__` so
  compressed-tensors quant configs (which don't carry `scale_fmt`) don't
  KeyError the model init.
- `wo_scale_name` selector that picks `weight_scale_inv` (FP8 native) vs
  `weight_scale` (compressed-tensors) at runtime.
- Wrapping two raw `torch.mm` calls inside the compressor through the actual
  `Linear()` so they go through the quant kernels properly.
- Two `quant_config=None` → `quant_config=quant_config` swaps so previously-
  unquantizable layers can now be quantized.

It does NOT cover:

- BF16 attention. The same `wo_scale_name` selector raises
  `NotImplementedError("DeepSeekV4 requires FP8 attention quantization")` if the
  attn weight has neither `weight_scale_inv` nor `weight_scale`. Quantize attn or
  patch the model code; there is no third option without forking the PR.
- Weight name remapping between transformers-v5 saved names and vLLM's expected
  native names.
- The `shared_experts.down_proj` save decomposition (see below).

## Gap 1: name remapping

llm-compressor's `kylesayrs/transformers-v5` branch loads the BF16 model via
transformers v5 (PR #45643), which renames internal modules to the v5 naming
scheme. `model.save_pretrained()` then writes those names. vLLM's V4 model code
in jasl/ds4-sm120 expects the older flat native names.

| Source (kylesayrs save)                        | Target (vLLM expected)                       |
|-----                                           |-----                                         |
| `model.embed_tokens.weight`                    | `embed.weight`                               |
| `lm_head.weight`                               | `head.weight`                                |
| `model.norm.weight`                            | `norm.weight`                                |
| `model.hc_head.hc_base/hc_fn/hc_scale`         | `hc_head_base/_fn/_scale`                    |
| `model.layers.X.attn_hc.base/.fn/.scale`       | `layers.X.hc_attn_base/_fn/_scale`           |
| `model.layers.X.ffn_hc.base/.fn/.scale`        | `layers.X.hc_ffn_base/_fn/_scale`            |
| `model.layers.X.input_layernorm`               | `layers.X.attn_norm`                         |
| `model.layers.X.post_attention_layernorm`      | `layers.X.ffn_norm`                          |
| `model.layers.X.self_attn.*`                   | `layers.X.attn.*`                            |
| `model.layers.X.mlp.*`                         | `layers.X.ffn.*`                             |
| `model.layers.X.self_attn.kv_proj`             | `layers.X.attn.wkv`                          |
| `model.layers.X.self_attn.q_a_proj/q_b_proj`   | `layers.X.attn.wq_a/wq_b`                    |
| `model.layers.X.self_attn.o_a_proj/o_b_proj`   | `layers.X.attn.wo_a/wo_b`                    |
| `model.layers.X.self_attn.q_a_norm`            | `layers.X.attn.q_norm`                       |
| `model.layers.X.self_attn.sinks`               | `layers.X.attn.attn_sink`                    |
| `model.layers.X.mlp.shared_experts.gate_proj`  | `layers.X.ffn.shared_experts.w1`             |
| `model.layers.X.mlp.shared_experts.up_proj`    | `layers.X.ffn.shared_experts.w3`             |
| `model.layers.X.mlp.shared_experts.down_proj`  | `layers.X.ffn.shared_experts.w2`             |

Implementation: header-only rewrite of the safetensors files. Each shard's
header is JSON; rewrite the keys, write the new header, copy the binary tensor
data unchanged. ~10 minutes for a 147 GB output.

We extended kylesayrs's existing `fix_checkpoint_keys.py` (in the
`kylesayrs/transformers-v5` branch) with the HC family renames + the
attention-projection renames. See `scripts/rewrite_for_vllm.py`.

The reference target structure is RedHat's published model
[`RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8`](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8) —
their `model.safetensors.index.json` uses exactly the flat native names vLLM
expects, so it's the canonical "what does the on-disk format look like" reference.

## Gap 2: `shared_experts.down_proj` save decomposition

This was the most surprising one. transformers v5's V4 implementation saves
`shared_experts.down_proj.weight` (a normal 2D `[hidden_size, moe_intermediate_size]`
tensor) as `hidden_size` separate keys named
`shared_experts.{0..hidden_size-1}.w2.weight.weight`, each containing one row of
the original tensor (shape `(moe_intermediate_size,)`, dtype bf16).

For V4-Flash that's 4096 keys per layer × 43 layers = 176 128 spurious tensors
where 43 should be. `gate_proj` and `up_proj` save normally (1 tensor each).
Only `down_proj` decomposes.

We didn't trace the cause inside transformers; we just refuse on the way out.
The diagnostic that confirmed this is recoverable was:

```python
from safetensors import safe_open
import glob
from collections import defaultdict
shapes = defaultdict(int)
for f in sorted(glob.glob("/path/to/model/*.safetensors")):
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            if "layers.0." in k and "shared_experts" in k:
                shapes[(tuple(sf.get_tensor(k).shape), str(sf.get_tensor(k).dtype))] += 1
print(shapes)
# {((2048,), 'torch.bfloat16'): 4096,
#  ((2048, 4096), 'torch.bfloat16'): 2}
```

4096 contiguous-indexed `(moe_intermediate_size,) bf16` rows, plus one whole
`gate_proj.weight` and one whole `up_proj.weight`. Stack the rows along dim 0 and
you get back `(hidden_size, moe_intermediate_size) = (4096, 2048) =
down_proj.weight`'s original shape. Save under the new name.

See `scripts/rewrite_for_vllm.py` for the refusion logic; it's grouped into the
same pass as the name rewrite. Detection regex:
`re:^(model\.layers\.\d+\.mlp\.shared_experts)\.(\d+)\.w2\.weight\.weight$`

## Gap 3: `quantization_config.ignore` rewrite

llm-compressor materializes the recipe's regex ignore patterns into a literal
list of full module paths at quant time, and writes that list into
`config.json`'s `quantization_config.ignore`. After our rename pass those paths
no longer match the renamed modules, so vLLM's compressed-tensors loader thinks
nothing is in ignore and quantizes everything (including modules that have only
BF16 weights on disk → KeyError on `.weight_packed` lookup).

Two options:

1. Rewrite the literal ignore paths through the same name rename map.
2. Replace the literal list with regex (`["lm_head", "re:.*attn.*",
   "re:.*shared_experts.*"]` covers what we want and survives the rename).

We use #2. See `scripts/patch_ignore_list.py`.

## Gap 4: BF16 attention is not supported by PR #41276

Per the `wo_scale_name` selector logic, attn weights must have either
`weight_scale_inv` (FP8) or `weight_scale` (compressed-tensors) attribute. If
neither, the model raises at init:

```
File "vllm/model_executor/layers/deepseek_v4_attention.py", line 380, in __init__
    raise NotImplementedError("DeepSeekV4 requires FP8 attention quantization")
```

Our first calibration recipe was

```python
recipe = GPTQModifier(
    config_groups={"default": QuantizationScheme(targets=["Linear"], **W4A16)},
    ignore=["lm_head", "re:.*self_attn.*", "re:.*shared_experts.*"],
    dampening_frac=0.1,
)
```

which produces BF16 attn weights (no scales). Hits the assertion.

Fix: drop `re:.*self_attn.*` from the ignore list. Attention gets W4A16 along with
the routed experts. Recipe becomes

```python
recipe = GPTQModifier(
    config_groups={"default": QuantizationScheme(targets=["Linear"], **W4A16)},
    ignore=["lm_head", "re:.*shared_experts.*"],
    dampening_frac=0.1,
)
```

This is consistent with what RedHat ships for their NVFP4-FP8 reference (their
recipe is configured by-targets rather than by-ignore, but the effect is the
same: attention is quantized, shared_experts stay BF16). See RedHat's published
config_groups for the exact regex they use:

```
targets: ['re:.*attn.*(wgate|wkv|wo_a|wo_b|wq_a|wq_b|fused_wkv_wgate|fused_wqa_wkv|gate_up_proj)$']
```

## Operational notes

- `/dev/shm` default 1 TB is insufficient for 8-rank torchrun calibration of
  the 543 GB BF16 V4-Flash. Resize to 1.8 TB:
  `sudo mount -o remount,size=1800G /dev/shm`. linearize_moe_model OOM'd at
  step 39/43 with the default.
- Isolate quant tooling in its own venv, not the DLAMI venv. `pip install
  --force-reinstall llmcompressor` cascade-downgrades torch to cu128 and
  breaks vLLM. Recovery is possible (cached wheels live under
  `/tmp/pip-unpack-*/`) but expensive in time.
- vLLM jasl/ds4-sm120 is editable-installed; .py edits are picked up live, no
  rebuild needed for quick patch iteration. Verify with
  `python -c "import vllm._C; print('OK')"` after edits.
- Cherry-pick PR #41276 directly: `git remote add kylesayrs
  https://github.com/neuralmagic/vllm.git; git fetch kylesayrs
  kylesayrs/deepseek-ct; git cherry-pick f910a73a93c54d3a3139d64add5da4624d619603`.
  Conflicts only on the `scale_fmt` line if you've hand-patched it; resolve in
  favor of upstream's defensive version.

## Key library versions used

- `torch==2.11.0+cu130` (cached wheel from the original DLAMI install)
- `transformers==5.8.0.dev0` from `huggingface/transformers@add-deepseek-v4`
- `compressed-tensors==0.15.1a20260428` (pre-release alpha required by kylesayrs branch)
- `llmcompressor==0.10.1.dev109+ga308bc0e` from `vllm-project/llm-compressor@kylesayrs/transformers-v5`
- `vllm==0.1.dev16267+g428e08ec2` (jasl ds4-sm120 + cherry-picked PR #41276)
- `safetensors==0.7.0`

## Process notes for anyone reproducing

1. Build vLLM jasl/ds4-sm120 in the DLAMI venv. Cherry-pick PR #41276.
2. Set up a separate venv for quantization. Install kylesayrs's llm-compressor
   editable + transformers from the add-deepseek-v4 branch + compressed-tensors
   pre-release.
3. Apply the helpers.py + modeling_deepseek_v4.py patches in that quant venv
   (see `patches/` folder).
4. Resize /dev/shm.
5. Dequantize the native FP4/FP8 model to BF16 with flagos `convert_weight.py`
   (`https://github.com/flagos-ai/DeepSeek-V4-FlagOS`). Patch the BF16 config:
   strip `quantization_config`, set `expert_dtype: bf16`.
6. Run quantize_v4_w4a16.py with the **attention-included** recipe. Outputs to
   `/workspace/model-w4a16-dryrun`.
7. Run rewrite_for_vllm.py on the output. Outputs to
   `/workspace/model-w4a16-dryrun-vllm`.
8. Run patch_ignore_list.py on the rewritten config.json (replace the literal
   ignore list with regex).
9. Patch other transformers-v5-style fields in config.json: ensure
   `compress_ratios`, `num_hash_layers`, `qk_rope_head_dim`, `torch_dtype`,
   `rope_scaling` are present; drop `rope_parameters`.
10. `vllm serve /workspace/model-w4a16-dryrun-vllm --tensor-parallel-size 8
    --kv-cache-dtype fp8 --tokenizer-mode deepseek_v4 --tool-call-parser
    deepseek_v4 --reasoning-parser deepseek_v4 --trust-remote-code
    --port 8002 ...`. Smoke + harness.

If anything new comes up, the `/workspace/output/REPORT.md` from this session
is the running log. Patches and helper scripts are committed to this repo.

## SHA rebase recovery (issue #1, 2026-05-08)

ZhouHr opened [issue #1](https://github.com/pasta-paul/dsv4-flash-w4a16-fp8/issues/1)
reporting a build failure on the live cherry-pick:

```
git remote add kylesayrs https://github.com/neuralmagic/vllm.git
git fetch --depth=200 kylesayrs kylesayrs/deepseek-ct
git cherry-pick f910a73a93
# fatal: bad revision 'f910a73a93'
```

The fetch succeeded but the cherry-pick failed because `f910a73a93` no longer
existed anywhere in the repository — Kyle had rebased the `kylesayrs/deepseek-ct`
branch and force-pushed, rewriting that commit. The original work survives as
the rebased commit `d09eeb4988acdeea17ab58eabe49197b11c6cc8a` ("support ct
quantization", 2026-05-01), with the same diff against neuralmagic mainline.

### Root cause

Pinning to a SHA on someone else's working branch is structurally fragile.
Working branches get rebased; force-pushes rewrite SHAs. We pinned the *name*
rather than the *content* and got bitten the first time Kyle iterated.

### Recovery: vendor the patch

`scripts/kylesayrs-deepseek-ct.patch` is now committed to this repo. It was
generated by:

```
# starting from a clean jasl/vllm@ds4-sm120 checkout:
git remote add nm https://github.com/neuralmagic/vllm.git
git fetch --depth=10 nm d09eeb4988acdeea17ab58eabe49197b11c6cc8a
git cherry-pick --no-commit d09eeb4988acdeea17ab58eabe49197b11c6cc8a
# clean auto-merge, no conflicts
git commit --author="Kyle Sayers <kylesayrs@gmail.com>" -m "support ct quantization ..."
git format-patch -1 HEAD --stdout > scripts/kylesayrs-deepseek-ct.patch
```

The patch is pre-rebased onto jasl/vllm@ds4-sm120, so `git apply --check`
succeeds with no 3-way merge needed. Authorship is preserved as Kyle Sayers.

### Dockerfile change

The cherry-pick block in `scripts/Dockerfile.dsv4-spark` was replaced with a
`COPY` + `git am` pair:

```dockerfile
COPY kylesayrs-deepseek-ct.patch /tmp/kylesayrs-deepseek-ct.patch
RUN git config --global user.email "builder@example.com" && \
    git config --global user.name "Docker Builder" && \
    git apply --check /tmp/kylesayrs-deepseek-ct.patch && \
    git am --keep-cr /tmp/kylesayrs-deepseek-ct.patch && \
    rm /tmp/kylesayrs-deepseek-ct.patch
```

No more network access to neuralmagic/vllm at build time. No more dependency
on Kyle's branch name or any specific commit on it.

### Commits intentionally NOT included

Kyle's branch has two newer commits on top of `d09eeb498`:

| SHA            | Subject                              | Date       | Why we skip |
|----------------|--------------------------------------|------------|-------------|
| `322ca2157`    | revoke support for fused_wkv_wgate   | 2026-05-07 | Our `patches/packed_modules_mapping.diff` declares `fused_wkv_wgate`; pulling this in requires also dropping that mapping entry. |
| `22f6da8c9`    | use config ignored_layers            | 2026-05-08 | Sits on top of `322ca2157`; not pulling that means not pulling this either. |

If Kyle's branch is eventually merged as an upstream PR, the merged form is
what we should track — not these in-progress commits individually.

### Preventing the next recurrence

Other parts of the Dockerfile still fetch from moving refs (FlashInfer `main`,
arbitrary `pull/$pr/head` for jasl-pinned PRs). They have weaker stability
needs than the kylesayrs patch — FlashInfer's `main` is push-only, not rebase,
and PR heads tend to be append-only too. But the same vendoring approach is
available for any of them if they bite us in the future.
