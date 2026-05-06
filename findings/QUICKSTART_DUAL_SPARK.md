# Quickstart — DSV4-Flash W4A16-FP8 on dual DGX Spark TP=2

End-to-end recipe to bring up `pastapaul/DeepSeek-V4-Flash-W4A16-FP8` on two DGX Spark GB10 boxes connected over QSFP, serving up to 1 M-token context graphs-ON.

The model is **public, self-contained, no HF token required**. You only need access to two DGX Spark GB10 nodes and the QSFP cable between them.

## 0. What you need

- **2× DGX Spark GB10** (SM 12.1a, 121 GiB UMA each, ARM64)
- **QSFP direct-connect** between the two boxes (RDMA-capable). No switch in the middle is fine.
- **~280 GiB free disk per box** (model weights + image cache)

No HF token needed for the quant. (The base `deepseek-ai/DeepSeek-V4-Flash` is gated, but you don't need it — the quant ships with all the tokenizer + config files it needs.)

## 1. Network — bring up the QSFP link

On both boxes, configure the QSFP NIC with a /30 between them and jumbo MTU. Example using `enp1s0f0np0`:

```bash
# On Spark A (head, rank 0)
sudo ip addr add 192.168.101.1/30 dev enp1s0f0np0
sudo ip link set dev enp1s0f0np0 mtu 9000
sudo ip link set dev enp1s0f0np0 up

# On Spark B (worker, rank 1)
sudo ip addr add 192.168.101.2/30 dev enp1s0f0np0
sudo ip link set dev enp1s0f0np0 mtu 9000
sudo ip link set dev enp1s0f0np0 up

# Verify from A:
ping -c 3 192.168.101.2          # should be < 1 ms RTT
```

Persist the config (netplan / NetworkManager / `/etc/network/interfaces`).

## 2. Pre-download the model on **both** boxes

The model is 142 GiB. Each Spark loads from its own local HF cache.

```bash
# On EACH Spark (no token needed — public quant)
huggingface-cli download pastapaul/DeepSeek-V4-Flash-W4A16-FP8
```

Disk path: `~/.cache/huggingface/hub/models--pastapaul--DeepSeek-V4-Flash-W4A16-FP8/`.

## 3. Build the image

The image is `vllm-w4a16-dsv4:exp` built from `jasl/vllm@ds4-sm120-experimental` plus a kylesayrs PR #41276 cherry-pick and a small `packed_modules_mapping` patch. Two ways:

### 3a. Build it yourself (recommended — ~25–40 min on a Spark)

```bash
# On Spark A (or a dev box that can scp to both Sparks)
git clone https://github.com/eugr/spark-vllm-docker
cd spark-vllm-docker

# Get our DSV4-specific Dockerfile + patch script
curl -O https://raw.githubusercontent.com/pasta-paul/dsv4-flash-w4a16-fp8/main/scripts/Dockerfile.dsv4-spark
curl -O https://raw.githubusercontent.com/pasta-paul/dsv4-flash-w4a16-fp8/main/scripts/patch_v4_packed_mapping.py
mv Dockerfile.dsv4-spark Dockerfile

./build-and-copy.sh \
  -t vllm-w4a16-dsv4:exp \
  --vllm-ref ds4-sm120-experimental \
  --rebuild-vllm \
  -c <other-spark-hostname> \
  --full-log
```

`-c <hostname>` will `docker save | scp | docker load` the built image to the second Spark automatically. Image is ~20 GB.

The Dockerfile applies these on top of `jasl/vllm@ds4-sm120-experimental`:
- Cherry-picks `f910a73a93` from `neuralmagic/vllm@kylesayrs/deepseek-ct` (vLLM PR #41276 — DSV4 compressed-tensors attention)
- Runs `patch_v4_packed_mapping.py` to inject `packed_modules_mapping` into `DeepseekV4ForCausalLM` (kylesayrs PR references but doesn't define this attr)
- Skips the workspace pre-reservation patch (now upstream as `jasl@1d6f5c4`)

### 3b. Skip the build, ask for the OCI tarball

If you'd rather not build, ping us on the [HF model discussions](https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8/discussions) — happy to share the `:exp` tarball directly.

## 4. Launch — head + worker

The launch command is identical on both boxes except `--node-rank` and `--headless`.

**Spark A** is rank 0 (the API server lives here). **Spark B** is rank 1 (`--headless` — it waits for RPCs from rank 0). Note `--served-model-name` takes **multiple values in one flag**, not a repeated flag — vLLM only honors the last value of repeated flags.

```bash
# Common — defined once, used on each box
ENGINE_CMD=(
  vllm serve pastapaul/DeepSeek-V4-Flash-W4A16-FP8
  --served-model-name DSV4-W4A16-FP8 deepseek-ai/DeepSeek-V4-Flash deepseek-v4-flash
  --trust-remote-code
  --kv-cache-dtype fp8 --block-size 256
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
  --max-model-len 1048576
  --max-num-seqs 1 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.90
  --host 0.0.0.0 --port 8888
  -tp 2 --nnodes 2
  --master-addr 192.168.101.1 --master-port 29501
)

ENV_FLAGS=(
  -e VLLM_TRITON_MLA_SPARSE=1
  -e VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=4
  -e VLLM_RPC_TIMEOUT=600000
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600
  -e TILELANG_CLEANUP_TEMP_FILES=1
  -e HF_HUB_OFFLINE=1
  -e NCCL_IB_DISABLE=0
  -e NCCL_NET_PLUGIN=none
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1
  -e NCCL_IB_MERGE_NICS=0
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0
)

DOCKER_FLAGS=(
  docker run -d --name vllm_node
  --gpus all --network=host --ipc=host
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
)
```

### On Spark A (rank 0, head, exposes API on `:8888`)

```bash
"${DOCKER_FLAGS[@]}" "${ENV_FLAGS[@]}" -e VLLM_HOST_IP=192.168.101.1 \
  vllm-w4a16-dsv4:exp \
  bash -c "$(printf '%q ' "${ENGINE_CMD[@]}" --node-rank 0)"
```

### On Spark B (rank 1, worker, headless)

```bash
"${DOCKER_FLAGS[@]}" "${ENV_FLAGS[@]}" -e VLLM_HOST_IP=192.168.101.2 \
  vllm-w4a16-dsv4:exp \
  bash -c "$(printf '%q ' "${ENGINE_CMD[@]}" --node-rank 1 --headless)"
```

## 5. Wait for `/health=200`

```bash
# From Spark A (or any client that can reach :8888)
until curl -sf http://localhost:8888/health > /dev/null; do
  echo "engine still booting..."
  sleep 30
done
echo "ready"

# All three names should appear in /v1/models:
curl http://localhost:8888/v1/models | jq '.data[].id'
# -> "DSV4-W4A16-FP8"
# -> "deepseek-ai/DeepSeek-V4-Flash"
# -> "deepseek-v4-flash"
```

Boot at 1 M-context graphs-ON takes ~5–7 min: ~2:30 weight load, ~2:30 KV-profile + graph capture.

## 6. First requests

Any of the three names work. We use `deepseek-v4-flash` below.

### Plain chat

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "What is 7*8?"}],
    "max_tokens": 50,
    "temperature": 0
  }'
```

### Think-max (heavy reasoning)

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "If a train leaves Chicago at 9:14 AM averaging 73 mph and another leaves Pittsburgh at 11:02 AM averaging 81 mph along the same route, when do they meet?"}],
    "thinking": {"type": "enabled"},
    "reasoning_effort": "max",
    "max_tokens": 32000,
    "temperature": 1.0
  }'
```

For `think-high`, change `"reasoning_effort": "max"` → `"high"`. For non-thinking, set `"thinking": {"type": "disabled"}` and drop `reasoning_effort`.

### Tool calling

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "What is the weather in Paris and Tokyo?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto",
    "max_tokens": 500,
    "temperature": 0.3
  }'
```

The model emits parallel `tool_calls` (one per city) with structured `arguments` JSON. Hand them to your tool runner, then send the results back as `role: "tool"` messages to continue the loop.

## 7. Operational notes

- **TP=2 is the only supported topology.** TP=1 OOMs even on 141 GB H200; TP≥4 hits an upstream `compressed-tensors` W4A16 MoE scale-sharding bug ([vllm-project/vllm#41511](https://github.com/vllm-project/vllm/issues/41511)).
- **`max-num-seqs=1` at 1 M context.** Multi-stream at long context exceeds 121 GiB UMA per Spark. If you want multi-stream, drop the context: `max-model-len=262144 + max-num-seqs=2` (Phase 4d's previous canonical) is the well-tested versatile alternative.
- **`--gpu-memory-utilization=0.90` not 0.92.** The experimental build's prefix-cache + split-KV paths reserve a little more memory at startup; 0.92 trips the boundary on first boot.
- **`--served-model-name` takes multiple values per flag, not a repeated flag.** Use `--served-model-name A B C` (space-separated). vLLM `nargs='+'` semantics mean `--served-model-name A --served-model-name B --served-model-name C` only keeps `C` and the others vanish.
- **`VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH` not needed on the experimental branch** — `cb60a48` enables SM12x sparse-MLA cudagraph by default.
- **Worker rank 1 must run with `--headless`.** Without it the worker tries to initialize its own engine and asserts `collective_rpc should not be called on follower node`.
- **Throughput at 1 M × 1**: smoke ~12 t/s, think-max sustained ~14–15 t/s. NIAH retrieval at 200 K-token haystack: 4/4 positions found.
- **Reasoning modes**: all three (`non-thinking`, `think-high`, `think-max`) work without server-side changes — clients select per-request via `thinking` and `reasoning_effort` fields.
- **OpenAI client** (`openai` Python lib) works — `base_url=http://<head-ip>:8888/v1`, `api_key="not-required"`.

## 8. Stopping / restarting

```bash
# Stop both
ssh spark_a 'docker rm -f vllm_node'
ssh spark_b 'docker rm -f vllm_node'

# Restart the same recipe — re-run §4 launch command on each box
```

## 9. Reference docs

- Full validation report: [`findings/spark_tp2_deployment.md`](spark_tp2_deployment.md)
- Phase 4e (this canonical): [`findings/spark_tp2_deployment.md#phase-4e`](spark_tp2_deployment.md#phase-4e--production-canonical-at-1-m-context-on-ds4-sm120-experimental-2026-05-06)
- Build context: [`scripts/Dockerfile.dsv4-spark`](../scripts/Dockerfile.dsv4-spark) + [`scripts/patch_v4_packed_mapping.py`](../scripts/patch_v4_packed_mapping.py)
- Raw evidence files: `findings/spark_tp2_phase4e_*.json` and `findings/spark_tp2_phase4e_probes/`
- Upstream PR: [`vllm-project/vllm#40991`](https://github.com/vllm-project/vllm/pull/40991) (DSV4 SM12x merge, includes our [Spark validation comment](https://github.com/vllm-project/vllm/pull/40991#issuecomment-4385208090))
- Closed issue (workspace allocator bug we filed and got upstreamed): [`#41700`](https://github.com/vllm-project/vllm/issues/41700)
