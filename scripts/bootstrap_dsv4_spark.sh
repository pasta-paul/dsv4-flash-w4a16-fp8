#!/usr/bin/env bash
# bootstrap_dsv4_spark.sh — zero-to-serving for DSV4-Flash W4A16-FP8 on dual DGX Spark TP=2.
#
# Run from anywhere with SSH access to both Sparks. Idempotent — safe to re-run.
#
# Usage:
#   ./bootstrap_dsv4_spark.sh \
#     --head-host spark-a \
#     --worker-host spark-b \
#     [--head-qsfp-ip 192.168.101.1] \
#     [--worker-qsfp-ip 192.168.101.2] \
#     [--qsfp-ifname enp1s0f0np0] \
#     [--vllm-ref ds4-sm120-experimental] \
#     [--image-tag vllm-w4a16-dsv4:exp] \
#     [--max-model-len 1048576] \
#     [--max-num-seqs 1] \
#     [--gpu-mem-util 0.90] \
#     [--ssh-user $USER] \
#     [--skip-build] [--skip-network] [--skip-download]
#
# What it does (in order):
#   1. SSH-reachability check for both hosts
#   2. Pre-download pastapaul/DeepSeek-V4-Flash-W4A16-FP8 on both (always idempotent via huggingface-cli)
#   3. Configure QSFP /30 on both (skippable)
#   4. Build vllm-w4a16-dsv4:exp on the HEAD box (skippable; uses build-and-copy.sh)
#   5. Distribute the image to WORKER (folded into step 4)
#   6. Stop any existing vllm_node containers
#   7. Launch worker rank 1 (--headless)
#   8. Launch head rank 0
#   9. Wait for /health=200, print build provenance + smoke test command

set -euo pipefail

# ---------------- defaults ----------------
HEAD_HOST=""
WORKER_HOST=""
HEAD_QSFP_IP="192.168.101.1"
WORKER_QSFP_IP="192.168.101.2"
QSFP_IFNAME="enp1s0f0np0"
VLLM_REF="ds4-sm120-experimental"
IMAGE_TAG="vllm-w4a16-dsv4:exp"
MAX_MODEL_LEN="1048576"
MAX_NUM_SEQS="1"
GPU_MEM_UTIL="0.90"
SSH_USER="${USER}"
SKIP_BUILD=0
SKIP_NETWORK=0
SKIP_DOWNLOAD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --head-host)        HEAD_HOST="$2"; shift 2;;
    --worker-host)      WORKER_HOST="$2"; shift 2;;
    --head-qsfp-ip)     HEAD_QSFP_IP="$2"; shift 2;;
    --worker-qsfp-ip)   WORKER_QSFP_IP="$2"; shift 2;;
    --qsfp-ifname)      QSFP_IFNAME="$2"; shift 2;;
    --vllm-ref)         VLLM_REF="$2"; shift 2;;
    --image-tag)        IMAGE_TAG="$2"; shift 2;;
    --max-model-len)    MAX_MODEL_LEN="$2"; shift 2;;
    --max-num-seqs)     MAX_NUM_SEQS="$2"; shift 2;;
    --gpu-mem-util)     GPU_MEM_UTIL="$2"; shift 2;;
    --ssh-user)         SSH_USER="$2"; shift 2;;
    --skip-build)       SKIP_BUILD=1; shift;;
    --skip-network)     SKIP_NETWORK=1; shift;;
    --skip-download)    SKIP_DOWNLOAD=1; shift;;
    -h|--help)          sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[[ -z "$HEAD_HOST" || -z "$WORKER_HOST" ]] && { echo "--head-host and --worker-host required"; exit 2; }

H="${SSH_USER}@${HEAD_HOST}"
W="${SSH_USER}@${WORKER_HOST}"
SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
DSV4_REPO_RAW="https://raw.githubusercontent.com/pasta-paul/dsv4-flash-w4a16-fp8/main"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Failure diagnostics — called when a container exits before /health=200.
# Dumps enough state for the user to file a useful bug report.
dump_failure_diag() {
  local which="$1"   # "head" or "worker"
  local target_host
  if [[ "$which" == "head" ]]; then target_host="$H"; else target_host="$W"; fi
  echo ""
  echo "================================================================"
  echo "FAILURE DIAGNOSTICS — ${which} (${target_host})"
  echo "================================================================"
  echo "--- last 300 lines of vllm_node logs ---"
  ssh $SSH_OPTS "$target_host" 'docker logs --tail 300 vllm_node 2>&1' || true
  echo ""
  echo "--- container env (VLLM_*, NCCL_*, TILELANG_*, HF_*, TORCH_*) ---"
  ssh $SSH_OPTS "$target_host" "docker exec vllm_node sh -c 'env | grep -E \"^(VLLM|NCCL|TILELANG|HF|TORCH)_\" | sort' 2>/dev/null || \
    docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' vllm_node 2>/dev/null | grep -E '^(VLLM|NCCL|TILELANG|HF|TORCH)_' | sort" || true
  echo ""
  echo "--- build provenance from image ---"
  ssh $SSH_OPTS "$target_host" "docker run --rm --entrypoint cat ${IMAGE_TAG} /workspace/build-metadata.yaml 2>/dev/null || echo '(no build-metadata.yaml — image predates instrumentation)'" || true
  echo ""
  echo "--- nvidia-smi ---"
  ssh $SSH_OPTS "$target_host" 'nvidia-smi 2>&1 | head -40' || true
  echo ""
  echo "--- dmesg tail (last 50 lines, may need sudo) ---"
  ssh $SSH_OPTS "$target_host" 'sudo -n dmesg 2>/dev/null | tail -50 || dmesg 2>/dev/null | tail -50 || echo "(dmesg unavailable without sudo)"' || true
  echo "================================================================"
}

# ---------------- 1. SSH reach ----------------
log "[1/9] SSH reachability check..."
ssh $SSH_OPTS "$H" 'true' || { echo "cannot SSH to head $H"; exit 3; }
ssh $SSH_OPTS "$W" 'true' || { echo "cannot SSH to worker $W"; exit 3; }
log "  ok — both Sparks reachable"

# ---------------- 2. Pre-download model ----------------
# Always invoke huggingface-cli download — it's idempotent and resumes via xet.
# The previous "is any .safetensors present?" gate passed on half-cached models
# and produced engines that crashed in kernel warmup on corrupt KV-cache tensors.
if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
  log "[2/9] Ensuring pastapaul/DeepSeek-V4-Flash-W4A16-FP8 is fully cached on both Sparks (~143 GiB)..."
  for HOST in "$H" "$W"; do
    ssh $SSH_OPTS "$HOST" '
      set -e
      command -v huggingface-cli >/dev/null 2>&1 || pip install --quiet --user huggingface_hub
      huggingface-cli download pastapaul/DeepSeek-V4-Flash-W4A16-FP8 >/dev/null
      echo "  [$(hostname)] cache verified"
    '
  done
else
  log "[2/9] skipping model download (--skip-download)"
fi

# ---------------- 3. QSFP network ----------------
if [[ $SKIP_NETWORK -eq 0 ]]; then
  log "[3/9] Configuring QSFP /30 on both Sparks..."
  ssh $SSH_OPTS "$H" "
    sudo ip addr replace ${HEAD_QSFP_IP}/30 dev ${QSFP_IFNAME}
    sudo ip link set dev ${QSFP_IFNAME} mtu 9000
    sudo ip link set dev ${QSFP_IFNAME} up
  "
  ssh $SSH_OPTS "$W" "
    sudo ip addr replace ${WORKER_QSFP_IP}/30 dev ${QSFP_IFNAME}
    sudo ip link set dev ${QSFP_IFNAME} mtu 9000
    sudo ip link set dev ${QSFP_IFNAME} up
  "
  log "  verifying connectivity..."
  ssh $SSH_OPTS "$H" "ping -c 2 -W 1 ${WORKER_QSFP_IP}" || { echo "QSFP link not reachable"; exit 4; }
  log "  ok — QSFP up, < 1 ms RTT"
else
  log "[3/9] skipping network setup (--skip-network)"
fi

# ---------------- 4+5. Build + distribute image ----------------
if [[ $SKIP_BUILD -eq 0 ]]; then
  log "[4-5/9] Building ${IMAGE_TAG} on ${HEAD_HOST} from jasl/vllm@${VLLM_REF} + vendored kylesayrs patch + packed_modules patch..."
  log "        (~25-40 min on a Spark; image ships to worker via docker save | scp | docker load)"
  ssh $SSH_OPTS "$H" "
    set -e
    BUILD_DIR=\$HOME/dsv4-spark-build
    mkdir -p \$BUILD_DIR && cd \$BUILD_DIR

    # eugr/spark-vllm-docker provides the build-and-copy.sh wrapper, ccache,
    # and the standard vllm/flashinfer build steps.
    if [[ ! -d spark-vllm-docker ]]; then
      git clone --depth 1 https://github.com/eugr/spark-vllm-docker
    fi
    cd spark-vllm-docker

    # Pull our DSV4-specific Dockerfile + both patches that the Dockerfile expects
    # to be in the build context. Without kylesayrs-deepseek-ct.patch the Dockerfile
    # fails at the COPY step with 'kylesayrs-deepseek-ct.patch: not found' (HF #4).
    curl -fsSL -o Dockerfile                  '${DSV4_REPO_RAW}/scripts/Dockerfile.dsv4-spark'
    curl -fsSL -o kylesayrs-deepseek-ct.patch '${DSV4_REPO_RAW}/scripts/kylesayrs-deepseek-ct.patch'
    curl -fsSL -o patch_v4_packed_mapping.py  '${DSV4_REPO_RAW}/scripts/patch_v4_packed_mapping.py'

    # Build + copy in one shot.
    ./build-and-copy.sh \
      -t '${IMAGE_TAG}' \
      --vllm-ref '${VLLM_REF}' \
      --rebuild-vllm \
      -c '${WORKER_HOST}' \
      --full-log
  "
  log "  build done — image present on both Sparks"
else
  log "[4-5/9] skipping build (--skip-build)"
fi

# ---------------- 6. Stop existing containers ----------------
log "[6/9] Stopping any existing vllm_node containers..."
ssh $SSH_OPTS "$H" 'docker rm -f vllm_node 2>/dev/null || true'
ssh $SSH_OPTS "$W" 'docker rm -f vllm_node 2>/dev/null || true'
sleep 2

# ---------------- 7+8. Launch ----------------
ENV_FLAGS=(
  -e "VLLM_TRITON_MLA_SPARSE=1"
  -e "VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=4"
  -e "VLLM_RPC_TIMEOUT=600000"
  -e "VLLM_ENGINE_READY_TIMEOUT_S=3600"
  -e "TILELANG_CLEANUP_TEMP_FILES=1"
  -e "HF_HUB_OFFLINE=1"
  -e "NCCL_IB_DISABLE=0"
  -e "NCCL_NET_PLUGIN=none"
  -e "NCCL_IB_SUBNET_AWARE_ROUTING=1"
  -e "NCCL_IB_MERGE_NICS=0"
  -e "GLOO_SOCKET_IFNAME=${QSFP_IFNAME}"
  -e "NCCL_SOCKET_IFNAME=${QSFP_IFNAME}"
)

DOCKER_FLAGS=(
  docker run -d --name vllm_node
  --gpus all --network=host --ipc=host
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864
  -v "\$HOME/.cache/huggingface:/root/.cache/huggingface"
)

ENGINE_CMD=(
  vllm serve pastapaul/DeepSeek-V4-Flash-W4A16-FP8
  --served-model-name DSV4-W4A16-FP8 deepseek-ai/DeepSeek-V4-Flash deepseek-v4-flash
  --trust-remote-code
  --kv-cache-dtype fp8 --block-size 256
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}" --max-num-batched-tokens 8192
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --host 0.0.0.0 --port 8888
  -tp 2 --nnodes 2
  --master-addr "${HEAD_QSFP_IP}" --master-port 29501
)

# Worker first (rank 1, headless) — it waits for the head to broadcast RPC.
log "[7/9] Launching worker rank 1 on ${WORKER_HOST} (headless)..."
ssh $SSH_OPTS "$W" "
  ${DOCKER_FLAGS[*]} ${ENV_FLAGS[*]} -e VLLM_HOST_IP=${WORKER_QSFP_IP} \
    ${IMAGE_TAG} \
    bash -c '$(printf '%q ' "${ENGINE_CMD[@]}" --node-rank 1 --headless)'
"
sleep 5

log "[8/9] Launching head rank 0 on ${HEAD_HOST} (API server on :8888)..."
ssh $SSH_OPTS "$H" "
  ${DOCKER_FLAGS[*]} ${ENV_FLAGS[*]} -e VLLM_HOST_IP=${HEAD_QSFP_IP} \
    ${IMAGE_TAG} \
    bash -c '$(printf '%q ' "${ENGINE_CMD[@]}" --node-rank 0)'
"

# ---------------- 9. Wait for /health ----------------
log "[9/9] Waiting for /health=200 on http://${HEAD_HOST}:8888 (~5-7 min cold start)..."
WAITED=0
until ssh $SSH_OPTS "$H" "curl -sf http://localhost:8888/health > /dev/null"; do
  WAITED=$((WAITED + 30))
  if [[ $WAITED -gt 1800 ]]; then
    echo "engine boot timeout (30 min) — dumping diagnostics from both nodes:"
    dump_failure_diag head
    dump_failure_diag worker
    exit 5
  fi
  HEAD_STATE=$(ssh $SSH_OPTS "$H" 'docker inspect --format "{{.State.Status}}" vllm_node 2>/dev/null || echo missing')
  WORKER_STATE=$(ssh $SSH_OPTS "$W" 'docker inspect --format "{{.State.Status}}" vllm_node 2>/dev/null || echo missing')
  if [[ "$HEAD_STATE" == *exited* ]]; then
    echo "engine container exited on head:"
    dump_failure_diag head
    echo ""
    echo "(also dumping worker state for context — it may have died first and taken head with it)"
    dump_failure_diag worker
    exit 6
  fi
  if [[ "$WORKER_STATE" == *exited* ]]; then
    echo "engine container exited on worker:"
    dump_failure_diag worker
    echo ""
    echo "(also dumping head state for context)"
    dump_failure_diag head
    exit 6
  fi
  log "  still booting (${WAITED}s elapsed, head=${HEAD_STATE}, worker=${WORKER_STATE})..."
  sleep 30
done

log ""
log "========================================================================"
log "ENGINE READY ✓"
log "  endpoint:        http://${HEAD_HOST}:8888/v1"
log "  model names:     DSV4-W4A16-FP8 | deepseek-ai/DeepSeek-V4-Flash | deepseek-v4-flash"
log "  context:         ${MAX_MODEL_LEN} tokens, max-num-seqs=${MAX_NUM_SEQS}"
log "  vllm ref:        jasl/vllm@${VLLM_REF}"
log "  image:           ${IMAGE_TAG}"
log "========================================================================"

# Build provenance — write to a local file for bug reports, also print key fields.
PROV_FILE="/tmp/dsv4-spark-build-metadata-$(date +%Y%m%d-%H%M%S).yaml"
if ssh $SSH_OPTS "$H" "docker exec vllm_node cat /workspace/build-metadata.yaml" > "$PROV_FILE" 2>/dev/null && [[ -s "$PROV_FILE" ]]; then
  log ""
  log "Build provenance written to ${PROV_FILE}:"
  sed 's/^/  /' "$PROV_FILE"
  log ""
  log "If you hit issues, please paste the contents of ${PROV_FILE} with your bug report."
else
  log ""
  log "(no /workspace/build-metadata.yaml in this image — likely a pre-instrumentation build)"
  rm -f "$PROV_FILE"
fi

log ""
log "Smoke test:"
log ""
cat <<EOF
  curl http://${HEAD_HOST}:8888/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{
      "model": "deepseek-v4-flash",
      "messages": [{"role": "user", "content": "What is 7*8?"}],
      "max_tokens": 50,
      "temperature": 0
    }'
EOF
log ""
log "More examples (think-max, tool calling): findings/QUICKSTART_DUAL_SPARK.md §6"
