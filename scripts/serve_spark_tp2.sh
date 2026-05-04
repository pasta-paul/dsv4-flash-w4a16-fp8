#!/usr/bin/env bash
# Serve pastapaul/DeepSeek-V4-Flash-W4A16-FP8 on dual DGX Spark GB10 (SM 12.1a)
# at TP=2 over QSFP RDMA. Two flags differ from the H200 recipe and are
# load-bearing on Spark UMA:
#
#   --enforce-eager          // workspace-lock workaround (see Known issues)
#   --headless               // worker rank only; head omits this
#
# Plus 0.92 gpu-mem-util (0.85 leaves no headroom for the attention workspace).
#
# Container image must contain:
#   - jasl/vllm@428e08e + cherry-picked neuralmagic/kylesayrs/deepseek-ct@f910a73a93
#     + scripts/patch_v4_packed_mapping.py
#   - transformers==5.8.0.dev0 (HF main; PR #45643 add-deepseek-v4 was merged
#     2026-05-02 and the branch deleted — install from main, not the branch)
#   - compressed-tensors==0.15.1.a20260428
#   - PyTorch 2.11.0+cu130
#
# Usage:
#   On the head node (rank 0):
#     HEAD_IP=192.168.x.y NODE_RANK=0 ./serve_spark_tp2.sh
#   On the worker node (rank 1):
#     HEAD_IP=192.168.x.y NODE_RANK=1 HEADLESS=1 ./serve_spark_tp2.sh
#
# Both nodes must reach the model snapshot at the same local path
# ($HF_HOME/hub/models--pastapaul--DeepSeek-V4-Flash-W4A16-FP8/snapshots/<rev>/)
# either via shared NFS or pre-rsync over the QSFP link.

set -euo pipefail

MODEL="${MODEL:-pastapaul/DeepSeek-V4-Flash-W4A16-FP8}"
SERVED_NAME="${SERVED_NAME:-deepseek-v4-flash}"
HEAD_IP="${HEAD_IP:?set HEAD_IP to the head-node QSFP IP}"
MASTER_PORT="${MASTER_PORT:-29501}"
NODE_RANK="${NODE_RANK:-0}"
NNODES="${NNODES:-2}"
PORT="${PORT:-8888}"
HEADLESS_FLAG=""
if [[ "${NODE_RANK}" -gt 0 || "${HEADLESS:-0}" == "1" ]]; then
  HEADLESS_FLAG="--headless"
fi

vllm serve "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --enforce-eager \
  --max-model-len 16384 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.92 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  -tp 2 \
  --nnodes "${NNODES}" \
  --master-addr "${HEAD_IP}" \
  --master-port "${MASTER_PORT}" \
  --node-rank "${NODE_RANK}" \
  ${HEADLESS_FLAG}
