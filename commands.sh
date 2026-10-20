#!/usr/bin/env bash
# Start one GPU-policy x CPU-policy pair on one GPU.
#
# Usage:
#   serve_cross_tier.sh <gpu-policy> <cpu-policy> <gpu> <port> [model]
#
# GPU policies: lru | prefix | session | pin
# CPU policies: lru | arc | prefix | session | pin
#
# The two pairs to try first:
#   serve_cross_tier.sh lru pin 0 8000
#   serve_cross_tier.sh prefix session 0 8001
#
# CPU offload is always attached, so GPU LRU here is LRU on the GPU free
# list with a CPU tier, not the GPU-only server from serve.sh.
# CPU bytes default to 8GiB. Override with CPU_BYTES.
# Pick the GPU with CUDA_VISIBLE_DEVICES. When that variable is already
# set, the gpu argument is only a label for the log and cache directory.
#   CUDA_VISIBLE_DEVICES=3 ./benchmark/serve_cross_tier.sh lru lru 3 8000
set -euo pipefail

GPU_POLICY="${1:?gpu policy: lru|prefix|session|pin}"
CPU_POLICY="${2:?cpu policy: lru|arc|prefix|session|pin}"
GPU="${3:?gpu index, used only when CUDA_VISIBLE_DEVICES is unset}"
PORT="${4:?port}"
MODEL="${5:-mistralai/Ministral-3-8B-Instruct-2512}"

ROOT="/home/ubuntu/sneh/opensource/eviction"
PATCHED="$ROOT/vllm"
VLLM="$PATCHED/.venv/bin/vllm"
SEED=$(python3 -c 'import random; print(random.randint(0, 1000000))')
LOG="${VLLM_LOG_PATH:-$ROOT/benchmark/logs/gpu-${GPU_POLICY}_cpu-${CPU_POLICY}-gpu${GPU}-p${PORT}-${SEED}.log}"
mkdir -p "$(dirname "$LOG")" "$ROOT/benchmark/logs"

export PATH="$PATCHED/.venv/bin:$PATH"
unset http_proxy https_proxy all_proxy socks_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY SOCKS_PROXY
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export HF_TOKEN="${HF_TOKEN:-}"
export PYTHONPATH="$PATCHED"

CACHE_ROOT="$ROOT/.cache/gpu${GPU}_p${PORT}"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"

unset VLLM_SUFFIX_PRESERVING_EVICTION
unset VLLM_SUFFIX_EVICTION_TIER_TOKENS
unset VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS
unset VLLM_SUFFIX_EVICTION_MAX_TIERS
unset VLLM_SUFFIX_EVICTION_IDLE_TTL_S
unset VLLM_SUFFIX_EVICTION_IDLE_DEMOTE
unset VLLM_SUFFIX_EVICTION_PROTECT_TOKENS
unset VLLM_SUFFIX_SESSION_DEAD_TTL_S
unset VLLM_SUFFIX_SESSION_DETECTOR
unset VLLM_SUFFIX_SESSION_QUANTILE
unset VLLM_SUFFIX_SESSION_GAP_WINDOW
unset VLLM_SUFFIX_SESSION_EWA_ALPHA
unset VLLM_SUFFIX_SESSION_DEAD_K
unset VLLM_SUFFIX_SESSION_MIN_GAPS
unset VLLM_SESSION_TTL_LIMIT_S
unset VLLM_SESSION_TTL_K
unset VLLM_CPU_SESSION_TTL_S
unset VLLM_CPU_SESSION_TTL_K
unset VLLM_PIN_SYSTEM_PROMPT

case "$GPU_POLICY" in
  lru)
    export VLLM_SERVER_DEV_MODE=1
    export VLLM_SUFFIX_PRESERVING_EVICTION=0
    ;;
  prefix)
    export VLLM_SERVER_DEV_MODE=1
    export VLLM_SUFFIX_PRESERVING_EVICTION=1
    export VLLM_SUFFIX_EVICTION_TIER_TOKENS=16384
    export VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS=16384
    export VLLM_SUFFIX_EVICTION_MAX_TIERS=16
    export VLLM_SUFFIX_EVICTION_IDLE_TTL_S=60
    ;;
  session)
    export VLLM_SERVER_DEV_MODE=1
    export VLLM_SUFFIX_PRESERVING_EVICTION=1
    export VLLM_SUFFIX_EVICTION_TIER_TOKENS=16384
    export VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS=16384
    export VLLM_SUFFIX_EVICTION_MAX_TIERS=16
    export VLLM_SUFFIX_EVICTION_IDLE_TTL_S=60
    export VLLM_SUFFIX_SESSION_DEAD_TTL_S=30
    export VLLM_SUFFIX_SESSION_DETECTOR=ewa
    export VLLM_SUFFIX_SESSION_EWA_ALPHA=0.3
    export VLLM_SUFFIX_SESSION_DEAD_K=3
    export VLLM_SUFFIX_SESSION_MIN_GAPS=2
    ;;
  pin)
    export VLLM_SERVER_DEV_MODE=1
    export VLLM_SUFFIX_PRESERVING_EVICTION=1
    export VLLM_SUFFIX_EVICTION_TIER_TOKENS=16384
    export VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS=16384
    export VLLM_SUFFIX_EVICTION_MAX_TIERS=16
    export VLLM_SUFFIX_EVICTION_IDLE_TTL_S=60
    export VLLM_SESSION_TTL_LIMIT_S=60
    export VLLM_SESSION_TTL_K=2
    export VLLM_SUFFIX_SESSION_EWA_ALPHA=0.3
    ;;
  *)
    echo "unknown gpu policy $GPU_POLICY (lru|prefix|session|pin)" >&2
    exit 1
    ;;
esac

case "$CPU_POLICY" in
  lru) CPU_NAME=lru ;;
  arc) CPU_NAME=arc ;;
  prefix) CPU_NAME=prefix_pruning ;;
  session) CPU_NAME=prefix_pruning_on_sessions ;;
  pin) CPU_NAME=first_request_pin ;;
  *)
    echo "unknown cpu policy $CPU_POLICY (lru|arc|prefix|session|pin)" >&2
    exit 1
    ;;
esac

# Tier bound for the CPU session rule. K multiplies the idle-gap average.
# Leave the bound unset for plain CPU LRU, ARC, and prefix pruning.
if [[ "$CPU_POLICY" == "session" || "$CPU_POLICY" == "pin" ]]; then
  export VLLM_CPU_SESSION_TTL_S="${VLLM_CPU_SESSION_TTL_S:-120}"
  export VLLM_CPU_SESSION_TTL_K="${VLLM_CPU_SESSION_TTL_K:-2}"
fi

CPU_BYTES="${CPU_BYTES:-53687091200}"
# Tier sizes belong to the CPU prefix, session, and pin policies.
# LRU and ARC keep the connector config to bytes and the policy name.
# Offload chunk is 256 tokens, same as the LRU run that reloaded a full prompt.
if [[ "$CPU_POLICY" == "lru" || "$CPU_POLICY" == "arc" ]]; then
  EXTRA_CONFIG="{\"cpu_bytes_to_use\":${CPU_BYTES},\"block_size\":256,\"eviction_policy\":\"${CPU_NAME}\"}"
else
  EXTRA_CONFIG="{\"cpu_bytes_to_use\":${CPU_BYTES},\"block_size\":256,\"eviction_policy\":\"${CPU_NAME}\",\"first_tier_tokens\":16384,\"tier_step_tokens\":16384,\"max_tiers\":8}"
fi
KV_CONFIG=$(printf '%s' \
  "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\","\
"\"kv_connector_extra_config\":${EXTRA_CONFIG}}")

echo "gpu_policy=$GPU_POLICY cpu_policy=$CPU_NAME cuda_visible_devices=$CUDA_VISIBLE_DEVICES port=$PORT model=$MODEL"
echo "log=$LOG"
cd "$PATCHED"
exec "$VLLM" serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --block-size 256 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --max-model-len 262144 \
  --tensor-parallel-size 1 \
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  --kv-transfer-config "$KV_CONFIG" \
  >"$LOG" 2>&1