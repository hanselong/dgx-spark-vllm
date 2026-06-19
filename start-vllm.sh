#!/usr/bin/env bash
set -euo pipefail

: "${VLLM_MODEL:?set VLLM_MODEL in .env}"

args=(
  vllm serve
  --model "${VLLM_MODEL}"
  --host 0.0.0.0
  --port 8000
  --served-model-name "${VLLM_SERVED_MODEL_NAME:-${VLLM_MODEL}}"
  --max-model-len "${VLLM_MAX_MODEL_LEN:-65536}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.65}"
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-3}"
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE:-1}"
  --attention-backend "${VLLM_ATTENTION_BACKEND:-flashinfer}"
  --disable-uvicorn-access-log
)

[ "${VLLM_TRUST_REMOTE_CODE:-true}" = "true" ] && args+=(--trust-remote-code)
[ "${VLLM_ENABLE_PREFIX_CACHING:-true}" = "true" ] && args+=(--enable-prefix-caching)
[ "${VLLM_ENABLE_CHUNKED_PREFILL:-true}" = "true" ] && args+=(--enable-chunked-prefill)
[ "${VLLM_ENABLE_AUTO_TOOL_CHOICE:-false}" = "true" ] && args+=(--enable-auto-tool-choice)
[ "${VLLM_KV_CACHE_METRICS:-true}" = "true" ] && args+=(--kv-cache-metrics)
[ "${VLLM_CUDAGRAPH_METRICS:-true}" = "true" ] && args+=(--cudagraph-metrics)
[ "${VLLM_ENABLE_MFU_METRICS:-true}" = "true" ] && args+=(--enable-mfu-metrics)

if [ -n "${VLLM_TOOL_CALL_PARSER:-}" ]; then
  args+=(--tool-call-parser "${VLLM_TOOL_CALL_PARSER}")
fi

if [ -n "${VLLM_REASONING_PARSER:-}" ]; then
  args+=(--reasoning-parser "${VLLM_REASONING_PARSER}")
fi

if [ -n "${VLLM_SPECULATIVE_CONFIG:-}" ]; then
  args+=(--speculative-config "${VLLM_SPECULATIVE_CONFIG}")
fi

if [ -n "${VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE:-}" ]; then
  args+=(--max-cudagraph-capture-size "${VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE}")
fi

exec "${args[@]}"
