# DGX Spark vLLM Compose Setup

Public, minimal Docker Compose setup for [NVIDIA DGX Spark / GB10](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) owners who want to run [vLLM](https://vllm.ai/) locally on the 128 GB unified-memory system. It was distilled from a working vLLM deployment and keeps the reusable inference pieces without private gateways, tunnels, or credentials.

This repo is intentionally [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)-first. The default image is `scitrera/dgx-spark-vllm:0.17.0-t5` because it is prebuilt for the GB10 / Blackwell target that generic upstream vLLM images may not handle well. The setup is still configurable enough to run other vLLM-compatible models by editing `.env`, but the defaults assume DGX Spark hardware.

## What You Get

- A single `vllm` service exposing the OpenAI-compatible API on DGX Spark.
- Persistent Hugging Face and vLLM caches under `./data/`.
- NVIDIA GPU access through Compose.
- Conservative localhost binding by default.
- Tunable model, context, batching, memory, parser, and speculative decoding settings.

## Requirements

- Docker Engine with the Compose plugin.
- NVIDIA driver and NVIDIA Container Toolkit.
- NVIDIA DGX Spark / GB10 with 128 GB unified memory.
- Enough free disk for model weights and caches.

The included defaults are tuned as a practical DGX Spark starting point:

- [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- 128K context by default, below the model's native 262K context to leave operational headroom
- `gpu-memory-utilization=0.65`
- `max-num-seqs=3` for modest concurrency on a single DGX Spark
- FlashInfer attention
- Qwen3 tool and reasoning parsers
- MTP speculative decoding disabled by default for stability

`Qwen/Qwen3.6-35B-A3B` first became available on Hugging Face on April 15, 2026. As of June 2026, it is the most reliable model we have found for this DGX Spark setup across agentic coding and general-purpose LLM use. Across about a month of use from local and remote clients, including PI coding agent, Qwen Coder, OpenClaw, and similar coding-agent workflows, this setup has consistently landed around 30 tokens/sec after startup and generally sub-1s time to first token, workload permitting. The A3B design matters here: the model has 35B total parameters but only about 3B active at a time, which helps keep latency and throughput practical while retaining stronger reasoning behavior.

## Quick Start

```bash
cp .env.example .env
docker compose up -d --build
docker compose logs -f vllm
```

The vLLM API binds to `127.0.0.1:8000` by default. vLLM serves OpenAI-compatible endpoints under `/v1`, so OpenAI SDK clients should use a base URL like `http://localhost:8000/v1`. It is not OpenAI-hosted and it does not add OpenAI API-key auth by itself.

```bash
curl http://localhost:8000/health

curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3-35b-a3b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

With the OpenAI Python SDK, point the client at vLLM's `/v1` base URL and use any placeholder API key unless you put an auth layer in front of it:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="qwen/qwen3-35b-a3b",
    messages=[{"role": "user", "content": "Reply with one short sentence."}],
    max_tokens=64,
)
print(response.choices[0].message.content)
```

## Public Exposure

Do not bind vLLM directly to the public internet. vLLM does not provide a production auth boundary by itself.

The compose file intentionally defaults to:

```env
VLLM_HOST=127.0.0.1
```

If you need remote access, put an authenticated access layer in front of it. Only change this to `0.0.0.0` when another layer is enforcing access control.

Common exposure patterns:

- Tailscale: simplest private sharing path for your own devices or a small trusted tailnet. Keep `VLLM_HOST=127.0.0.1` and proxy from the host or bind only to the Tailscale interface.
- Personal WireGuard tunnel: good when you already operate your own network and want explicit peer, firewall, and routing control. Bind carefully and restrict the port to tunnel peers.
- Cloudflare AI Gateway: useful when you want Cloudflare-managed observability, caching, rate limiting, retries, or provider-style routing in front of AI traffic. For self-hosted vLLM, still avoid exposing the raw origin directly; pair it with a secure origin path such as a tunnel, private network, or authenticated reverse proxy.

## Configuration

Edit `.env` after copying `.env.example`.

Common settings:

| Variable | Purpose |
| --- | --- |
| `VLLM_MODEL` | Hugging Face model ID or local model path. |
| `VLLM_SERVED_MODEL_NAME` | Name clients send in `model`. |
| `HUGGING_FACE_HUB_TOKEN` | Optional token for gated Hugging Face repositories. |
| `VLLM_MAX_MODEL_LEN` | Maximum context length. |
| `VLLM_GPU_MEMORY_UTILIZATION` | Fraction of DGX Spark unified memory vLLM may target; default `0.65`. |
| `VLLM_MAX_NUM_SEQS` | Maximum concurrent sequences; default `3` on DGX Spark. |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | Batching limit; lower this if startup or serving is unstable. |
| `VLLM_TOOL_CALL_PARSER` | Optional model-specific tool parser, for example `qwen3_coder`. |
| `VLLM_REASONING_PARSER` | Optional reasoning parser, for example `qwen3`. |
| `VLLM_SPECULATIVE_CONFIG` | Optional JSON string for speculative decoding. |

For non-Qwen models, start by clearing the Qwen-specific settings:

```env
VLLM_ENABLE_AUTO_TOOL_CHOICE=false
VLLM_TOOL_CALL_PARSER=
VLLM_REASONING_PARSER=
VLLM_SPECULATIVE_CONFIG=
VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE=
```

## Optional LiteLLM Proxy

[vLLM](https://vllm.ai/) already exposes OpenAI-compatible `/v1` endpoints, which is enough for many local clients. [LiteLLM](https://www.litellm.ai/) is worth adding if you want a front-door proxy with a stable OpenAI-style base URL, bearer-token auth, request normalization, provider routing, retries, logging hooks, or a place to add more models later.

Start vLLM plus LiteLLM with the optional profile:

```bash
cp .env.example .env
# Edit LITELLM_MASTER_KEY before exposing beyond localhost.
docker compose --profile litellm up -d --build
```

LiteLLM binds to `127.0.0.1:4000` by default and routes `qwen/qwen3-35b-a3b` to the `vllm` service internally. If you change the served model name, update `LITELLM_MODEL_NAME` and `LITELLM_VLLM_MODEL` in `.env` as well.

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3-35b-a3b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

Use LiteLLM as the base URL for OpenAI SDK clients when you want the proxy layer:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4000/v1", api_key="sk-change-me")
response = client.chat.completions.create(
    model="qwen/qwen3-35b-a3b",
    messages=[{"role": "user", "content": "Reply with one short sentence."}],
    max_tokens=64,
)
print(response.choices[0].message.content)
```

The [LiteLLM](https://www.litellm.ai/) config lives in `litellm_config.yaml`. It uses LiteLLM's `hosted_vllm/` provider route and talks to `http://vllm:8000` on the Compose network. The `hosted_vllm/` prefix is what tells LiteLLM to treat the upstream as a vLLM/OpenAI-compatible server.

## FAQ

### Why vLLM? What is the alternative?

[vLLM](https://vllm.ai/) is the right default here because this repo is about keeping a [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) available as a reliable inference service. It has stronger scheduling, batching, KV-cache management, metrics, and OpenAI-compatible serving behavior for long-running agentic workloads and multiple users. That matters for clients like OpenClaw, coding agents, and remote users where a single hung or restarted backend can waste a lot of work.

Ollama is still useful. It is excellent for local experimentation, quick model pulls, trying small models, and interactive single-user workflows. For a persistent shared endpoint with long contexts, concurrent requests, and coding-agent sessions that may run for a long time, vLLM has been the more reliable serving layer.

### Do I need LiteLLM?

No. Direct [vLLM](https://vllm.ai/) is the simplest path and already exposes OpenAI-compatible `/v1` endpoints. Use [LiteLLM](https://www.litellm.ai/) when you want a front-door proxy: bearer-token auth, a stable base URL, request normalization, logging hooks, retries, or future routing across more than one model/provider.

If you are only calling this from trusted local tools on the same machine or over a private network, direct vLLM is usually enough. If friends, teammates, or multiple apps will use it, LiteLLM is a reasonable next layer.

### Should I expose vLLM directly?

No. Keep vLLM bound to localhost unless another layer is protecting it. vLLM is an inference server, not an internet-facing auth gateway. Use Tailscale, WireGuard, Cloudflare AI Gateway plus a secure origin path, LiteLLM, or another authenticated proxy if the endpoint needs to be reached remotely.

### Why 128K context instead of the full model context?

The model supports a larger context than the default used here, but context length directly competes with memory, concurrency, and stability. A 128K window is already large enough for many agentic coding workflows while leaving room for three concurrent sequences and other services on the machine.

If you need a larger context, raise `VLLM_MAX_MODEL_LEN` gradually and watch what vLLM reports at startup. You may need to lower concurrency, reduce batching, or raise memory utilization.

### Can I use a different model?

Yes, but treat the defaults as model-specific rather than universal. The Qwen parser, reasoning parser, FlashInfer behavior, memory target, and context size were chosen for `Qwen/Qwen3.6-35B-A3B` on DGX Spark. For a different model, start conservatively: clear Qwen-specific parser settings, reduce context, and confirm health before increasing concurrency.

### Why bind to localhost by default?

Localhost is the least surprising safe default. It lets local clients and local proxies connect without accidentally publishing an unauthenticated inference endpoint on your LAN or the internet. Change `VLLM_HOST` or `LITELLM_HOST` only after deciding what access-control layer owns authentication.

### Why `gpu-memory-utilization=0.65`?

DGX Spark has 128 GB of unified memory, but this compose file intentionally does not hand all of it to vLLM. The `0.65` default leaves practical headroom for the host and for other local services you may want to run on the same machine, such as embeddings, image models, monitoring, proxies, or development tools.

If this box is dedicated only to vLLM, you can try raising `VLLM_GPU_MEMORY_UTILIZATION`. Watch startup logs, latency, swap pressure, and whether other services remain responsive.

### Why `max-num-seqs=3`?

`VLLM_MAX_NUM_SEQS` controls how many concurrent sequences vLLM is allowed to schedule. With this model, 128K context, and `gpu-memory-utilization=0.65`, vLLM reports roughly this level of supported concurrency on DGX Spark. The default is therefore conservative and aligned with what the engine can actually support in this memory budget.

If you want more concurrency, the main levers are:

1. Increase `VLLM_GPU_MEMORY_UTILIZATION`.
2. Decrease `VLLM_MAX_MODEL_LEN`.
3. Decrease `VLLM_MAX_NUM_BATCHED_TOKENS`.
4. Use a smaller model.
5. Accept lower per-request throughput under contention.

### How fast is it?

In real use from local and remote clients, this setup has consistently produced roughly 30 tokens/sec after about a month of coding-agent workloads. That includes PI coding agent, Qwen Coder, OpenClaw, and similar clients. Time to first token has generally been below one second, assuming the model is already loaded and the machine is not saturated.

Throughput will vary with prompt length, output length, concurrent requests, network path, cache state, and whether another GPU-heavy service is active. Treat 30 tokens/sec as a field-tested baseline for this specific DGX Spark + Qwen3.6-35B-A3B setup, not a universal benchmark.

### Does MTP speculative decoding work?

Not reliably enough to enable by default here. When MTP worked, it reached roughly 40 tokens/sec in testing, but it also tended to crash and restart vLLM. For a shared inference endpoint, stability is more valuable than the extra throughput, so `VLLM_SPECULATIVE_CONFIG` is blank in `.env.example`.

If you want to experiment, try:

```env
VLLM_SPECULATIVE_CONFIG={"method":"mtp","num_speculative_tokens":2}
VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE=256
```

Expect to watch logs closely and be ready to roll back if vLLM restarts under load.

## Local Models

Place local models under `./data/models`. The directory is mounted read-only at `/models`.

Example:

```env
VLLM_MODEL=/models/my-model
VLLM_SERVED_MODEL_NAME=my-model
```

## Caches

The setup keeps large generated/downloaded assets out of git:

- `./data/hf-cache` -> `/data/hf`
- `./data/vllm-cache` -> `/root/.cache/vllm`
- `./data/models` -> `/models:ro`

Rebuilding the container does not delete downloaded model weights.

## Operations

```bash
docker compose ps
docker compose logs -f vllm
docker compose restart vllm
docker compose down
```

To fully remove downloaded weights and compiled caches:

```bash
docker compose down
sudo rm -rf data/hf-cache data/vllm-cache
```

## Tuning Notes

If the container exits during startup or the host is under memory pressure:

1. Lower `VLLM_MAX_MODEL_LEN`.
2. Lower `VLLM_GPU_MEMORY_UTILIZATION`.
3. Lower `VLLM_MAX_NUM_BATCHED_TOKENS`.
4. Disable speculative decoding by clearing `VLLM_SPECULATIVE_CONFIG`.
5. Disable parser-specific features if the model does not support them.

For non-DGX Spark hardware, use a smaller model first and raise context/concurrency only after the health check passes.
