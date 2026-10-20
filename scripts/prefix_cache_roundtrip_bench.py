#!/usr/bin/env python3
"""
Prepare N unique-prefix prompts, dispatch them concurrently to a local vLLM
OpenAI-compatible /v1/chat/completions endpoint, measure TTFT / latency, wait,
call /reset_prefix_cache (requires VLLM_SERVER_DEV_MODE), then replay the same
requests and measure again.

Example:
  python prefix_cache_roundtrip_bench.py \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --concurrency 4 \
      --prefix-size 128048 \
      --output-len 64 \
      --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests
from transformers import AutoTokenizer


@dataclass
class PreparedRequest:
    request_id: str
    prompt: str
    prompt_len: int
    prefix_len: int


@dataclass
class RequestMetrics:
    request_id: str
    success: bool
    prompt_len: int = 0
    output_tokens: int = 0
    ttft_s: float = 0.0
    latency_s: float = 0.0
    tpot_s: float = 0.0
    itl_s: list[float] = field(default_factory=list)
    generated_text: str = ""
    error: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Unique-prefix prompts → concurrent chat.completions → "
            "reset_prefix_cache → replay"
        )
    )
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="vLLM OpenAI server base URL (default: %(default)s)",
    )
    p.add_argument(
        "--model",
        required=True,
        help="Model name as served by vLLM (also used for tokenizer unless overridden)",
    )
    p.add_argument(
        "--tokenizer",
        default=None,
        help="HuggingFace tokenizer id/path (default: same as --model)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help=(
            "Number of unique-prefix requests to prepare and fire in parallel "
            "(default: %(default)s)"
        ),
    )
    p.add_argument(
        "--prefix-size",
        type=int,
        required=True,
        help="Unique prefix length in tokens for each request",
    )
    p.add_argument(
        "--output-len",
        type=int,
        default=64,
        help="max_tokens / completion length (default: %(default)s)",
    )
    p.add_argument(
        "--wait-seconds",
        type=float,
        default=7.0,
        help="Seconds to wait before reset_prefix_cache (default: %(default)s)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: %(default)s)",
    )
    p.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ask the server to ignore EOS so output length is closer to --output-len",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-request HTTP timeout in seconds (default: %(default)s)",
    )
    p.add_argument(
        "--reset-running-requests",
        action="store_true",
        help="Pass reset_running_requests=true to /reset_prefix_cache",
    )
    return p.parse_args()


def build_hello_prefix_text(
    tokenizer: AutoTokenizer,
    prefix_size: int,
    *,
    request_index: int,
) -> str:
    """
    Build a prefix string of approximately `prefix_size` tokens by tokenizing
    "hello" and repeating it. A short unique marker is prepended so each
    concurrent request has a distinct prefix (avoids full KV sharing across
    the batch).
    """
    hello_ids = tokenizer.encode("hello", add_special_tokens=False)
    if not hello_ids:
        raise RuntimeError('tokenizer.encode("hello") returned no tokens')

    # Unique head so the N prepared requests are not identical.
    unique_ids = tokenizer.encode(
        f"id={request_index};", add_special_tokens=False
    )
    if len(unique_ids) >= prefix_size:
        prefix_ids = unique_ids[:prefix_size]
    else:
        need = prefix_size - len(unique_ids)
        reps = (need + len(hello_ids) - 1) // len(hello_ids)
        fill = (hello_ids * reps)[:need]
        prefix_ids = unique_ids + fill

    return tokenizer.decode(prefix_ids)


def build_prepared_requests(
    tokenizer: AutoTokenizer,
    *,
    concurrency: int,
    prefix_size: int,
) -> list[PreparedRequest]:
    """
    Build concurrency prompts, each with a unique prefix of ~prefix_size
    tokens (unique marker + repeated "hello"), plus a short query suffix.
    """
    if concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if prefix_size < 1:
        raise ValueError("--prefix-size must be >= 1")

    suffix = (
        "\n\n### Question\n"
        "Write a one-sentence summary of the content above."
    )
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    hello_ids = tokenizer.encode("hello", add_special_tokens=False)

    prepared: list[PreparedRequest] = []
    for i in range(concurrency):
        prefix_text = build_hello_prefix_text(
            tokenizer, prefix_size, request_index=i
        )
        prompt = prefix_text + suffix
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        req = PreparedRequest(
            request_id=f"req-{i}",
            prompt=prompt,
            prompt_len=len(prompt_ids),
            prefix_len=prefix_size,
        )
        prepared.append(req)
        print(
            f"  prepared {req.request_id}: "
            f"target_prefix_tokens={prefix_size}, "
            f"prompt_tokens={req.prompt_len}, "
            f"hello_tokens_per_unit={len(hello_ids)}, "
            f"suffix_tokens={len(suffix_ids)}"
        )
    return prepared


def send_chat_completion_streaming(
    *,
    base_url: str,
    model: str,
    prepared: PreparedRequest,
    output_len: int,
    temperature: float,
    ignore_eos: bool,
    timeout: float,
) -> RequestMetrics:
    """POST /v1/chat/completions with stream=True; measure TTFT and e2e latency."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "user", "content": prepared.prompt},
        ],
        "max_tokens": output_len,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True

    metrics = RequestMetrics(
        request_id=prepared.request_id,
        success=False,
        prompt_len=prepared.prompt_len,
    )

    generated: list[str] = []
    st = time.perf_counter()
    most_recent = st
    first_token_seen = False
    output_tokens = 0

    try:
        with requests.post(
            url,
            json=payload,
            stream=True,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status_code != 200:
                # Consume body for error text (non-streaming error responses).
                metrics.error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                return metrics

            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break

                data = json.loads(data_str)
                now = time.perf_counter()

                if usage := data.get("usage"):
                    if usage.get("completion_tokens") is not None:
                        output_tokens = usage["completion_tokens"]
                    if usage.get("prompt_tokens") is not None:
                        metrics.prompt_len = usage["prompt_tokens"]

                choices = data.get("choices") or []
                if not choices:
                    continue

                # Chat Completions streaming uses delta.content
                delta = choices[0].get("delta") or {}
                text = delta.get("content") or ""
                if text or choices[0].get("finish_reason"):
                    if not first_token_seen and text:
                        first_token_seen = True
                        metrics.ttft_s = now - st
                    elif first_token_seen and text:
                        metrics.itl_s.append(now - most_recent)
                    if text:
                        most_recent = now
                        generated.append(text)

            metrics.latency_s = most_recent - st
            metrics.generated_text = "".join(generated)
            if output_tokens <= 0:
                output_tokens = max(
                    1, len(metrics.itl_s) + (1 if first_token_seen else 0)
                )
            metrics.output_tokens = output_tokens

            if first_token_seen:
                metrics.success = True
                if output_tokens > 1:
                    metrics.tpot_s = (metrics.latency_s - metrics.ttft_s) / (
                        output_tokens - 1
                    )
                else:
                    metrics.tpot_s = 0.0
            else:
                metrics.error = "No token chunks received (cannot compute TTFT)"
    except Exception as exc:  # noqa: BLE001 — surface any client/server failure
        metrics.error = str(exc)

    return metrics


def reset_prefix_cache(
    base_url: str,
    *,
    reset_running_requests: bool = False,
    timeout: float = 60.0,
) -> bool:
    url = f"{base_url.rstrip('/')}/reset_prefix_cache"
    params = {}
    if reset_running_requests:
        params["reset_running_requests"] = "true"

    resp = requests.post(url, params=params, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"reset_prefix_cache failed: HTTP {resp.status_code}: {resp.text[:500]}"
        )
    body = resp.json()
    success = bool(body.get("success", False))
    print(f"  /reset_prefix_cache → {body}")
    return success


def start_profile(base_url: str, *, timeout: float = 60.0) -> None:
    url = f"{base_url.rstrip('/')}/start_profile"
    resp = requests.post(url, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"start_profile failed: HTTP {resp.status_code}: {resp.text[:500]}"
        )
    print(f"  /start_profile → HTTP {resp.status_code}")


def stop_profile(base_url: str, *, timeout: float = 60.0) -> None:
    url = f"{base_url.rstrip('/')}/stop_profile"
    resp = requests.post(url, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"stop_profile failed: HTTP {resp.status_code}: {resp.text[:500]}"
        )
    print(f"  /stop_profile → HTTP {resp.status_code}")


def summarize_round(
    name: str,
    results: list[RequestMetrics],
    *,
    wall_s: float,
) -> None:
    ok = [r for r in results if r.success]
    print(f"\n=== {name} ({len(ok)}/{len(results)} ok, wall={wall_s * 1000:.2f} ms) ===")
    for r in results:
        if r.success:
            print(
                f"  {r.request_id}: "
                f"ttft={r.ttft_s * 1000:.2f} ms, "
                f"latency={r.latency_s * 1000:.2f} ms, "
                f"tpot={r.tpot_s * 1000:.2f} ms, "
                f"prompt_tokens={r.prompt_len}, "
                f"output_tokens={r.output_tokens}"
            )
        else:
            print(f"  {r.request_id}: FAILED — {r.error}")

    if not ok:
        return

    def _stats(vals: list[float]) -> str:
        if len(vals) == 1:
            return f"mean={vals[0]:.2f}"
        return (
            f"mean={statistics.mean(vals):.2f}, "
            f"median={statistics.median(vals):.2f}, "
            f"min={min(vals):.2f}, "
            f"max={max(vals):.2f}"
        )

    ttfts = [r.ttft_s * 1000 for r in ok]
    lats = [r.latency_s * 1000 for r in ok]
    tpots = [r.tpot_s * 1000 for r in ok]
    print(f"  TTFT  (ms): {_stats(ttfts)}")
    print(f"  Latency (ms): {_stats(lats)}")
    print(f"  TPOT  (ms): {_stats(tpots)}")


def run_round(
    name: str,
    *,
    prepared: list[PreparedRequest],
    base_url: str,
    model: str,
    output_len: int,
    temperature: float,
    ignore_eos: bool,
    timeout: float,
) -> list[RequestMetrics]:
    n = len(prepared)
    print(f"\n--- Dispatching {n} chat.completions concurrently ({name}) ---")
    for req in prepared:
        print(f"  → queued {req.request_id} (prompt_tokens={req.prompt_len})")

    results_by_id: dict[str, RequestMetrics] = {}
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {
            pool.submit(
                send_chat_completion_streaming,
                base_url=base_url,
                model=model,
                prepared=req,
                output_len=output_len,
                temperature=temperature,
                ignore_eos=ignore_eos,
                timeout=timeout,
            ): req.request_id
            for req in prepared
        }
        for fut in as_completed(futures):
            req_id = futures[fut]
            m = fut.result()
            results_by_id[req_id] = m
            if m.success:
                print(
                    f"  ← {m.request_id} done: "
                    f"ttft={m.ttft_s * 1000:.2f} ms, "
                    f"latency={m.latency_s * 1000:.2f} ms"
                )
            else:
                print(f"  ← {m.request_id} FAILED: {m.error}")
    wall_s = time.perf_counter() - wall_start

    # Preserve preparation order in the summary.
    results = [results_by_id[req.request_id] for req in prepared]
    summarize_round(name, results, wall_s=wall_s)
    return results


def main() -> int:
    args = parse_args()
    tokenizer_id = args.tokenizer or args.model

    print(f"Loading tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    print(
        f"Preparing {args.concurrency} requests with unique "
        f"prefix_size={args.prefix_size} tokens ..."
    )
    prepared = build_prepared_requests(
        tokenizer,
        concurrency=args.concurrency,
        prefix_size=args.prefix_size,
    )

    # Round 1 — populate / measure before reset
    run_round(
        "round-1 (before reset)",
        prepared=prepared,
        base_url=args.base_url,
        model=args.model,
        output_len=args.output_len,
        temperature=args.temperature,
        ignore_eos=args.ignore_eos,
        timeout=args.timeout,
    )

    print(f"\nWaiting {args.wait_seconds} s before reset_prefix_cache ...")
    time.sleep(args.wait_seconds)

    print("Calling /start_profile ...")
    try:
        start_profile(args.base_url, timeout=min(60.0, args.timeout))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Is the server started with a profiler enabled "
            "(e.g. VLLM_TORCH_PROFILER_DIR set)?",
            file=sys.stderr,
        )
        return 1

    print("Calling /reset_prefix_cache ...")
    try:
        ok = reset_prefix_cache(
            args.base_url,
            reset_running_requests=args.reset_running_requests,
            timeout=min(60.0, args.timeout),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Is the server started with VLLM_SERVER_DEV_MODE=1?",
            file=sys.stderr,
        )
        return 1

    if not ok:
        print(
            "WARNING: reset_prefix_cache returned success=false "
            "(blocks may still be held). Continuing anyway."
        )

    # Round 2 — same prompts after cache reset
    run_round(
        "round-2 (after reset)",
        prepared=prepared,
        base_url=args.base_url,
        model=args.model,
        output_len=args.output_len,
        temperature=args.temperature,
        ignore_eos=args.ignore_eos,
        timeout=args.timeout,
    )

    print("\nCalling /stop_profile ...")
    try:
        stop_profile(args.base_url, timeout=min(60.0, args.timeout))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
