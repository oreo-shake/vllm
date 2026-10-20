#!/usr/bin/env python3
"""
Benchmark overlap of cache retrieval/loading vs computation in vLLM/LMCache.

Flow:
  1. Dispatch 1 request (req-0) to populate/store prefix cache.
  2. Wait 60s (or --wait-seconds) and optionally reset prefix cache.
  3. Call /start_profile on the vLLM server.
  4. Dispatch req-0 (repeat prompt) first, wait --stagger-seconds, then dispatch
     req-1 (new unique prompt) to ensure KV cache transfer starts before computation.
  5. Call /stop_profile (waits without timeout as trace dumping may take 15+ mins).

Example:
  python prefix_cache_overlap.py \
      --model poolside/Laguna-XS-2.1 \
      --concurrency 2 \
      --prefix-size 256000 \
      --output-len 64 \
      --stagger-seconds 0.2 \
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
        description="Benchmark overlap of prefix cache loading vs prefill computation."
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
        default=2,
        help=(
            "Total concurrent requests in Round 2 (1 repeated + N-1 unique). "
            "Minimum is 2 (default: %(default)s)"
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
        default=16,
        help="max_tokens / completion length (default: %(default)s)",
    )
    p.add_argument(
        "--wait-seconds",
        type=float,
        default=60.0,
        help="Seconds to wait before cache reset / Round 2 (default: %(default)s)",
    )
    p.add_argument(
        "--stagger-seconds",
        type=float,
        default=0.2,
        help=(
            "Delay in seconds between dispatching req-0 (repeat prompt) "
            "and subsequent unique requests in Round 2 (default: %(default)s)"
        ),
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
    p.add_argument(
        "--skip-reset",
        action="store_true",
        help="Skip calling /reset_prefix_cache between round 1 and round 2",
    )
    return p.parse_args()


def build_hello_prefix_text(
    tokenizer: AutoTokenizer,
    prefix_size: int,
    *,
    request_index: int,
) -> str:
    """Build a prefix string of approximately `prefix_size` tokens with a unique head."""
    hello_ids = tokenizer.encode("hello", add_special_tokens=False)
    if not hello_ids:
        raise RuntimeError('tokenizer.encode("hello") returned no tokens')

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
    """Build concurrency prompts, each with a distinct unique prefix."""
    if concurrency < 2:
        raise ValueError("--concurrency must be >= 2 to test overlap")
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
        label = f"req-{i} (repeat)" if i == 0 else f"req-{i} (unique)"
        req = PreparedRequest(
            request_id=f"req-{i}",
            prompt=prompt,
            prompt_len=len(prompt_ids),
            prefix_len=prefix_size,
        )
        prepared.append(req)
        print(
            f"  prepared {label}: "
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
    except Exception as exc:  # noqa: BLE001
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


def start_profile(base_url: str, timeout: float = 60.0) -> None:
    """POST /start_profile to enable server profiling during Round 2."""
    url = f"{base_url.rstrip('/')}/start_profile"
    print(f"Calling {url} ...")
    try:
        resp = requests.post(url, timeout=timeout)
        if resp.status_code == 200:
            print(f"  /start_profile → {resp.text[:200].strip()}")
        else:
            print(
                f"WARNING: start_profile returned HTTP {resp.status_code}: "
                f"{resp.text[:500]}"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR starting profile: {exc}", file=sys.stderr)


def stop_profile(base_url: str) -> None:
    """POST /stop_profile to dump profiler trace (no timeout set)."""
    url = f"{base_url.rstrip('/')}/stop_profile"
    print(
        f"Calling {url} (waiting without timeout as trace dumping can take 15+ mins) ..."
    )
    try:
        resp = requests.post(url, timeout=None)
        if resp.status_code == 200:
            print(f"  /stop_profile → {resp.text[:200].strip()}")
        else:
            print(
                f"WARNING: stop_profile returned HTTP {resp.status_code}: "
                f"{resp.text[:500]}"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR stopping profile: {exc}", file=sys.stderr)


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
    stagger_seconds: float = 0.0,
) -> list[RequestMetrics]:
    n = len(prepared)
    print(f"\n--- Dispatching {n} chat.completions ({name}) ---")

    results_by_id: dict[str, RequestMetrics] = {}
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {}
        for idx, req in enumerate(prepared):
            if idx > 0 and stagger_seconds > 0:
                print(f"  (waiting {stagger_seconds:.3f}s before sending next request...)")
                time.sleep(stagger_seconds)

            print(f"  → dispatched {req.request_id} (prompt_tokens={req.prompt_len})")
            fut = pool.submit(
                send_chat_completion_streaming,
                base_url=base_url,
                model=model,
                prepared=req,
                output_len=output_len,
                temperature=temperature,
                ignore_eos=ignore_eos,
                timeout=timeout,
            )
            futures[fut] = req.request_id

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

    results = [results_by_id[req.request_id] for req in prepared]
    summarize_round(name, results, wall_s=wall_s)
    return results


def main() -> int:
    args = parse_args()
    tokenizer_id = args.tokenizer or args.model

    effective_concurrency = max(2, args.concurrency)

    print(f"Loading tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    print(
        f"Preparing prompts with prefix_size={args.prefix_size} tokens..."
    )
    prepared = build_prepared_requests(
        tokenizer,
        concurrency=effective_concurrency,
        prefix_size=args.prefix_size,
    )

    req_repeat = prepared[0]          # req-0: sent in Round 1 & first in Round 2
    reqs_round2 = prepared            # req-0 (repeat) followed by req-1... (unique)

    # 1. Dispatch SINGLE request (req-0) to store KV cache
    print("\n=== STEP 1: Sending initial request (req-0) to populate cache ===")
    run_round(
        "round-1 (single request)",
        prepared=[req_repeat],
        base_url=args.base_url,
        model=args.model,
        output_len=args.output_len,
        temperature=args.temperature,
        ignore_eos=args.ignore_eos,
        timeout=args.timeout,
    )

    # 2. Wait 60s
    print(f"\n=== STEP 2: Waiting {args.wait_seconds} seconds ===")
    time.sleep(args.wait_seconds)

    # 3. Optional Prefix Cache Reset
    if not args.skip_reset:
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
                "WARNING: reset_prefix_cache returned success=false. Continuing anyway."
            )
    else:
        print("Skipping /reset_prefix_cache endpoint call.")

    st_pr = 0
    # 4. Start Profile before starting Round 2
    if(st_pr):
        print("\n=== STEP 3: Starting Profiler ===")
        start_profile(args.base_url)

    # 5. Dispatch req-0 first, wait stagger_seconds, then send req-1...
    print("\n=== STEP 4: Replaying req-0 first, then new request(s) ===")
    run_round(
        "round-2 (staggered cache transfer vs computation overlap test)",
        prepared=reqs_round2,
        base_url=args.base_url,
        model=args.model,
        output_len=args.output_len,
        temperature=args.temperature,
        ignore_eos=args.ignore_eos,
        timeout=args.timeout,
        stagger_seconds=args.stagger_seconds,
    )

    if(st_pr):
        # 6. Stop Profile after Round 2 completes
        print("\n=== STEP 5: Stopping Profiler & Dumping Traces ===")
        stop_profile(args.base_url)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())