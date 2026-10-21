#!/usr/bin/env python3
"""
Compare TTFT for one session: a long first turn, then a short continuation.

Each turn is its own session: ``--session-id`` plus ``-0`` and ``-1``.
The continuation is the first prompt plus a few extra tokens, so the
follow-up recomputes only that tail. Both turns are the first request of
their session, which is what a CPU pin policy holds.

Flow:
  1. Build a prompt of --size tokens and a follow-up that adds --extra-tokens.
  2. Round 1: send the long turn, then the short continuation.
  3. Wait --wait-seconds (default 60s).
  4. Call /reset_prefix_cache (unless --skip-reset).
  5. Round 2: resend both turns; record TTFT.
  6. Print a side-by-side TTFT comparison.

Example:
  python compare_ttft_session.py \
      --model mistralai/Ministral-3-8B-Instruct-2512 \
      --size 256000 \
      --extra-tokens 256 \
      --session-id pin-session \
      --output-len 16 \
      --wait-seconds 60 \
      --base-url http://127.0.0.1:8004
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import requests
from transformers import AutoTokenizer


@dataclass
class PreparedRequest:
    request_id: str
    prompt: str
    prompt_len: int
    target_len: int
    session_id: str


@dataclass
class RequestMetrics:
    request_id: str
    success: bool
    prompt_len: int = 0
    target_len: int = 0
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
            "Send one session (long turn, then a short continuation), "
            "reset the GPU prefix cache after a wait, resend both turns, "
            "and compare TTFT."
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
        "--size",
        type=int,
        default=256000,
        help="Token length of the first turn (default: %(default)s)",
    )
    p.add_argument(
        "--extra-tokens",
        type=int,
        default=256,
        help=(
            "Tokens appended for the continuation. Only these are new "
            "work on top of the first turn (default: %(default)s)"
        ),
    )
    p.add_argument(
        "--session-id",
        default="pin-session",
        help=(
            "Session id prefix. Turn 0 sends '<id>-0' and turn 1 sends "
            "'<id>-1' (default: %(default)s)"
        ),
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


def _exact_text(tokenizer: AutoTokenizer, token_ids: list[int]) -> str:
    """Decode ids, then keep a string that re-encodes to those ids."""
    text = tokenizer.decode(token_ids)
    again = tokenizer.encode(text, add_special_tokens=False)
    if again == token_ids:
        return text
    # A trailing partial token can decode to text that grows on re-encode.
    # Drop from the end until the round trip is stable, then pad back with
    # a single-token filler that does not merge into the previous token.
    hello_ids = tokenizer.encode(" hello", add_special_tokens=False)
    if len(hello_ids) != 1:
        hello_ids = tokenizer.encode("hello", add_special_tokens=False)[:1]
    filler = hello_ids[0]
    trimmed = list(token_ids)
    while trimmed and tokenizer.encode(
        tokenizer.decode(trimmed), add_special_tokens=False
    ) != trimmed:
        trimmed.pop()
    while len(trimmed) < len(token_ids):
        trimmed.append(filler)
        text = tokenizer.decode(trimmed)
        if tokenizer.encode(text, add_special_tokens=False) != trimmed:
            trimmed.pop()
            break
    text = tokenizer.decode(trimmed)
    if tokenizer.encode(text, add_special_tokens=False) != trimmed:
        raise RuntimeError("could not build a prompt that round-trips token ids")
    if len(trimmed) != len(token_ids):
        token_ids[:] = trimmed
    return text


def build_session_prompts(
    tokenizer: AutoTokenizer,
    size: int,
    extra_tokens: int,
    session_id: str,
) -> list[PreparedRequest]:
    """First turn of ``size`` tokens, then the same tokens plus a short tail."""
    if size < 1:
        raise ValueError(f"--size must be >= 1, got {size}")
    if extra_tokens < 1:
        raise ValueError(f"--extra-tokens must be >= 1, got {extra_tokens}")

    hello_ids = tokenizer.encode("hello", add_special_tokens=False)
    if not hello_ids:
        raise RuntimeError('tokenizer.encode("hello") returned no tokens')
    head_ids = tokenizer.encode("id=session;size=%d;" % size, add_special_tokens=False)
    if len(head_ids) >= size:
        raise ValueError(f"--size {size} is shorter than the unique head")

    fill_need = size - len(head_ids)
    reps = (fill_need + len(hello_ids) - 1) // len(hello_ids)
    base_ids = head_ids + (hello_ids * reps)[:fill_need]
    base_text = _exact_text(tokenizer, base_ids)
    tail_reps = (extra_tokens + len(hello_ids) - 1) // len(hello_ids)
    follow_ids = list(base_ids) + (hello_ids * tail_reps)[:extra_tokens]
    follow_text = _exact_text(tokenizer, follow_ids)
    base_check = tokenizer.encode(base_text, add_special_tokens=False)
    follow_check = tokenizer.encode(follow_text, add_special_tokens=False)
    shared = 0
    for left, right in zip(base_check, follow_check):
        if left != right:
            break
        shared += 1
    session_ids = (f"{session_id}-0", f"{session_id}-1")
    print(
        f"  session prompts: first={len(base_check)} tokens "
        f"session={session_ids[0]!r}, "
        f"continuation={len(follow_check)} tokens "
        f"session={session_ids[1]!r}, "
        f"shared_prefix={shared} tokens, "
        f"new_tail={len(follow_check) - shared} tokens"
    )
    return [
        PreparedRequest(
            request_id=f"turn-0-{len(base_check)}",
            prompt=base_text,
            prompt_len=len(base_check),
            target_len=size,
            session_id=session_ids[0],
        ),
        PreparedRequest(
            request_id=f"turn-1-{len(follow_check)}",
            prompt=follow_text,
            prompt_len=len(follow_check),
            target_len=size + extra_tokens,
            session_id=session_ids[1],
        ),
    ]


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
        "session_id": prepared.session_id,
        "priority": 0,
    }
    if ignore_eos:
        payload["ignore_eos"] = True

    metrics = RequestMetrics(
        request_id=prepared.request_id,
        success=False,
        prompt_len=prepared.prompt_len,
        target_len=prepared.target_len,
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
    print(f"\n--- Dispatching {len(prepared)} chat.completions ({name}) ---")
    results: list[RequestMetrics] = []
    wall_start = time.perf_counter()

    for req in prepared:
        print(
            f"  → sending {req.request_id} "
            f"(prompt_tokens={req.prompt_len}, session_id={req.session_id})"
        )
        m = send_chat_completion_streaming(
            base_url=base_url,
            model=model,
            prepared=req,
            output_len=output_len,
            temperature=temperature,
            ignore_eos=ignore_eos,
            timeout=timeout,
        )
        results.append(m)
        if m.success:
            print(
                f"  ← {m.request_id} done: "
                f"ttft={m.ttft_s * 1000:.2f} ms, "
                f"latency={m.latency_s * 1000:.2f} ms, "
                f"prompt_tokens={m.prompt_len}, "
                f"output_tokens={m.output_tokens}"
            )
        else:
            print(f"  ← {m.request_id} FAILED: {m.error}")

    wall_s = time.perf_counter() - wall_start
    ok = sum(1 for r in results if r.success)
    print(f"\n=== {name} ({ok}/{len(results)} ok, wall={wall_s * 1000:.2f} ms) ===")
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
    return results


def print_comparison(
    round1: list[RequestMetrics],
    round2: list[RequestMetrics],
) -> None:
    print("\n=== TTFT comparison (round-1 vs round-2) ===")
    print(
        f"{'request':<28} {'size':>8} {'r1_ttft_ms':>12} {'r2_ttft_ms':>12} "
        f"{'delta_ms':>12} {'delta_%':>10}"
    )
    by_id_r2 = {r.request_id: r for r in round2}
    for r1 in round1:
        r2 = by_id_r2.get(r1.request_id)
        if not r1.success or r2 is None or not r2.success:
            r1_s = f"{r1.ttft_s * 1000:.2f}" if r1.success else "FAIL"
            r2_s = (
                f"{r2.ttft_s * 1000:.2f}"
                if r2 is not None and r2.success
                else "FAIL"
            )
            print(
                f"{r1.request_id:<28} {r1.target_len:>8} {r1_s:>12} {r2_s:>12} "
                f"{'n/a':>12} {'n/a':>10}"
            )
            continue

        d_ms = (r2.ttft_s - r1.ttft_s) * 1000
        d_pct = (r2.ttft_s / r1.ttft_s - 1.0) * 100.0 if r1.ttft_s > 0 else 0.0
        print(
            f"{r1.request_id:<28} {r1.target_len:>8} "
            f"{r1.ttft_s * 1000:>12.2f} {r2.ttft_s * 1000:>12.2f} "
            f"{d_ms:>12.2f} {d_pct:>9.1f}%"
        )


def main() -> int:
    args = parse_args()

    tokenizer_id = args.tokenizer or args.model
    print(f"Loading tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    print(
        f"Preparing sessions {args.session_id}-0 and {args.session_id}-1: "
        f"size={args.size}, extra_tokens={args.extra_tokens}"
    )
    prepared = build_session_prompts(
        tokenizer, args.size, args.extra_tokens, args.session_id
    )

    # Round 1: cold / first-pass TTFT
    print("\n=== STEP 1: Round 1 (initial requests) ===")
    round1 = run_round(
        "round-1",
        prepared=prepared,
        base_url=args.base_url,
        model=args.model,
        output_len=args.output_len,
        temperature=args.temperature,
        ignore_eos=args.ignore_eos,
        timeout=args.timeout,
    )

    # Wait for async store / drain
    print(f"\n=== STEP 2: Waiting {args.wait_seconds} seconds ===")
    time.sleep(args.wait_seconds)

    # Reset GPU prefix cache so Round 2 reloads from offload (if enabled)
    if not args.skip_reset:
        print("\n=== STEP 3: Calling /reset_prefix_cache ===")
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
        print("\n=== STEP 3: Skipping /reset_prefix_cache ===")

    # Round 2: same prompts again
    print("\n=== STEP 4: Round 2 (same requests after wait/reset) ===")
    round2 = run_round(
        "round-2",
        prepared=prepared,
        base_url=args.base_url,
        model=args.model,
        output_len=args.output_len,
        temperature=args.temperature,
        ignore_eos=args.ignore_eos,
        timeout=args.timeout,
    )

    print_comparison(round1, round2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
