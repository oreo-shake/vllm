#!/usr/bin/env python3
"""Recompute vs reload TTFT benchmark across FS storage tiers."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import requests
from transformers import AutoTokenizer

MODEL = "mistralai/Ministral-3-8B-Reasoning-2512"
ENDPOINT = "http://127.0.0.1:8000/v1/completions"
HEALTH = "http://127.0.0.1:8000/health"
# INPUT_LENS = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]
INPUT_LENS = [256000, 257000]
OUTPUT_LEN = 8
CPU_BYTES = 107374182400
BLOCK_SIZE = 256
WRITE_DRAIN_MIN_SEC = 60
WRITE_DRAIN_MAX_SEC = 900
SERVER_READY_TIMEOUT = 600
DEFAULT_REQUESTS_PER_LENGTH = 1
DEFAULT_FS_CAPACITY_GIB = 8.0

VENV_BIN = Path("/home/ubuntu/build/vllm-cpu-evict/bin")
REPO = Path("/home/ubuntu/dev/vllm-cpu-eviction")
DEFAULT_RESULTS_DIR = REPO / "bench_results" / "tier_ttft"


def generate_exact_length_token_ids(
    tokenizer, target_length: int, variant: int = 0
) -> list[int]:
    """Deterministic prompt of exact length; unique prefix per variant."""
    text_blocks = [
        "The quick brown fox jumps over the lazy dog. ",
        "Artificial intelligence is rapidly transforming global infrastructure. ",
        "Quantum computing leverages the principles of quantum mechanics. ",
        "Photosynthesis is a vital process used by plants and other organisms. ",
        "The Great Wall of China is a historical series of fortifications. ",
        "In computer science, an algorithm is a finite sequence of instructions. ",
        "The history of aviation spans over more than two thousand years. ",
        "Music is the expressive art of arranging sounds in time. ",
        "A black hole is a region of spacetime where gravity is absolute. ",
        "Deep within the ocean, extraordinary creatures adapt to extreme pressure. ",
        "Cybersecurity measures are essential for protecting modern digital assets. ",
        "The Renaissance was a fervent period of European cultural rebirth. ",
    ]
    rng = random.Random(f"{target_length}:{variant}")
    tokens: list[int] = []
    if tokenizer.bos_token_id is not None:
        tokens.append(tokenizer.bos_token_id)

    deterministic_uuid = uuid.uuid5(
        uuid.NAMESPACE_OID, f"tier_ttft_bench_{target_length}_{variant}"
    )
    unique_prefix_str = f"CACHE_BUSTER_{deterministic_uuid.hex} | "
    tokens.extend(tokenizer.encode(unique_prefix_str, add_special_tokens=False))

    while len(tokens) < target_length:
        rng.shuffle(text_blocks)
        tokens.extend(tokenizer.encode("".join(text_blocks), add_special_tokens=False))
    return tokens[:target_length]


def measure_ttft(prompt_ids: list[int], max_tokens: int) -> float:
    payload = {
        "model": MODEL,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }
    ttft = None
    start = time.perf_counter()
    with requests.post(ENDPOINT, json=payload, stream=True, timeout=3600) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            text_delta = choices[0].get("text") or ""
            if text_delta and ttft is None:
                ttft = time.perf_counter() - start
    if ttft is None:
        raise RuntimeError("No token deltas received; TTFT unavailable")
    return ttft


def clean_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        path.unlink()


def clean_shm() -> None:
    shm = Path("/dev/shm")
    for child in shm.iterdir():
        if child.name.startswith("vllm_offload_"):
            try:
                child.unlink()
            except OSError as exc:
                print(f"[warn] could not remove {child}: {exc}")


def clean_storage(root_dir: str) -> None:
    root = Path(root_dir)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return
    for child in root.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as exc:
            print(f"[warn] could not remove {child}: {exc}")


def kv_transfer_config(
    root_dir: str, fs_capacity_bytes: int, fs_eviction_policy: str
) -> str:
    cfg = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": CPU_BYTES,
            "block_size": BLOCK_SIZE,
            "eviction_policy": "lru",
            "secondary_tiers": [
                {
                    "type": "fs",
                    "root_dir": root_dir,
                    "n_read_threads": 32,
                    "n_write_threads": 16,
                    "capacity_bytes": fs_capacity_bytes,
                    "eviction_policy": fs_eviction_policy,
                    "session_dead_ttl_s": 200,
                    "max_tiers": 8
                }
            ],
        },
    }
    return json.dumps(cfg)


def start_server(
    root_dir: str,
    log_path: Path,
    fs_capacity_bytes: int,
    fs_eviction_policy: str,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["VLLM_KV_CONNECTOR_HIT_LOG"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PATH"] = f"{VENV_BIN}:{env.get('PATH', '')}"

    cmd = [
        str(VENV_BIN / "vllm"),
        "serve",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        "262144",
        "--kv-transfer-config",
        kv_transfer_config(root_dir, fs_capacity_bytes, fs_eviction_policy),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Line-buffered so hit/metrics lines are visible during the run.
    log_f = open(log_path, "w", buffering=1)
    print(f"[server] starting, log={log_path}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._log_f = log_f  # type: ignore[attr-defined]
    return proc


def wait_ready(proc: subprocess.Popen, timeout: int = SERVER_READY_TIMEOUT) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited early with code {proc.returncode}")
        try:
            r = requests.get(HEALTH, timeout=2)
            if r.status_code == 200:
                print("[server] ready")
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server not ready within {timeout}s")


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        print(f"[server] stopping pid={proc.pid}")
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print("[server] force kill")
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=30)
    log_f = getattr(proc, "_log_f", None)
    if log_f is not None:
        log_f.close()


def parse_external_hits(log_path: Path) -> list[int]:
    """Parse external= token counts from Prefix cache hit log lines."""
    hits: list[int] = []
    if not log_path.exists():
        return hits
    for line in log_path.read_text(errors="replace").splitlines():
        if "Prefix cache hits:" not in line:
            continue
        try:
            marker = "external="
            i = line.index(marker) + len(marker)
            j = i
            while j < len(line) and line[j].isdigit():
                j += 1
            hits.append(int(line[i:j]))
        except (ValueError, IndexError):
            continue
    return hits


def live_cascade_jobs() -> float | None:
    """Read active FS cascade jobs from the live /metrics endpoint."""
    try:
        r = requests.get("http://127.0.0.1:8000/metrics", timeout=5)
        r.raise_for_status()
    except requests.RequestException as exc:
        print(f"[drain] metrics fetch failed: {exc}")
        return None
    # Gauge lines look like:
    # vllm:kv_offload_tiering_active_cascade_jobs{...,tier="1:fs"} 0.0
    for line in r.text.splitlines():
        if line.startswith("#"):
            continue
        if "kv_offload_tiering_active_cascade_jobs{" not in line:
            continue
        if 'tier="1:fs"' not in line and "tier='1:fs'" not in line:
            continue
        try:
            return float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
    return None


def wait_write_drain(log_path: Path) -> None:
    """Wait at least 60s, then until live FS cascade jobs drain (or timeout)."""
    del log_path  # kept for call-site compatibility
    print(f"[drain] waiting minimum {WRITE_DRAIN_MIN_SEC}s...", flush=True)
    time.sleep(WRITE_DRAIN_MIN_SEC)
    deadline = time.time() + (WRITE_DRAIN_MAX_SEC - WRITE_DRAIN_MIN_SEC)
    stable_zero = 0
    while time.time() < deadline:
        jobs = live_cascade_jobs()
        print(f"[drain] active_cascade_jobs={jobs}", flush=True)
        if jobs == 0.0:
            stable_zero += 1
            if stable_zero >= 2:
                print("[drain] cascade idle; continuing", flush=True)
                return
        else:
            stable_zero = 0
        time.sleep(5)
    print(
        f"[drain] WARNING: timed out after {WRITE_DRAIN_MAX_SEC}s "
        f"with active_cascade_jobs={live_cascade_jobs()}",
        flush=True,
    )


def run_phase(
    tokenizer,
    phase: str,
    log_path: Path,
    requests_per_length: int,
) -> list[dict]:
    results = []
    for length in INPUT_LENS:
        for variant in range(requests_per_length):
            prompt_ids = generate_exact_length_token_ids(tokenizer, length, variant)
            assert len(prompt_ids) == length
            print(
                f"[{phase}] input_len={length} request={variant + 1}/"
                f"{requests_per_length} sending...",
                flush=True,
            )
            t0 = time.perf_counter()
            ttft = measure_ttft(prompt_ids, OUTPUT_LEN)
            wall = time.perf_counter() - t0
            row = {
                "phase": phase,
                "input_len": length,
                "request_index": variant + 1,
                "ttft_s": ttft,
                "wall_s": wall,
                "disk_hit_log_lines": 0,
                "external_hit_tokens": 0,
            }
            results.append(row)
            print(
                f"[{phase}] input_len={length} request={variant + 1}/"
                f"{requests_per_length} ttft={ttft * 1000:.1f}ms "
                f"wall={wall:.2f}s",
                flush=True,
            )
            time.sleep(1)
    return results


def attach_hit_stats(results: list[dict], log_path: Path, phase: str) -> None:
    """Attach hit-log stats after the phase so log buffers are flushed."""
    hits = parse_external_hits(log_path)
    phase_rows = [r for r in results if r["phase"] == phase]
    if phase == "recompute":
        if hits:
            print(f"[check] WARNING: recompute saw {len(hits)} hit log lines: {hits}")
        return
    if len(hits) != len(phase_rows):
        print(
            f"[check] WARNING: reload expected {len(phase_rows)} disk hit lines, "
            f"got {len(hits)}: {hits}"
        )
    for row, ext in zip(phase_rows, hits):
        row["disk_hit_log_lines"] = 1
        row["external_hit_tokens"] = ext
        expected = (row["input_len"] // BLOCK_SIZE) * BLOCK_SIZE
        # Connector may omit the final incomplete/partial block.
        if ext < expected - BLOCK_SIZE:
            print(
                f"[check] WARNING: input_len={row['input_len']} "
                f"external_hit_tokens={ext} << aligned={expected}"
            )


def plot_results(storage_name: str, results: list[dict], out_path: Path) -> None:
    recomp = {
        length: statistics.mean(
            r["ttft_s"]
            for r in results
            if r["phase"] == "recompute" and r["input_len"] == length
        )
        for length in INPUT_LENS
        if any(
            r["phase"] == "recompute" and r["input_len"] == length for r in results
        )
    }
    reload = {
        length: statistics.mean(
            r["ttft_s"]
            for r in results
            if r["phase"] == "reload" and r["input_len"] == length
        )
        for length in INPUT_LENS
        if any(r["phase"] == "reload" and r["input_len"] == length for r in results)
    }
    xs = [x for x in INPUT_LENS if x in recomp and x in reload]
    ys_r = [recomp[x] * 1000 for x in xs]
    ys_l = [reload[x] * 1000 for x in xs]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(xs, ys_r, marker="o", label="recompute")
    ax.plot(xs, ys_l, marker="s", label="reload")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], rotation=45, ha="right")
    ax.set_xlabel("Input length (tokens)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title(f"TTFT: recompute vs reload ({storage_name})")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    ymax = max(max(ys_r), max(ys_l)) if xs else 1.0
    for x, yr, yl in zip(xs, ys_r, ys_l):
        ax.annotate(
            f"recomp {recomp[x]:.3f}s\nreload {reload[x]:.3f}s",
            xy=(x, max(yr, yl)),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="0.7",
                alpha=0.9,
            ),
        )
    ax.set_ylim(0, ymax * 1.22)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def run_storage(
    storage_name: str,
    root_dir: str,
    tokenizer,
    results_dir: Path,
    requests_per_length: int,
    fs_capacity_bytes: int,
    fs_eviction_policy: str,
) -> list[dict]:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_recomp = results_dir / f"{storage_name}_recompute_server.log"
    log_reload = results_dir / f"{storage_name}_reload_server.log"
    all_results: list[dict] = []
    proc = None

    try:
        print(f"\n===== {storage_name}: clean state =====")
        stop_lingering_servers()
        clean_shm()
        clean_storage(root_dir)

        print(f"===== {storage_name}: phase 1 recompute =====", flush=True)
        proc = start_server(
            root_dir, log_recomp, fs_capacity_bytes, fs_eviction_policy
        )
        wait_ready(proc)
        all_results.extend(
            run_phase(tokenizer, "recompute", log_recomp, requests_per_length)
        )
        attach_hit_stats(all_results, log_recomp, "recompute")

        wait_write_drain(log_recomp)

        stop_server(proc)
        proc = None
        eviction_marker = (
            "Storage tier 'fs' eviction active:"
            if fs_eviction_policy == "prefix_pruning"
            else "Storage tier 'fs' LRU eviction active:"
        )
        if eviction_marker in log_recomp.read_text(errors="replace"):
            print(f"[check] confirmed FS eviction under {fs_eviction_policy}")
        else:
            print(
                "[check] WARNING: no FS eviction summary found; "
                "this run may not have exceeded its storage capacity."
            )

        # Persist disk; clear CPU/shm so reload is storage-only.
        print(f"===== {storage_name}: clear /dev/shm before reload =====", flush=True)
        clean_shm()

        print(f"===== {storage_name}: phase 2 reload =====", flush=True)
        proc = start_server(
            root_dir, log_reload, fs_capacity_bytes, fs_eviction_policy
        )
        wait_ready(proc)
        all_results.extend(
            run_phase(tokenizer, "reload", log_reload, requests_per_length)
        )

        stop_server(proc)
        proc = None
        attach_hit_stats(all_results, log_reload, "reload")
    finally:
        stop_server(proc)
        print(f"===== {storage_name}: final cleanup =====", flush=True)
        clean_shm()
        clean_storage(root_dir)

    # Correctness summary
    for phase in ("recompute", "reload"):
        rows = [r for r in all_results if r["phase"] == phase]
        total_ext = sum(r["external_hit_tokens"] for r in rows)
        print(
            f"[check] {storage_name}/{phase}: sum external_hit_tokens={total_ext}",
            flush=True,
        )

    json_path = results_dir / f"{storage_name}_results.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    plot_results(storage_name, all_results, results_dir / f"{storage_name}_ttft.png")
    return all_results


def stop_lingering_servers() -> None:
    # Best-effort: free port 8000 if a prior run is stuck.
    try:
        r = requests.get(HEALTH, timeout=1)
        if r.status_code == 200:
            print("[warn] server already up on :8000; attempting pkill")
            subprocess.run(["pkill", "-f", "vllm serve"], check=False)
            time.sleep(5)
    except requests.RequestException:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storages",
        nargs="+",
        default=["ebs", "nvme"],
        choices=["ebs", "nvme"],
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for server logs, JSON results, and plots",
    )
    parser.add_argument(
        "--requests-per-length",
        type=int,
        default=DEFAULT_REQUESTS_PER_LENGTH,
        help="Distinct prompts per input length, run sequentially in each phase",
    )
    parser.add_argument(
        "--fs-capacity-gib",
        type=float,
        default=DEFAULT_FS_CAPACITY_GIB,
        help="FS tier capacity in GiB; lower this to force storage eviction",
    )
    parser.add_argument(
        "--fs-eviction-policy",
        choices=["lru", "prefix_pruning"],
        default="prefix_pruning",
        help="Eviction policy for the FS tier",
    )
    args = parser.parse_args()
    if args.requests_per_length < 1:
        parser.error("--requests-per-length must be at least 1")
    if args.fs_capacity_gib <= 0:
        parser.error("--fs-capacity-gib must be positive")
    fs_capacity_bytes = int(args.fs_capacity_gib * (1024**3))
    results_dir = args.results_dir.resolve()

    roots = {
        "ebs": "/mnt/ebs/",
        "nvme": "/mnt/nvme0/",
    }

    print(f"Loading tokenizer for {MODEL}...")
    print(f"Results dir: {results_dir}", flush=True)
    print(
        f"Benchmark workload: {args.requests_per_length} distinct prompt(s) "
        f"per input length, sequential; FS capacity={args.fs_capacity_gib:g} GiB; "
        f"FS eviction={args.fs_eviction_policy}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    for name in args.storages:
        run_storage(
            name,
            roots[name],
            tokenizer,
            results_dir,
            args.requests_per_length,
            fs_capacity_bytes,
            args.fs_eviction_policy,
        )

    print("\nDone. Results in", results_dir)


if __name__ == "__main__":
    main()
