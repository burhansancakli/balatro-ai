"""
Balatrobot Comprehensive Benchmark
====================================
Benchmarks Balatrobot across multiple dimensions for research documentation:
  1. Reset strategy comparison  (start vs load)
  2. Episode speed baseline     (random agent, N episodes)
  3. RPC latency per method     (how long each API call takes)
  4. Training feasibility       (projected episodes/hour for 1-8 parallel instances)

Usage:
    python benchmark_full.py

Requirements:
    pip install requests

Make sure Balatrobot is running first:
    uvx balatrobot serve --fast --no-shaders --fps-cap 1000 --gamespeed 4
"""

from pathlib import Path

import requests
import time
import statistics
import random
import os
import json
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
URL              = "http://127.0.0.1:12346"
TIMEOUT          = 30
SAVE_PATH        = str(Path.cwd() / "benchmark_game_saves/balatro_bench_save.jkr")
print(f"Save path: {SAVE_PATH}")
RESET_TRIALS     = 20   # trials for reset benchmark
EPISODE_TRIALS   = 20   # episodes for episode benchmark
RPC_LATENCY_N    = 30   # calls per method for latency benchmark
DECK             = "RED"
STAKE            = "WHITE"
SEED             = "BENCH01"

# ─────────────────────────────────────────────
# RPC HELPER
# ─────────────────────────────────────────────
def rpc(method: str, params: dict = {}) -> dict:
    response = requests.post(
        URL,
        json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
        timeout=TIMEOUT,
    )
    data = response.json()
    if "error" in data:
        print(data)
        raise Exception(f"[{method}] {data['error']['message']}")
    return data["result"]


# ─────────────────────────────────────────────
# FORMATTING HELPERS
# ─────────────────────────────────────────────
def hline(char="─", width=60):
    print(char * width)

def section(title: str):
    print()
    hline("═")
    print(f"  {title}")
    hline("═")

def stats_block(label: str, times: list, unit="s"):
    avg  = statistics.mean(times)
    mn   = min(times)
    mx   = max(times)
    std  = statistics.stdev(times) if len(times) > 1 else 0
    p95  = sorted(times)[int(len(times) * 0.95)]
    print(f"  {label}")
    print(f"    avg    : {avg:.4f}{unit}")
    print(f"    min    : {mn:.4f}{unit}")
    print(f"    max    : {mx:.4f}{unit}")
    print(f"    stdev  : {std:.4f}{unit}")
    print(f"    p95    : {p95:.4f}{unit}")
    return avg


# ─────────────────────────────────────────────
# SECTION 1: HEALTH CHECK
# ─────────────────────────────────────────────
def check_health():
    section("1 / Health Check")
    try:
        t = time.perf_counter()
        result = rpc("health")
        latency = time.perf_counter() - t
        print(f"  Status  : {result['status']}")
        print(f"  Latency : {latency*1000:.1f}ms")
        print(f"  Server  : {URL}")
        print(f"  Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
        print(f"  Make sure Balatrobot is running on {URL}")
        return False


# ─────────────────────────────────────────────
# SECTION 2: RESET STRATEGY BENCHMARK
# ─────────────────────────────────────────────
def bench_reset():
    section("2 / Reset Strategy Benchmark")
    print(f"  Comparing two reset methods over {RESET_TRIALS} trials each.\n")

    # --- Method A: start ---
    print(f"  [A] start() reset — {RESET_TRIALS} trials")
    start_times = []
    for i in range(RESET_TRIALS):
        t = time.perf_counter()
        state = rpc("gamestate")
        if state["state"] != "MENU":
            rpc("menu")
        rpc("start", {"deck": DECK, "stake": STAKE, "seed": SEED})
        start_times.append(time.perf_counter() - t)
        print(f"    trial {i+1:02d}: {start_times[-1]:.3f}s")

    print()
    avg_start = stats_block("start() summary", start_times)

    # --- Create save file ---
    print(f"\n  Creating save file at {SAVE_PATH} ...")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    state = rpc("gamestate")
    if state["state"] != "MENU":
        rpc("menu")
    rpc("start", {"deck": DECK, "stake": STAKE, "seed": SEED})
    rpc("save", {"path": SAVE_PATH})
    print(f"  Save file created.\n")

    # --- Method B: load ---
    print(f"  [B] load() reset — {RESET_TRIALS} trials")
    load_times = []
    for i in range(RESET_TRIALS):
        t = time.perf_counter()
        rpc("load", {"path": SAVE_PATH})
        load_times.append(time.perf_counter() - t)
        print(f"    trial {i+1:02d}: {load_times[-1]:.3f}s")

    print()
    avg_load = stats_block("load() summary", load_times)

    speedup = avg_start / avg_load
    print(f"\n  Speedup (load vs start): {speedup:.2f}x")
    print(f"  Recommended strategy   : load()")

    return avg_load, avg_start


# ─────────────────────────────────────────────
# SECTION 3: RPC METHOD LATENCY
# ─────────────────────────────────────────────
def bench_rpc_latency():
    section("3 / RPC Method Latency")
    print(f"  Measuring per-method latency over {RPC_LATENCY_N} calls each.\n")

    results = {}

    # health — always safe to call
    print(f"  Measuring: health")
    times = []
    for _ in range(RPC_LATENCY_N):
        t = time.perf_counter()
        rpc("health")
        times.append((time.perf_counter() - t) * 1000)
    avg = statistics.mean(times)
    results["health"] = avg
    print(f"    avg: {avg:.2f}ms")

    # gamestate
    print(f"  Measuring: gamestate")
    times = []
    for _ in range(RPC_LATENCY_N):
        t = time.perf_counter()
        rpc("gamestate")
        times.append((time.perf_counter() - t) * 1000)
    avg = statistics.mean(times)
    results["gamestate"] = avg
    print(f"    avg: {avg:.2f}ms")

    # play — need to be in SELECTING_HAND state
    print(f"  Measuring: play  (requires active game)")
    rpc("load", {"path": SAVE_PATH})
    rpc("select")  # enter SELECTING_HAND
    play_times = []
    for i in range(min(10, RPC_LATENCY_N)):  # limited since it consumes hands
        state = rpc("gamestate")
        if state["state"] != "SELECTING_HAND":
            break
        hand_size = state["hand"]["count"]
        cards = list(range(min(5, hand_size)))
        t = time.perf_counter()
        rpc("play", {"cards": cards})
        play_times.append((time.perf_counter() - t) * 1000)
    if play_times:
        avg = statistics.mean(play_times)
        results["play"] = avg
        print(f"    avg: {avg:.2f}ms  (over {len(play_times)} calls)")

    print()
    print(f"  {'Method':<15} {'Avg latency':>12}")
    hline("-", 30)
    for method, avg_ms in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {method:<15} {avg_ms:>10.2f}ms")

    overall_avg = statistics.mean(results.values())
    print()

    return results


# ─────────────────────────────────────────────
# SECTION 4: EPISODE SPEED BENCHMARK
# ─────────────────────────────────────────────
def run_single_episode() -> dict:
    """Run one full episode with a random agent. Returns stats."""
    t_start = time.perf_counter()
    steps = 0
    rpc_calls = 0

    rpc("load", {"path": SAVE_PATH}); rpc_calls += 1

    state = rpc("gamestate"); rpc_calls += 1

    while state["state"] != "GAME_OVER":
        steps += 1
        match state["state"]:
            case "BLIND_SELECT":
                state = rpc("select"); rpc_calls += 1
            case "SELECTING_HAND":
                hand_size = state["hand"]["count"]
                cards = list(range(min(5, hand_size)))
                state = rpc("play", {"cards": cards}); rpc_calls += 1
            case "ROUND_EVAL":
                state = rpc("cash_out"); rpc_calls += 1
            case "SHOP":
                state = rpc("next_round"); rpc_calls += 1
            case "SMODS_BOOSTER_OPENED":
                state = rpc("pack", {"skip": True}); rpc_calls += 1
            case _:
                state = rpc("gamestate"); rpc_calls += 1
        if steps >= 500:
            break

    duration = time.perf_counter() - t_start
    return {
        "duration_s": duration,
        "steps": steps,
        "rpc_calls": rpc_calls,
        "ante": state.get("ante_num", 0),
        "round": state.get("round_num", 0),
        "won": state.get("won", False),
        "ms_per_step": (duration / steps * 1000) if steps > 0 else 0,
    }


def bench_episodes():
    section("4 / Episode Speed Benchmark  (random agent)")
    print(f"  Running {EPISODE_TRIALS} full episodes with a random agent.\n")

    results = []
    total_t = time.perf_counter()

    for i in range(1, EPISODE_TRIALS + 1):
        print(f"  Episode {i:02d}/{EPISODE_TRIALS} ...", end=" ", flush=True)
        try:
            r = run_single_episode()
            results.append(r)
            print(f"{r['duration_s']:.2f}s  |  steps={r['steps']}  |  rpc={r['rpc_calls']}  |  ante={r['ante']}  |  won={r['won']}")
        except Exception as e:
            print(f"FAILED: {e}")

    total_elapsed = time.perf_counter() - total_t

    if not results:
        print("  No episodes completed.")
        return None

    durations   = [r["duration_s"]   for r in results]
    steps_list  = [r["steps"]        for r in results]
    rpc_list    = [r["rpc_calls"]    for r in results]
    antes       = [r["ante"]         for r in results]
    ms_per_step = [r["ms_per_step"]  for r in results]
    wins        = sum(1 for r in results if r["won"])

    print()
    avg_dur   = stats_block("Episode duration", durations)
    print()
    avg_steps = stats_block("Steps per episode", steps_list, unit="")
    print()
    avg_rpc   = stats_block("RPC calls per episode", rpc_list, unit="")
    print()
    avg_mps   = stats_block("Time per step", ms_per_step, unit="ms")

    eps_per_min = 60 * len(results) / total_elapsed
    eps_per_hr  = eps_per_min * 60

    print(f"\n  Win rate           : {wins}/{len(results)} ({100*wins/len(results):.0f}%)")
    print(f"  Avg ante reached   : {statistics.mean(antes):.2f}")
    print(f"  Episodes / minute  : {eps_per_min:.1f}")
    print(f"  Episodes / hour    : {eps_per_hr:.0f}")
    print(f"  Total bench time   : {total_elapsed:.1f}s")

    return {
        "avg_dur": avg_dur,
        "avg_steps": avg_steps,
        "eps_per_min": eps_per_min,
        "eps_per_hr": eps_per_hr,
        "avg_mps": avg_mps,
    }


# ─────────────────────────────────────────────
# SECTION 5: TRAINING FEASIBILITY PROJECTION
# ─────────────────────────────────────────────
def training_projection(eps_per_min_single: float, avg_dur: float, avg_steps: float):
    section("5 / Training Results")

    print(f"  Based on measured {eps_per_min_single:.1f} eps/min on a single instance.\n")

    instance_counts = [1, 2, 4, 6, 8]
    target_episodes = [1_000, 10_000, 50_000, 100_000]

    # Per-instance projection table
    print(f"  {'Instances':<12} {'eps/min':>10} {'eps/hr':>10} {'10k eps':>12} {'100k eps':>12}  Assessment")
    hline("-", 72)
    for n in instance_counts:
        epm  = eps_per_min_single * n
        eph  = epm * 60
        t10k = 10_000 / epm
        t100k= 100_000 / epm
        t10k_str  = f"{t10k:.0f}min" if t10k < 60 else f"{t10k/60:.1f}hr"
        t100k_str = f"{t100k:.0f}min" if t100k < 60 else f"{t100k/60:.1f}hr"
        assessment = (
            "EPM: "+f"{epm:.1f}"
        )
        print(f"  {n:<12} {epm:>10.1f} {eph:>10.0f} {t10k_str:>12} {t100k_str:>12}  {assessment}")

    # Target episodes table
    print(f"\n  Time to reach training targets (4 parallel instances):")
    epm_4 = eps_per_min_single * 4
    hline("-", 45)
    print(f"  {'Target':>10}  {'Time':>12}  {'Overnight?':>12}")
    hline("-", 45)
    for target in target_episodes:
        mins = target / epm_4
        hrs  = mins / 60
        time_str = f"{mins:.0f}min" if mins < 60 else f"{hrs:.1f}hr"
        overnight = "Yes" if hrs <= 8 else ("Maybe" if hrs <= 16 else "No")
        print(f"  {target:>10,}  {time_str:>12}  {overnight:>12}")

    # Research context
    # Step-based projection (for when env is more complex)
    steps_per_hr = (avg_steps / avg_dur) * 3600
    print(f"  Steps / second (1 instance) : {avg_steps/avg_dur:.1f}")
    print(f"  Steps / hour   (1 instance) : {steps_per_hr:,.0f}")
    print(f"  Steps / hour   (4 instances): {steps_per_hr*4:,.0f}")
    print(f"\n  Note: PPO typically needs 1M-10M steps for simple envs.")
    print(f"  At 4 instances: 1M steps takes ~{1_000_000/(steps_per_hr*4):.1f}hr")


# ─────────────────────────────────────────────
# SECTION 6: SUMMARY
# ─────────────────────────────────────────────
def print_summary(avg_load, avg_start, rpc_results, ep_results):
    section("6 / Summary for Research Diary")
    print(f"""
  Environment
    Platform       : Balatrobot v1.4.1 (Windows, no headless)
    Server flags   : --fast --no-shaders --fps-cap 1000 --gamespeed 4
    Reset strategy : load() from saved state
    Seed           : {SEED}  (fixed for reproducibility)
    Deck           : {DECK}

  Reset Performance
    start() avg    : {avg_start:.3f}s
    load()  avg    : {avg_load:.3f}s
    Speedup        : {avg_start/avg_load:.2f}x  (load is faster)

  RPC Latency
    health         : {rpc_results.get('health', 0):.1f}ms
    gamestate      : {rpc_results.get('gamestate', 0):.1f}ms
    play           : {rpc_results.get('play', 0):.1f}ms

  Episode Metrics (random agent baseline)
    Avg duration   : {ep_results['avg_dur']:.2f}s
    Avg steps      : {ep_results['avg_steps']:.1f}
    Avg ms/step    : {ep_results['avg_mps']:.1f}ms
    Eps / minute   : {ep_results['eps_per_min']:.1f}  (1 instance)
    Eps / hour     : {ep_results['eps_per_hr']:.0f}   (1 instance)

  Training Projection
    Instances      : 4 parallel  (separate ports)
    Projected eps/hr (4x): {ep_results['eps_per_hr']*4:.0f}
    Time for 50k eps (4x): {50_000/(ep_results['eps_per_min']*4)/60:.1f}hr
    """)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Balatrobot Comprehensive Benchmark                     ║")
    print("║   Strategy-Conditioned RL for Balatro  —  Group 7        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if not check_health():
        exit(1)

    avg_load, avg_start = bench_reset()
    rpc_results         = bench_rpc_latency()
    ep_results          = bench_episodes()

    if ep_results:
        training_projection(
            ep_results["eps_per_min"],
            ep_results["avg_dur"],
            ep_results["avg_steps"],
        )
        print_summary(avg_load, avg_start, rpc_results, ep_results)
