"""
train.py — PPO training with automatic Balatrobot instance management
======================================================================
Automatically starts, monitors, and restarts Balatrobot instances.
No more manual terminal juggling.

Usage:
    python train.py              # start instances + train
    python train.py --setup-only # just create save files

Requirements:
    pip install stable-baselines3 gymnasium requests numpy
"""

import os
from pathlib import Path
import random
import time
import argparse
import subprocess
import requests
import signal
import atexit

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.utils import set_random_seed

from env import BalatroEnv, DECK, STAKE


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

PORTS  = [12346, 12347, 12348, 12349]
#PORTS = random.sample(range(10000, 65535), 4)
SEEDS  = ["TRAIN01", "TRAIN02", "TRAIN03", "TRAIN04"]

SAVE_DIR = Path.cwd() / "balatro_saves"
MODEL_DIR       = "./models"
LOG_DIR         = "./logs"
TOTAL_STEPS     = 100_000
N_STEPS         = 256
BATCH_SIZE      = 64
N_EPOCHS        = 10
LEARNING_RATE   = 3e-4
GAMMA           = 0.99
EVAL_FREQ       = 2_000
CHECKPOINT_FREQ = 10_000

# Balatrobot server flags — maximum speed
BALATROBOT_FLAGS = [
    "--fast",
    "--no-shaders",
    "--fps-cap", "1000",
    "--gamespeed", "4",
]

HEALTH_TIMEOUT   = 5   # seconds to wait for instance to become healthy
HEALTH_INTERVAL  = 2    # seconds between health check polls


# ─────────────────────────────────────────────────────────────
# INSTANCE MANAGER
# ─────────────────────────────────────────────────────────────

class BalatrobotManager:
    """
    Starts and manages N Balatrobot instances, one per port.
    Automatically kills all instances on exit.
    """

    def __init__(self, ports: list):
        self.ports     = ports
        self.processes = {}   # port -> subprocess.Popen
        atexit.register(self.kill_all)
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print("\nShutdown signal received, killing Balatrobot instances...")
        self.kill_all()
        exit(0)

    def start_instance(self, port: int) -> bool:
        """Start a single Balatrobot instance on the given port."""
        # Kill existing process on this port if any
        if port in self.processes:
            self.kill_instance(port)

        cmd = ["uvx", "balatrobot", "serve", "--port", str(port)] + BALATROBOT_FLAGS
        print(f"  [port {port}] Starting: {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                
            )
            self.processes[port] = proc
            return True
        except FileNotFoundError:
            print(f"  [port {port}] ❌ 'uvx' not found. Is Balatrobot installed?")
            return False
        except Exception as e:
            print(f"  [port {port}] ❌ Failed to start: {e}")
            return False

    def wait_healthy(self, port: int) -> bool:
        """Poll until the instance on port responds to health check."""
        url      = f"http://127.0.0.1:{port}"
        deadline = time.time() + HEALTH_TIMEOUT
        print(f"  [port {port}] Waiting for health check...", end="", flush=True)

        while time.time() < deadline:
            try:
                r = requests.post(url, json={
                    "jsonrpc": "2.0", "method": "health", "params": {}, "id": 1
                }, timeout=3)
                if r.json().get("result", {}).get("status") == "ok":
                    print(" ✅")
                    return True
            except Exception:
                pass
            print(".", end="", flush=True)
            time.sleep(HEALTH_INTERVAL)

        print(f" ❌ timeout after {HEALTH_TIMEOUT}s")
        return False

    def start_all(self) -> bool:
        """Start all instances and wait for them to be healthy."""
        print(f"Starting {len(self.ports)} Balatrobot instances...\n")

        # Start all instances first (in parallel)
        for port in self.ports:
            self.start_instance(port)
            time.sleep(1)   # slight stagger to avoid Steam conflicts

        # Then wait for all to become healthy
        print()
        all_ok = True
        for port in self.ports:
            if not self.wait_healthy(port):
                all_ok = False
                break

        return all_ok

    def kill_instance(self, port: int):
        proc = self.processes.get(port)
        if proc and proc.poll() is None:
            try:
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # kill entire group
                proc.wait(timeout=5)
            except ProcessLookupError:
                pass  # already dead
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # force kill
                except Exception:
                    proc.kill()
        self.processes.pop(port, None)

    def kill_all(self):
        for port in list(self.processes.keys()):
            self.kill_instance(port)
        print("All Balatrobot instances stopped.")


# ─────────────────────────────────────────────────────────────
# SAVE FILE SETUP
# ─────────────────────────────────────────────────────────────

def create_save_files(ports: list, seeds: list):
    """Create one fresh save file per instance."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    for port, seed in zip(ports, seeds):
        save_path = os.path.join(SAVE_DIR, f"fresh_{port}.jkr")
        url       = f"http://127.0.0.1:{port}"
        print(f"[port {port}] Creating save file (seed={seed})...")

        try:
            # Check current state
            r = requests.post(url, json={
                "jsonrpc": "2.0", "method": "gamestate", "params": {}, "id": 1
            }, timeout=10)
            state = r.json()["result"]["state"]

            if state != "MENU":
                requests.post(url, json={
                    "jsonrpc": "2.0", "method": "menu", "params": {}, "id": 1
                }, timeout=10)
                time.sleep(1)

            # Start fresh run with fixed seed
            requests.post(url, json={
                "jsonrpc": "2.0", "method": "start",
                "params": {"deck": DECK, "stake": STAKE, "seed": seed},
                "id": 1
            }, timeout=20)

            time.sleep(1)

            # Save
            requests.post(url, json={
                "jsonrpc": "2.0", "method": "save",
                "params": {"path": save_path},
                "id": 1
            }, timeout=10)

            print(f"  ✅ Saved to {save_path}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")


# ─────────────────────────────────────────────────────────────
# ENV FACTORY
# ─────────────────────────────────────────────────────────────

def make_env(port: int, seed: str, rank: int):
    save_path = os.path.join(SAVE_DIR, f"fresh_{port}.jkr")

    def _init():
        env = BalatroEnv(port=port, save_path=save_path, seed=seed)
        env.reset(seed=rank)
        return env

    set_random_seed(rank)
    return _init


# ─────────────────────────────────────────────────────────────
# STRATEGY LOG CALLBACK
# ─────────────────────────────────────────────────────────────

class StrategyLogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.strategy_counts = [0, 0, 0]

    def _on_step(self) -> bool:
        actions = self.locals.get("actions", [])
        for a in actions:
            if 0 <= int(a) < 3:
                self.strategy_counts[int(a)] += 1

        total = sum(self.strategy_counts)
        if total > 0 and total % 1000 < len(actions):
            self.logger.record("strategy/flush_pct",  100 * self.strategy_counts[0] / total)
            self.logger.record("strategy/pair_pct",   100 * self.strategy_counts[1] / total)
            self.logger.record("strategy/mult_pct",   100 * self.strategy_counts[2] / total)

        return True


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train(manager: BalatrobotManager):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    print(f"\nBuilding {len(PORTS)} parallel environments...")
    env_fns = [
        make_env(port, seed, rank)
        for rank, (port, seed) in enumerate(zip(PORTS, SEEDS))
    ]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, LOG_DIR)
    print(f"✅ Environments ready")

    eval_env = SubprocVecEnv([make_env(PORTS[0], SEEDS[0], 99)])
    eval_env = VecMonitor(eval_env)

    model = PPO(
        policy          = "MlpPolicy",
        env             = vec_env,
        n_steps         = N_STEPS,
        batch_size      = BATCH_SIZE,
        n_epochs        = N_EPOCHS,
        learning_rate   = LEARNING_RATE,
        gamma           = GAMMA,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,
        verbose         = 1,
        tensorboard_log = LOG_DIR,
        policy_kwargs   = dict(net_arch=[64, 64]),
    )

    callbacks = [
        CheckpointCallback(
            save_freq   = CHECKPOINT_FREQ // len(PORTS),
            save_path   = MODEL_DIR,
            name_prefix = "balatro_ppo",
        ),
        EvalCallback(
            eval_env,
            eval_freq            = EVAL_FREQ // len(PORTS),
            best_model_save_path = os.path.join(MODEL_DIR, "best"),
            log_path             = LOG_DIR,
            deterministic        = True,
            render               = False,
        ),
        StrategyLogCallback(),
    ]

    print(f"\nStarting PPO training — {TOTAL_STEPS:,} steps")
    print(f"Monitor: tensorboard --logdir {LOG_DIR}\n")
    t_start = time.time()

    model.learn(
        total_timesteps = TOTAL_STEPS,
        callback        = callbacks,
        progress_bar    = True,
    )

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")

    final_path = os.path.join(MODEL_DIR, "balatro_ppo_final")
    model.save(final_path)
    print(f"Model saved to {final_path}")

    vec_env.close()
    eval_env.close()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-only", action="store_true",
                        help="Only start instances and create save files")
    parser.add_argument("--no-launch", action="store_true",
                        help="Skip launching instances (if already running)")
    args = parser.parse_args()

    manager = BalatrobotManager(PORTS)

    if not args.no_launch:
        ok = manager.start_all()
        if not ok:
            print("\n❌ Not all instances healthy. Check Balatrobot installation.")
            manager.kill_all()
            exit(1)
        print(f"\n✅ All {len(PORTS)} instances running\n")
        time.sleep(3)   # give games time to fully initialize

    # Create save files
    missing = [
        p for p in PORTS
        if not os.path.exists(os.path.join(SAVE_DIR, f"fresh_{p}.jkr"))
    ]
    if missing:
        print("Creating save files...")
        create_save_files(PORTS, SEEDS)
        print()

    if args.setup_only:
        print("Setup complete. Run 'python train.py --no-launch' to train.")
        exit(0)

    train(manager)
