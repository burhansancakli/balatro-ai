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
from typing import Any
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from game_status_callback import GameStatusCallback

from env import BalatroEnv
from episode_recorder import EpisodeRecorderWrapper
from config import (
    DECK, STAKE, SAVE_DIR, MODEL_DIR, LOG_DIR,
    TOTAL_STEPS, N_STEPS, BATCH_SIZE, N_EPOCHS,
    LEARNING_RATE, LR_FINAL_FRACTION, GAMMA,
    CLIP_RANGE_START, CLIP_RANGE_FINAL,
    ENT_COEF_START, ENT_COEF_FINAL, NET_ARCH,
    EVAL_FREQ, N_EVAL_EPISODES, CHECKPOINT_FREQ,
)
from instance_manager import BalatrobotManager
from shop_action_log_callback import ShopActionLogCallback
from research_callback import ResearchCallback
from run_checkpoint_callback import RunCheckpointCallback


# ─────────────────────────────────────────────────────────────
# SCHEDULES & CALLBACKS
# ─────────────────────────────────────────────────────────────

def linear_schedule(start: float, end: float):
    """SB3 schedule: interpolates from start (progress=1) to end (progress=0)."""
    def fn(progress_remaining: float) -> float:
        return end + progress_remaining * (start - end)
    return fn


class EntropyAnnealCallback(BaseCallback):
    """Linearly anneal the entropy bonus over training.

    High entropy early makes the agent try all strategies and shop
    actions; low entropy late lets it commit to what works. SB3 has
    no built-in ent_coef schedule, so this sets model.ent_coef before
    each update.
    """

    def __init__(self, start: float, end: float, total_steps: int):
        super().__init__(verbose=0)
        self._start = start
        self._end = end
        self._total = max(total_steps, 1)

    def _on_rollout_end(self) -> None:
        progress_remaining = max(0.0, 1.0 - self.num_timesteps / self._total)
        self.model.ent_coef = self._end + progress_remaining * (self._start - self._end)
        self.logger.record("train/ent_coef_current", self.model.ent_coef)

    def _on_step(self) -> bool:
        return True


# ─────────────────────────────────────────────────────────────
# ENV FACTORY
# ─────────────────────────────────────────────────────────────

def make_env(port: int, seed: str, rank: int, backend=None, record_dir: str = None):
    save_path = os.path.join(SAVE_DIR, f"fresh_{seed}.jkr")

    def _init():
        env = BalatroEnv(port=port, save_path=save_path, seed=seed, backend=backend)
        # rank seeds each env's RNG stream — with randomize_seeds this
        # gives every env its own endless stream of unseen game seeds
        env.reset(seed=rank)
        if record_dir:
            env = EpisodeRecorderWrapper(env, record_dir=record_dir, rank=rank)
        return env

    set_random_seed(rank)
    return _init


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train(backends: list, ports: list, seeds: list, resume_path: str = None,
          use_emulator: bool = False, total_steps: int = TOTAL_STEPS,
          device_arg: str = "auto"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    session_id = time.strftime("%Y%m%d_%H%M%S")
    record_dir = os.path.join(LOG_DIR, "episodes", session_id)
    print(f"\nRecording episodes to: {record_dir}")

    print(f"\nBuilding {len(ports)} parallel environments...")
    env_fns: Any = [
        make_env(port, seed, rank, backend=backends[rank] if backends else None, record_dir=record_dir)
        for rank, (port, seed) in enumerate(zip(ports, seeds))
    ]

    # Each env step is compute-heavy (plays a whole ante with Monte Carlo
    # discard evaluation), so with multiple envs SubprocVecEnv's true
    # process parallelism beats DummyVecEnv's sequential stepping even in
    # emulator mode — the IPC overhead is negligible next to the step cost.
    if use_emulator and len(env_fns) == 1:
        vec_env = DummyVecEnv(env_fns)
        print(f"  Using DummyVecEnv (single emulator env — no IPC needed)")
    else:
        vec_env = SubprocVecEnv(env_fns)
        print(f"  Using SubprocVecEnv ({len(env_fns)} parallel processes)")
    vec_env = VecMonitor(vec_env, LOG_DIR)
    print(f" Environments ready")

    if use_emulator:
        # The eval env MUST NOT share a SimBackend with a training env —
        # SimBackend is stateful, so a shared instance lets eval episodes
        # clobber the training env's game state mid-rollout (crashes with
        # IllegalActionError phase mismatches at every EVAL_FREQ boundary).
        from emulator.bridge import SimBackend
        eval_backend = SimBackend(simplified=True)
        eval_env = DummyVecEnv([make_env(ports[0], seeds[0], 99, backend=eval_backend)])
    else:
        eval_env = SubprocVecEnv([make_env(ports[0], seeds[0], 99, backend=backends[0] if backends else None)])
    eval_env = VecMonitor(eval_env)

    # For a small MLP policy, CPU beats GPU: per-update tensors are tiny,
    # so GPU transfer latency outweighs any compute win (SB3 docs
    # recommend CPU for MlpPolicy PPO). Use --device cuda to override.
    if device_arg == "auto":
        device = "cpu"
    else:
        device = device_arg
    print(f"Using device: {device}")

    if resume_path:
        print(f"\nResuming training from {resume_path}")
        model = PPO.load(resume_path, env=vec_env, tensorboard_log=LOG_DIR, device=device)
        # Extract step count from filename (e.g. balatro_ppo_44000_steps.zip)
        import re
        match = re.search(r"(\d+)_steps", Path(resume_path).stem)
        if match:
            start_steps = int(match.group(1))
            model.num_timesteps = start_steps
            print(f"  Resuming from step {start_steps:,}")
    else:
        model = PPO(
            policy          = "MlpPolicy",
            env             = vec_env,
            n_steps         = N_STEPS,
            batch_size      = BATCH_SIZE,
            n_epochs        = N_EPOCHS,
            learning_rate   = linear_schedule(LEARNING_RATE, LEARNING_RATE * LR_FINAL_FRACTION),
            gamma           = GAMMA,
            gae_lambda      = 0.95,
            clip_range      = linear_schedule(CLIP_RANGE_START, CLIP_RANGE_FINAL),
            ent_coef        = ENT_COEF_START,
            verbose         = 1,
            tensorboard_log = LOG_DIR,
            policy_kwargs   = dict(net_arch=NET_ARCH),
            device          = device,
        )

    callbacks = [
        RunCheckpointCallback(
            save_freq   = max(1, CHECKPOINT_FREQ // len(ports)),
            save_path   = MODEL_DIR,
        ),
        EvalCallback(
            eval_env,
            eval_freq            = max(1, EVAL_FREQ // len(ports)),
            n_eval_episodes      = N_EVAL_EPISODES,
            best_model_save_path = os.path.join(MODEL_DIR, "best"),
            log_path             = LOG_DIR,
            deterministic        = True,
            render               = False,
        ),
        EntropyAnnealCallback(ENT_COEF_START, ENT_COEF_FINAL, total_steps),
        ShopActionLogCallback(),
        ResearchCallback(),
        GameStatusCallback(total_steps=total_steps),
    ]

    print(f"\nStarting PPO training — {total_steps:,} steps")
    print(f"Monitor: tensorboard --logdir {LOG_DIR}\n")
    t_start = time.time()

    model.learn(
        total_timesteps = total_steps,
        callback        = callbacks,
        log_interval    = 1,
        progress_bar    = False,
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
    parser.add_argument("-n", "--instances", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a saved model .zip to resume training from")
    parser.add_argument("--emulator", action="store_true",
                        help="Run using jackdaw emulator (no Balatrobot needed)")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS,
                        help=f"Total training timesteps (default: {TOTAL_STEPS:,})")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Torch device (default: auto = cpu, which is fastest for this small MLP)")

    args, unknown_args = parser.parse_known_args()
    from instance_manager import BALATROBOT_FLAGS
    BALATROBOT_FLAGS.extend(unknown_args)

    PORTS = random.sample(range(10000, 65535), args.instances)
    SEEDS = [f"TRAIN{i:02d}" for i in range(2, args.instances + 2)]

    backends = BalatrobotManager(PORTS, emulator=args.emulator, simplified=args.emulator).start()

    print(f"\n All {len(PORTS)} instances ready\n")
    if not args.emulator:
        time.sleep(3)   # give games time to fully initialize

    if args.setup_only:
        print("Setup complete. Run 'python train.py' to train.")
        exit(0)

    train(backends, PORTS, SEEDS, resume_path=args.resume, use_emulator=args.emulator,
          total_steps=args.steps, device_arg=args.device)
