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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from game_status_callback import GameStatusCallback

from env import BalatroEnv
from config import (
    DECK, STAKE, SAVE_DIR, MODEL_DIR, LOG_DIR,
    TOTAL_STEPS, N_STEPS, BATCH_SIZE, N_EPOCHS,
    LEARNING_RATE, GAMMA, EVAL_FREQ, CHECKPOINT_FREQ,
)
from instance_manager import BalatrobotManager
from shop_action_log_callback import ShopActionLogCallback
from research_callback import ResearchCallback
from run_checkpoint_callback import RunCheckpointCallback


# ─────────────────────────────────────────────────────────────
# ENV FACTORY
# ─────────────────────────────────────────────────────────────

def make_env(port: int, seed: str, rank: int, backend=None):
    save_path = os.path.join(SAVE_DIR, f"fresh_{seed}.jkr")

    def _init():
        env = BalatroEnv(port=port, save_path=save_path, seed=seed, backend=backend)
        env.reset(seed=rank)
        return env

    set_random_seed(rank)
    return _init


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train(backends: list, ports: list, seeds: list, resume_path: str = None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    print(f"\nBuilding {len(ports)} parallel environments...")
    env_fns: Any = [
        make_env(port, seed, rank, backend=backends[rank] if backends else None)
        for rank, (port, seed) in enumerate(zip(ports, seeds))
    ]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, LOG_DIR)
    print(f" Environments ready")

    eval_env = SubprocVecEnv([make_env(ports[0], seeds[0], 99, backend=backends[0] if backends else None)])
    eval_env = VecMonitor(eval_env)

    if resume_path:
        print(f"\nResuming training from {resume_path}")
        model = PPO.load(resume_path, env=vec_env, tensorboard_log=LOG_DIR)
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
            learning_rate   = LEARNING_RATE,
            gamma           = GAMMA,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.05,
            verbose         = 1,
            tensorboard_log = LOG_DIR,
            policy_kwargs   = dict(net_arch=[64, 64]),
        )

    callbacks = [
        RunCheckpointCallback(
            save_freq   = CHECKPOINT_FREQ // len(ports),
            save_path   = MODEL_DIR,
        ),
        EvalCallback(
            eval_env,
            eval_freq            = EVAL_FREQ // len(ports),
            best_model_save_path = os.path.join(MODEL_DIR, "best"),
            log_path             = LOG_DIR,
            deterministic        = True,
            render               = False,
        ),
        ShopActionLogCallback(),
        ResearchCallback(),
        GameStatusCallback(),
    ]

    print(f"\nStarting PPO training — {TOTAL_STEPS:,} steps")
    print(f"Monitor: tensorboard --logdir {LOG_DIR}\n")
    t_start = time.time()

    model.learn(
        total_timesteps = TOTAL_STEPS,
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
    parser.add_argument("-n", "--instances", type=int, default=1,
                        help="Number of parallel Balatrobot instances (default: 1)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a saved model .zip to resume training from")
    parser.add_argument("--emulator", action="store_true",
                        help="Run using jackdaw emulator (no Balatrobot needed)")
    
    args, unknown_args = parser.parse_known_args()
    from instance_manager import BALATROBOT_FLAGS
    BALATROBOT_FLAGS.extend(unknown_args)

    PORTS = random.sample(range(10000, 65535), args.instances)
    SEEDS = [f"TRAIN{i:02d}" for i in range(2, args.instances + 2)]

    backends = BalatrobotManager(PORTS, emulator=args.emulator).start()

    print(f"\n All {len(PORTS)} instances ready\n")
    time.sleep(3)   # give games time to fully initialize

    if args.setup_only:
        print("Setup complete. Run 'python train.py' to train.")
        exit(0)

    train(backends, PORTS, SEEDS, resume_path=args.resume)
