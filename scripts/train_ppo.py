"""Train MaskablePPO on the Balatro gymnasium environment.

Requires the ``train`` optional dependency group::

    uv pip install -e ".[train]"

Usage::

    python scripts/train_ppo.py --total-timesteps 500000
    python scripts/train_ppo.py --total-timesteps 50000 --log-dir runs/exp1 --seed 42
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from jackdaw.env.game_interface import DirectAdapter, SimplifiedDirectAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv


class BalatroMetricsCallback(BaseCallback):
    """Log Balatro-specific episode metrics to tensorboard."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._antes: list[int] = []
        self._rounds: list[int] = []
        self._wins: list[bool] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "balatro/ante_reached" in info:
                self._antes.append(info["balatro/ante_reached"])
                self._rounds.append(info["balatro/rounds_beaten"])
                self._wins.append(info["balatro/won"])
        return True

    def _on_rollout_end(self) -> None:
        if not self._antes:
            return
        self.logger.record("balatro/mean_ante_reached", np.mean(self._antes))
        self.logger.record("balatro/max_ante_reached", np.max(self._antes))
        self.logger.record("balatro/mean_rounds_beaten", np.mean(self._rounds))
        self.logger.record("balatro/win_rate", np.mean(self._wins))
        self._antes.clear()
        self._rounds.clear()
        self._wins.clear()


class ETACallback(BaseCallback):
    """Print a compact ETA line to the terminal after every rollout."""

    def __init__(self, total_timesteps: int, print_every: int = 1) -> None:
        super().__init__(verbose=0)
        self._total = total_timesteps
        self._print_every = print_every  # rollouts between prints
        self._rollout_count = 0
        self._start: float = 0.0

    def _on_training_start(self) -> None:
        self._start = time.monotonic()

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self._print_every != 0:
            return

        elapsed = time.monotonic() - self._start
        step = self.num_timesteps
        total = self._total

        pct = step / total * 100
        sps = step / elapsed if elapsed > 0 else 0
        remaining_s = (total - step) / sps if sps > 0 else 0

        eta_wall = datetime.now() + timedelta(seconds=remaining_s)
        eta_str = eta_wall.strftime("%H:%M:%S")

        def fmt_time(s: float) -> str:
            s = int(s)
            h, m = divmod(s, 3600)
            m, sec = divmod(m, 60)
            return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"

        bar_width = 20
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        print(
            f"  [{bar}] {pct:5.1f}%  "
            f"{step:>{len(str(total))},.0f} / {total:,.0f}  |  "
            f"{sps:,.0f} steps/s  |  "
            f"elapsed {fmt_time(elapsed)}  |  "
            f"ETA {fmt_time(remaining_s)} ({eta_str})"
        )

    def _on_step(self) -> bool:
        return True


def make_env(
    seed: int = 0,
    max_steps: int = 10_000,
    simplified: bool = False,
    record_dir: str | None = None,
) -> BalatroGymnasiumEnv:
    factory = SimplifiedDirectAdapter if simplified else DirectAdapter
    env = BalatroGymnasiumEnv(
        adapter_factory=factory,
        max_steps=max_steps,
        seed_prefix=f"PPO_{seed}",
        reward_shaping=True,
    )
    if record_dir:
        from jackdaw.env.episode_recorder import EpisodeRecorderWrapper
        env = EpisodeRecorderWrapper(env, record_dir)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Balatro")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--log-dir", type=str, default="runs/balatro_ppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument(
        "--simplified",
        action="store_true",
        default=False,
        help=(
            "Use simplified environment: joker whitelist, no vouchers/boosters, "
            "fixed 4 hands/4 discards, boss blinds disabled"
        ),
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default=None,
        help="Directory to write episode trajectory logs (JSONL). Disabled if not set.",
    )
    args = parser.parse_args()

    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    env = make_env(
        seed=args.seed,
        max_steps=args.max_steps,
        simplified=args.simplified,
        record_dir=args.record_dir,
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(log_path),
        ent_coef=0.05,
        learning_rate=1e-4,
        n_steps=4096,
        clip_range=0.15,
    )

    env_label = "simplified" if args.simplified else "standard"
    rec_note = f"  recording → {args.record_dir}" if args.record_dir else ""
    print(f"Training for {args.total_timesteps:,} timesteps  [{env_label} env]{rec_note}")
    print()
    try:
        callbacks = CallbackList([BalatroMetricsCallback(), ETACallback(args.total_timesteps)])
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving current model...")

    save_path = log_path / "balatro_ppo"
    model.save(str(save_path))
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
