"""Gymnasium wrapper that records full episode trajectories to JSONL.

Usage::

    from jackdaw.env.episode_recorder import EpisodeRecorderWrapper

    base_env = BalatroGymnasiumEnv(...)
    env = EpisodeRecorderWrapper(base_env, record_dir="logs/episodes")

Each completed episode is written as one JSON line to
``<record_dir>/episodes.jsonl``.  The file is appended across training
runs so logs accumulate.  File writes are guarded by a threading lock so
that ``DummyVecEnv`` (which runs envs in the same process) is safe; for
``SubprocVecEnv`` each subprocess writes its own file.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

# Module-level lock — safe when DummyVecEnv runs all envs in one thread.
_write_lock = threading.Lock()


class EpisodeRecorderWrapper(gym.Wrapper):
    """Records episode trajectories (actions, rewards, game state) to JSONL.

    Parameters
    ----------
    env:
        The base Gymnasium environment to wrap.
    record_dir:
        Directory where ``episodes.jsonl`` will be written.
    """

    def __init__(self, env: gym.Env, record_dir: str | os.PathLike) -> None:
        super().__init__(env)
        self._record_dir = Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._out_path = self._record_dir / "episodes.jsonl"

        self._episode_num: int = 0
        self._step_actions: list[Any] = []
        self._step_rewards: list[float] = []
        self._current_seed: str = ""

        # Count episodes already in the file so numbering continues correctly.
        if self._out_path.exists():
            with open(self._out_path, encoding="utf-8") as f:
                self._episode_num = sum(1 for line in f if line.strip())

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._current_seed = info.get("seed", "")
        self._step_actions = []
        self._step_rewards = []
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_actions.append(int(action) if isinstance(action, (np.integer, int)) else action)
        self._step_rewards.append(float(reward))

        if terminated or truncated:
            self._flush_episode(info)
            self._episode_num += 1
            self._step_actions = []
            self._step_rewards = []

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_episode(self, final_info: dict[str, Any]) -> None:
        gs = self._get_raw_state()
        record = {
            "episode": self._episode_num,
            "seed": self._current_seed,
            "ante_reached": final_info.get("balatro/ante_reached", 0),
            "rounds_beaten": final_info.get("balatro/rounds_beaten", 0),
            "won": bool(final_info.get("balatro/won", False)),
            "total_reward": round(sum(self._step_rewards), 4),
            "steps": len(self._step_actions),
            "actions": self._step_actions,
            "hand_history": self._serialize_hand_history(gs),
            "action_log": list(gs.get("action_log", [])),
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _write_lock:
            with open(self._out_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _get_raw_state(self) -> dict[str, Any]:
        """Walk the wrapper stack to reach the engine game_state dict.

        Handles the actual Jackdaw hierarchy:
          EpisodeRecorderWrapper → BalatroGymnasiumEnv (self._inner)
            → BalatroEnvironment (self._adapter)
        """
        env = self.env
        # Fast path: BalatroGymnasiumEnv stores BalatroEnvironment as _inner
        if hasattr(env, "_inner"):
            inner = env._inner
            if hasattr(inner, "_adapter"):
                return inner._adapter.raw_state
        # Generic fallback: look for .adapter at each wrapper level
        while hasattr(env, "env"):
            if hasattr(env, "adapter"):
                return env.adapter.raw_state
            env = env.env
        if hasattr(env, "adapter"):
            return env.adapter.raw_state
        return {}

    @staticmethod
    def _serialize_hand_history(gs: dict[str, Any]) -> list[dict]:
        """Serialize hand_history to a JSON-safe list."""
        raw = gs.get("hand_history", [])
        out = []
        for entry in raw:
            destroyed_ids = {id(c) for c in entry.get("destroyed_cards", [])}
            scoring_ids = {id(c) for c in entry.get("scoring_cards", [])}
            cards = []
            for c in entry.get("cards", []):
                base = getattr(c, "base", None)
                if base is not None:
                    rank = getattr(base, "rank", None)
                    suit = getattr(base, "suit", None)
                    r = rank.value if hasattr(rank, "value") else str(rank)
                    s = suit.value if hasattr(suit, "value") else str(suit)
                    ability = c.ability if isinstance(c.ability, dict) else {}
                    cards.append({
                        "label": f"{r}{s}",
                        "rank": r, "suit": s,
                        "enhancement": ability.get("name", "") if ability.get("name", "Base") != "Base" else "",
                        "edition": c.edition,
                        "seal": c.seal,
                        "scored": id(c) in scoring_ids,
                        "destroyed": id(c) in destroyed_ids,
                    })
                else:
                    ability = c.ability if isinstance(c.ability, dict) else {}
                    cards.append({
                        "label": ability.get("name", getattr(c, "center_key", "?")),
                        "scored": False, "destroyed": id(c) in destroyed_ids,
                    })
            jokers = []
            for j in entry.get("jokers", []):
                ability = j.ability if isinstance(j.ability, dict) else {}
                jokers.append(ability.get("name", getattr(j, "center_key", "?")))
            out.append({
                "ante": entry["ante"],
                "round": entry["round"],
                "hand_num": entry["hand_num"],
                "hand_type": entry["hand_type"],
                "cards": cards,
                "jokers": jokers,
                "chips": entry["chips"],
                "mult": entry["mult"],
                "total": entry["total"],
                "dollars_earned": entry.get("dollars_earned", 0),
                "debuffed": entry.get("debuffed", False),
            })
        return out
