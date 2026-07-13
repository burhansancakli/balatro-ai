"""Gymnasium wrapper that records full episode trajectories to JSONL.

Usage::

    from episode_recorder import EpisodeRecorderWrapper

    base_env = BalatroEnv(...)
    env = EpisodeRecorderWrapper(base_env, record_dir="logs/episodes", rank=0)

Each completed episode is appended as one JSON line to
``<record_dir>/episodes_<rank>.jsonl``.  Per-rank files avoid write
collisions when SubprocVecEnv runs each env in its own process.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

_write_lock = threading.Lock()


class EpisodeRecorderWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, record_dir: str | Path, rank: int = 0) -> None:
        super().__init__(env)
        record_dir = Path(record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)
        self._out_path = record_dir / f"episodes_{rank}.jsonl"
        self._rank = rank

        self._episode_num: int = 0
        self._step_actions: list[int] = []
        self._step_rewards: list[float] = []
        self._action_log: list[dict] = []

        if self._out_path.exists():
            with open(self._out_path, encoding="utf-8") as f:
                self._episode_num = sum(1 for line in f if line.strip())

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._step_actions = []
        self._step_rewards = []
        self._action_log = []
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_actions.append(self._json_safe_action(action))
        self._step_rewards.append(float(reward))

        ante = info.get("ante", 0)
        round_num = info.get("round", 0)
        strategy = info.get("strategy_name", "")

        # Shop decision for this step (carries the declared strategy)
        shop_entry = self._parse_shop_action(info.get("action_label", "skip"), ante, round_num)
        if strategy:
            shop_entry["strategy"] = strategy
        self._action_log.append(shop_entry)

        # Hand plays / discards that happened while running the ante
        for log_str in info.get("hand_logs", []):
            entry = self._parse_hand_log(log_str)
            if entry:
                self._action_log.append(entry)

        if terminated or truncated:
            self._flush(info)
            self._episode_num += 1
            self._step_actions = []
            self._step_rewards = []
            self._action_log = []

        return obs, reward, terminated, truncated, info

    # ── Parsers ────────────────────────────────────────────────

    @staticmethod
    def _json_safe_action(action: Any):
        """Convert an action (int or MultiDiscrete ndarray) to a JSON type."""
        if isinstance(action, np.ndarray):
            return [int(x) for x in action.ravel()]
        if isinstance(action, (np.integer, int)):
            return int(action)
        if isinstance(action, (list, tuple)):
            return [int(x) for x in action]
        return action

    @staticmethod
    def _parse_shop_action(label: str, ante: int, round_num: int) -> dict:
        if label.startswith("buy:"):
            return {"type": "buy", "ante": ante, "round": round_num, "card": label[4:], "cost": 0}
        if label.startswith("sell:"):
            return {"type": "sell", "ante": ante, "round": round_num, "card": label[5:], "gold": 0}
        return {"type": "skip", "ante": ante, "round": round_num}

    @staticmethod
    def _parse_hand_log(log_str: str) -> dict | None:
        """Parse a hand_log string from env._play_hand into a structured entry.

        Formats produced by env.py:
          play:    "[seed X] ante=N round=N strategy=S hand=H played=[c1, c2, ...]"
          discard: "[seed X] ante=N round=N strategy=S action=discard discarded=[c1, ...]"
        """
        ante_m   = re.search(r'ante=(\d+)', log_str)
        round_m  = re.search(r'round=(\d+)', log_str)
        strat_m  = re.search(r'strategy=(\S+)', log_str)
        ante      = int(ante_m.group(1))  if ante_m  else 0
        round_num = int(round_m.group(1)) if round_m else 0
        strategy  = strat_m.group(1) if strat_m else ""

        if "action=discard" in log_str:
            cards_m = re.search(r'discarded=\[([^\]]*)\]', log_str)
            cards = [c.strip() for c in cards_m.group(1).split(",")] if cards_m else []
            entry = {"type": "discard", "ante": ante, "round": round_num, "cards": cards}
        elif "hand=" in log_str:
            hand_m  = re.search(r'hand=(\S+)',          log_str)
            cards_m = re.search(r'played=\[([^\]]*)\]', log_str)
            hand_type = hand_m.group(1) if hand_m else "?"
            cards = [c.strip() for c in cards_m.group(1).split(",")] if cards_m else []
            entry = {"type": "play", "ante": ante, "round": round_num,
                     "hand_type": hand_type, "cards": cards, "total": 0}
        else:
            return None

        if strategy:
            # Normalize enum name (MULT_BUILD) to display form (Mult Build)
            entry["strategy"] = strategy.replace("_", " ").title()
        return entry

    # ── Flush ──────────────────────────────────────────────────

    def _flush(self, final_info: dict) -> None:
        gs = getattr(self.env, "_last_gamestate", {}) or {}
        record = {
            "episode":       self._episode_num,
            "seed":          getattr(self.env, "seed", ""),
            "ante_reached":  final_info.get("ante_reached", 0),
            "rounds_beaten": final_info.get("round", 0),
            "won":           bool(final_info.get("won", False)),
            "total_reward":  round(sum(self._step_rewards), 4),
            "steps":         len(self._step_actions),
            "actions":       self._step_actions,
            "action_log":    self._action_log,
            "hand_history":  [],  # derived from action_log in the UI
            "jokers_at_end": self._serialize_jokers(gs),
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _write_lock:
            with open(self._out_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    @staticmethod
    def _serialize_jokers(gs: dict) -> list[dict]:
        jokers = gs.get("jokers", {}) or {}
        cards  = jokers.get("cards", []) or []
        return [{"label": c.get("label", "?"), "key": c.get("key", "")} for c in cards]
