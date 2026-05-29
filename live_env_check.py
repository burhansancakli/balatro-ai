"""
live_env_check.py - One-step live BalatroEnv diagnostic.

Requires a running Balatrobot server on the selected port. This script is
intentionally verbose so it is useful when debugging RPC, reset, play, shop,
and joker-buying flow.
"""

import argparse
import os
from typing import Any

from env import BalatroEnv
from strategy import Strategy


SAVE_DIR = "C:/tmp/balatro_saves"


def joker_labels(state: dict[str, Any]) -> list[str]:
    """Return readable joker labels from a Balatrobot gamestate."""
    jokers = state.get("jokers", {}) or {}
    cards = jokers.get("cards", []) or []
    return [card.get("label", "?") for card in cards]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one live BalatroEnv step.")
    parser.add_argument("--port", type=int, default=12346)
    parser.add_argument(
        "--strategy",
        choices=[strategy.name for strategy in Strategy],
        default=Strategy.MULT_BUILD.name,
    )
    parser.add_argument("--save-path", default=None)
    args = parser.parse_args()

    save_path = args.save_path or os.path.join(SAVE_DIR, f"fresh_{args.port}.jkr")
    strategy = Strategy[args.strategy]

    print("== Live BalatroEnv check ==")
    print(f"Port      : {args.port}")
    print(f"Save path : {save_path}")
    print(f"Strategy  : {strategy.name}")
    print()

    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Save file not found: {save_path}")

    env = BalatroEnv(port=args.port, save_path=save_path)

    obs, reset_info = env.reset()
    print("Reset:")
    print(f"  obs shape      : {obs.shape}")
    print(f"  state          : {reset_info.get('state')}")
    print(f"  ante           : {reset_info.get('ante')}")
    print(f"  jokers bought  : {reset_info.get('jokers_bought')}")
    print()

    obs, reward, terminated, truncated, info = env.step(int(strategy))
    state = env._last_gamestate or {}

    print("Step:")
    print(f"  obs shape      : {obs.shape}")
    print(f"  reward         : {reward:.4f}")
    print(f"  terminated     : {terminated}")
    print(f"  truncated      : {truncated}")
    print(f"  state          : {state.get('state')}")
    print(f"  ante reached   : {info.get('ante_reached')}")
    print(f"  round          : {info.get('round')}")
    print(f"  jokers bought  : {info.get('jokers_bought')}")
    print(f"  jokers         : {joker_labels(state)}")
    print(f"  won            : {info.get('won')}")
    print()
    print("OK: live env step completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
