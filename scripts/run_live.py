"""Run a trained MaskablePPO agent through Balatro episodes.

Usage::

    # Local (in-process engine)
    python scripts/run_live.py --episodes 10

    # Live (connect to running Balatro + balatrobot mod)
    python scripts/run_live.py --live --episodes 5
    python scripts/run_live.py --live --host 192.168.1.42 --port 12346
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from sb3_contrib import MaskablePPO

from jackdaw.env.game_interface import DirectAdapter, GameAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv


def _make_local_factory() -> Callable[[], GameAdapter]:
    return DirectAdapter


def _make_live_factory(host: str, port: int) -> Callable[[], GameAdapter]:
    def factory() -> GameAdapter:
        from jackdaw.bridge.backend import LiveBackend
        from jackdaw.env.game_interface import BridgeAdapter

        return BridgeAdapter(LiveBackend(host, port))

    return factory


def make_env(
    adapter_factory: Callable[[], GameAdapter],
    seed: int = 42,
    max_steps: int = 10_000,
) -> BalatroGymnasiumEnv:
    return BalatroGymnasiumEnv(
        adapter_factory=adapter_factory,
        max_steps=max_steps,
        seed_prefix=f"LIVE_{seed}",
    )


def run_episode(env: BalatroGymnasiumEnv, model: MaskablePPO) -> dict:
    """Run a single episode and return stats."""
    obs, info = env.reset()
    mask = info.get("action_mask")
    total_reward = 0.0
    steps = 0

    terminated, truncated = False, False
    while not (terminated or truncated):
        if mask is None or (hasattr(mask, "any") and not mask.any()):
            print(f"  step {steps}: no legal actions, breaking")
            break  # no legal actions — episode ended
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        action_idx = int(action)
        # Clamp to legal range (the wrapper also clamps, but this avoids
        # hitting the empty-action-table guard on the final step)
        if hasattr(mask, "sum"):
            n_legal = int(mask.sum())
            if n_legal > 0 and action_idx >= n_legal:
                action_idx = n_legal - 1

        # Debug: show what the model is doing
        action_table = env._action_table
        gs = info.get("raw_state", {})
        phase = gs.get("phase", "?")
        print(f"  step {steps}: phase={phase}, n_actions={len(action_table)}, n_legal={int(mask.sum()) if mask is not None else 0}")
        if action_idx < len(action_table):
            fa = action_table[action_idx]
            from jackdaw.env.action_space import ActionType as AT
            at_name = AT(fa.action_type).name if fa.action_type < len(AT) else str(fa.action_type)
            print(f"    → action_type={at_name}, entity={fa.entity_target}, cards={fa.card_target}")
        else:
            print(f"    → action_idx={action_idx} (out of range, table len={len(action_table)})")

        obs, reward, terminated, truncated, info = env.step(action_idx)
        mask = info.get("action_mask")
        total_reward += reward
        steps += 1

        if terminated or truncated:
            gs2 = info.get("raw_state", {})
            phase2 = gs2.get("phase", "?")
            print(f"    → EPISODE END: terminated={terminated}, truncated={truncated}, phase={phase2}")

    return {
        "steps": steps,
        "total_reward": total_reward,
        "ante_reached": info.get("balatro/ante_reached", 0),
        "rounds_beaten": info.get("balatro/rounds_beaten", 0),
        "won": info.get("balatro/won", False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained PPO agent on Balatro")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/balatro_ppo/balatro_ppo",
        help="Path to saved MaskablePPO model (without .zip extension)",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=10_000, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment")

    # Live / local mode
    parser.add_argument(
        "--live",
        action="store_true",
        help="Connect to a running Balatro instance via balatrobot (HTTP JSON-RPC)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="balatrobot host")
    parser.add_argument("--port", type=int, default=12346, help="balatrobot port")
    args = parser.parse_args()

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        model_path_zip = model_path.with_suffix(".zip")
        if model_path_zip.exists():
            model_path = model_path_zip
        else:
            raise FileNotFoundError(
                f"Model not found at {args.model}. "
                f"Expected: {model_path} or {model_path_zip}"
            )

    print(f"Loading model from {model_path}...")
    model = MaskablePPO.load(str(model_path))
    print("Model loaded successfully!")

    # Build adapter factory
    if args.live:
        print(f"Connecting to live Balatro at {args.host}:{args.port}...")
        adapter_factory = _make_live_factory(args.host, args.port)
    else:
        print("Using local in-process engine.")
        adapter_factory = _make_local_factory()

    # Create environment
    env = make_env(adapter_factory=adapter_factory, seed=args.seed, max_steps=args.max_steps)

    # Run episodes
    print(f"\nRunning {args.episodes} episodes...\n")
    print(f"{'Ep':>3}  {'Steps':>5}  {'Reward':>8}  {'Ante':>4}  {'Rounds':>6}  {'Won':>5}")
    print("-" * 45)

    all_stats: list[dict] = []
    for ep in range(args.episodes):
        stats = run_episode(env, model)
        all_stats.append(stats)
        won_str = "✓" if stats["won"] else "✗"
        print(
            f"{ep + 1:>3}  {stats['steps']:>5}  "
            f"{stats['total_reward']:>8.3f}  {stats['ante_reached']:>4}  "
            f"{stats['rounds_beaten']:>6}  {won_str:>5}"
        )

    # Summary
    print("-" * 45)
    antes = [s["ante_reached"] for s in all_stats]
    rounds = [s["rounds_beaten"] for s in all_stats]
    wins = sum(s["won"] for s in all_stats)
    rewards = [s["total_reward"] for s in all_stats]
    
    print(f"\nSummary over {args.episodes} episodes:")
    print(f"  Win rate:        {wins / len(all_stats):.1%} ({wins}/{len(all_stats)})")
    print(f"  Mean ante:       {sum(antes) / len(antes):.1f} (max: {max(antes)})")
    print(f"  Mean rounds:     {sum(rounds) / len(rounds):.1f} (max: {max(rounds)})")
    print(f"  Mean reward:     {sum(rewards) / len(rewards):.3f}")


if __name__ == "__main__":
    main()
