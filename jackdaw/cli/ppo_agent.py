"""Load a saved MaskablePPO model and wrap it as a (gs, legal_actions) -> Action callable."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from jackdaw.engine.actions import Action
from jackdaw.env.action_space import (
    ActionType,
    FactoredAction,
    factored_to_engine_action,
    get_action_mask,
)
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.observation import encode_observation

MAX_ACTIONS = 500
CARD_COMBO_BUDGET = 200

_SPEC = balatro_game_spec()
_ENTITY_INFO: list[tuple[str, int, int]] = [
    (et.name, et.max_count, et.feature_dim) for et in _SPEC.entity_types
]

_SIMPLE_TYPES = frozenset(
    i for i, at in enumerate(_SPEC.action_types)
    if not at.needs_entity_target and not at.needs_card_select
)
_ENTITY_ONLY_TYPES = frozenset(
    i for i, at in enumerate(_SPEC.action_types)
    if at.needs_entity_target and not at.needs_card_select
)
_CARD_ONLY_TYPES = frozenset(
    i for i, at in enumerate(_SPEC.action_types)
    if at.needs_card_select and not at.needs_entity_target
)
_BOTH_TYPES = frozenset(
    i for i, at in enumerate(_SPEC.action_types)
    if at.needs_entity_target and at.needs_card_select
)

# Swap actions are pure reordering — strip them so a weak model can't loop on them
_SWAP_TYPES = frozenset(
    i for i, at in enumerate(_SPEC.action_types)
    if at.name.startswith("Swap")
)


def _card_combos(legal_cards: np.ndarray, min_sel: int, max_sel: int) -> list[tuple[int, ...]]:
    upper = min(len(legal_cards), max_sel)
    lower = min(min_sel, upper)
    result: list[tuple[int, ...]] = []
    for k in range(lower, upper + 1):
        result.extend(tuple(int(c) for c in combo) for combo in combinations(legal_cards, k))
    return result


def _build_action_table(gs: dict[str, Any]) -> tuple[list[FactoredAction], np.ndarray]:
    """Return (action_table, bool_mask) from the current game state."""
    mask = get_action_mask(gs)
    actions: list[FactoredAction] = []
    rng = np.random.default_rng()

    for t in range(len(mask.type_mask)):
        if not mask.type_mask[t] or t in _SWAP_TYPES:
            continue

        if t in _SIMPLE_TYPES:
            actions.append(FactoredAction(action_type=t))

        elif t in _ENTITY_ONLY_TYPES:
            if t in mask.entity_masks:
                for idx in np.nonzero(mask.entity_masks[t])[0]:
                    actions.append(FactoredAction(action_type=t, entity_target=int(idx)))

        elif t in _CARD_ONLY_TYPES:
            legal_cards = np.nonzero(mask.card_mask)[0]
            combos = _card_combos(legal_cards, mask.min_card_select, mask.max_card_select)
            if len(combos) > CARD_COMBO_BUDGET:
                idx = rng.choice(len(combos), size=CARD_COMBO_BUDGET, replace=False)
                combos = [combos[i] for i in sorted(idx)]
            for combo in combos:
                actions.append(FactoredAction(action_type=t, card_target=combo))

        elif t in _BOTH_TYPES:
            if t in mask.entity_masks:
                legal_cards = np.nonzero(mask.card_mask)[0]
                for eidx in np.nonzero(mask.entity_masks[t])[0]:
                    combos = _card_combos(legal_cards, mask.min_card_select, mask.max_card_select)
                    for combo in combos:
                        actions.append(FactoredAction(action_type=t, entity_target=int(eidx), card_target=combo))

    bool_mask = np.zeros(MAX_ACTIONS, dtype=bool)
    bool_mask[: len(actions)] = True
    return actions, bool_mask


def _build_obs(gs: dict[str, Any]) -> dict[str, np.ndarray]:
    game_obs = encode_observation(gs).to_game_observation()
    obs: dict[str, np.ndarray] = {"global": game_obs.global_context.astype(np.float32)}
    counts: list[int] = []
    for name, max_count, feat_dim in _ENTITY_INFO:
        arr = game_obs.entities.get(name)
        padded = np.zeros((max_count, feat_dim), dtype=np.float32)
        if arr is not None and arr.shape[0] > 0:
            n = min(arr.shape[0], max_count)
            padded[:n] = arr[:n]
            counts.append(n)
        else:
            counts.append(0)
        obs[name] = padded
    obs["entity_counts"] = np.array(counts, dtype=np.float32)
    return obs


def make_ppo_agent(model_path: str) -> Any:
    """Load a saved MaskablePPO model and return an agent callable.

    The returned callable has signature ``(gs, legal_actions) -> Action``
    matching what ``GameObserver.simulate_watched`` expects.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        raise ImportError(
            "sb3_contrib is required for PPO agent.\n"
            "Run: pip install sb3-contrib stable-baselines3"
        )

    print(f"  Loading PPO model from {path} …")
    model = MaskablePPO.load(str(path))
    print("  Model loaded.\n")

    def ppo_agent(gs: dict[str, Any], legal_actions: list[Action]) -> Action:
        obs = _build_obs(gs)
        action_table, bool_mask = _build_action_table(gs)

        if not action_table:
            # Fallback: random from engine legal actions
            import random
            return random.choice(legal_actions)

        # Add batch dimension
        obs_batch = {k: v[np.newaxis] for k, v in obs.items()}
        action_idx, _ = model.predict(obs_batch, action_masks=bool_mask[np.newaxis], deterministic=True)
        action_idx = int(np.asarray(action_idx).flat[0])

        if action_idx >= len(action_table):
            import random
            return random.choice(legal_actions)

        fa = action_table[action_idx]
        try:
            return factored_to_engine_action(fa, gs)
        except Exception:
            import random
            return random.choice(legal_actions)

    return ppo_agent
