"""Simplified game environment restrictions.

Mirrors the Lua mod that restricts Balatro to a smaller, more RL-friendly
state space.  Apply these restrictions on top of a normal run to get:

* Joker whitelist — only 14 simple scoring jokers appear in the shop
* Shop: jokers only — no vouchers, boosters, or consumables
* Fixed per-round resources — 8 hand cards, 4 hands, 4 discards
* Boss blinds disabled — no debuffs, no set-time penalties
* SkipBlind removed from legal actions — agent must always fight

Typical usage
-------------
After ``initialize_run``::

    gs = initialize_run(back_key, stake, seed)
    apply_to_run(gs)  # patches starting_params and marks gs["simplified"]

After each ``step`` that may have populated the shop::

    step(gs, action)
    if gs.get("phase") == GamePhase.SHOP:
        apply_to_shop(gs)

When computing legal actions::

    legal = filter_legal_actions(get_legal_actions(gs))
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

#: Joker center_keys allowed in the shop.
ALLOWED_JOKERS: frozenset[str] = frozenset(
    [
        "j_joker",
        "j_lusty_joker",
        "j_jolly",
        "j_droll",
        "j_crafty",
        "j_smeared",
        "j_zany",
        "j_mad",
        "j_sly",
        "j_wily",
        "j_abstract",
        "j_half",
        "j_scary_face",
        "j_greedy_joker",
    ]
)

HAND_SIZE: int = 8
HANDS_PER_ROUND: int = 4
DISCARDS_PER_ROUND: int = 4


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def apply_to_run(gs: dict[str, Any]) -> None:
    """Patch a freshly-initialized game state with simplified restrictions.

    Call this once after ``initialize_run``.  Sets the ``simplified`` flag
    in the game state so that ``game.py`` can disable boss blind effects
    at the point of blind selection.
    """
    sp = gs["starting_params"]
    sp["hand_size"] = HAND_SIZE
    sp["hands"] = HANDS_PER_ROUND
    sp["discards"] = DISCARDS_PER_ROUND

    # Propagate immediately to the live fields used by start_round
    gs["hand_size"] = HAND_SIZE
    rr = gs["round_resets"]
    rr["hands"] = HANDS_PER_ROUND
    rr["discards"] = DISCARDS_PER_ROUND

    # Flag consumed by game.py's _handle_select_blind to disable boss effects
    gs["simplified"] = True


def apply_to_shop(gs: dict[str, Any]) -> None:
    """Ensure the shop contains only allowed jokers, filling empty slots.

    Steps:
    1. Clear vouchers and boosters.
    2. Determine shop size: 1–3 jokers, picked randomly once per shop visit
       (keyed by ante+round+reroll-count so a fresh roll only happens on a
       genuinely new visit or an explicit reroll).
    3. On a new visit: keep only cards already in the allowed-joker
       whitelist, then fill any remaining slots with random picks from
       ALLOWED_JOKERS, avoiding duplicates already present in the shop.
    4. On repeat calls within the *same* visit (e.g. after the player buys
       a card): only strip disallowed cards — do NOT replenish slots that
       were emptied by a purchase. Refilling there would let the player buy
       the same slot over and over instead of permanently shrinking it.
    Safe to call after every step when in SHOP phase — it is idempotent.
    Rerolls (which regenerate ``shop_cards`` via ``_reroll_shop_cards``) are
    detected via ``round_scores.times_rerolled`` and treated as a new visit.
    """
    gs["shop_vouchers"] = []
    gs["shop_boosters"] = []

    rng = gs.get("rng")
    ante: int = gs.get("round_resets", {}).get("ante", 1)
    round_num: int = gs.get("round", 0)
    times_rerolled: int = gs.get("round_scores", {}).get("times_rerolled", 0)

    # A "visit" is a distinct (ante, round, reroll-count) — buying a card
    # does not change this key, so it does NOT trigger a refill; rerolling
    # bumps times_rerolled, which does.
    visit_key = (ante, round_num, times_rerolled)

    if gs.get("_simp_shop_visit_key") != visit_key:
        if rng is not None:
            seed_val = rng.seed(f"simp_shop_size_{ante}_{round_num}_{times_rerolled}")
            shop_max, _ = rng.element([1, 2, 3], seed_val)
            shop_max = int(shop_max)
        else:
            import random
            shop_max = random.randint(1, 3)
        gs["_simp_shop_max"] = shop_max
        gs["_simp_shop_visit_key"] = visit_key

        owned_keys = _owned_joker_keys(gs)
        kept = [
            c for c in gs.get("shop_cards", [])
            if _is_allowed_joker(c) and getattr(c, "center_key", "") not in owned_keys
        ]

        # Fill empty slots — avoid duplicating keys already in the shop or
        # already owned by the player
        slot = 0
        while len(kept) < shop_max:
            used_keys = {getattr(c, "center_key", "") for c in kept} | owned_keys
            available = [k for k in _ALLOWED_JOKERS_SORTED if k not in used_keys]
            if not available:
                # Every allowed joker is either in the shop or already owned
                # — allow shop-duplicates (but never an owned key) as a last resort
                available = [k for k in _ALLOWED_JOKERS_SORTED if k not in owned_keys] or _ALLOWED_JOKERS_SORTED
            key = _pick_joker_key(rng, ante, round_num, times_rerolled, slot, available)
            card = _make_joker_card(key, gs, ante)
            if card is not None:
                kept.append(card)
            slot += 1
            if slot > shop_max * 3:
                break  # safety: avoid infinite loop if card creation keeps failing

        gs["shop_cards"] = kept
    else:
         # Same visit as last call — strip anything disallowed or now-owned
        # (the player may have just bought one of these slots); leave
        # purchased slots empty (no refill).
        owned_keys = _owned_joker_keys(gs)
        gs["shop_cards"] = [
            c for c in gs.get("shop_cards", [])
            if _is_allowed_joker(c) and getattr(c, "center_key", "") not in owned_keys
        ]


def filter_legal_actions(actions: list[Any]) -> list[Any]:
    """Remove ``SkipBlind`` from a legal-action list."""
    from emulator.engine.actions import SkipBlind

    return [a for a in actions if not isinstance(a, SkipBlind)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Stable sorted list used for deterministic RNG picks
_ALLOWED_JOKERS_SORTED: list[str] = sorted(ALLOWED_JOKERS)


def _owned_joker_keys(gs: dict[str, Any]) -> set[str]:
    """Return the center_keys of jokers currently in the player's joker area."""
    return {getattr(j, "center_key", "") for j in gs.get("jokers", [])}


def _is_allowed_joker(card: Any) -> bool:
    """Return True if *card* is a joker in ALLOWED_JOKERS."""
    ability = getattr(card, "ability", {})
    if not isinstance(ability, dict):
        return False
    if ability.get("set") != "Joker":
        return False
    return getattr(card, "center_key", "") in ALLOWED_JOKERS


def _pick_joker_key(
    rng: Any, ante: int, round_num: int, times_rerolled: int, slot: int, available: list[str]
) -> str:
    """Pick a joker key from *available* using the game RNG for determinism."""
    if rng is not None:
        seed_val = rng.seed(f"simp_fill_{slot}_{ante}_{round_num}_{times_rerolled}")
        key, _ = rng.element(available, seed_val)
        return key
    import random
    return random.choice(available)


def _make_joker_card(key: str, gs: dict[str, Any], ante: int) -> Any:
    """Create a shop-priced joker Card for *key*."""
    try:
        from emulator.engine.card import Card
        card = Card()
        card.set_ability(key)
        card.set_cost(
            inflation=gs.get("inflation", 0),
            discount_percent=gs.get("discount_percent", 0),
            ante=ante,
        )
        return card
    except Exception:
        return None
