"""Adapter between jackdaw Action types and balatrobot JSON-RPC calls.

Converts our frozen dataclass actions into balatrobot RPC payloads,
and converts balatrobot game state responses into our game_state dict.

See ``docs/balatrobot-action-mapping.md`` for the full mapping table.
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine.actions import (
    Action,
    BuyCard,
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    OpenBooster,
    PickPackCard,
    PlayHand,
    RedeemVoucher,
    Reroll,
    SelectBlind,
    SellCard,
    SkipBlind,
    SkipPack,
    SortHand,
    SwapHandLeft,
    SwapHandRight,
    SwapJokersLeft,
    SwapJokersRight,
    UseConsumable,
)

# ---------------------------------------------------------------------------
# Action → balatrobot RPC
# ---------------------------------------------------------------------------

_STATE_MAP = {
    "BLIND_SELECT": GamePhase.BLIND_SELECT,
    "SELECTING_HAND": GamePhase.SELECTING_HAND,
    "ROUND_EVAL": GamePhase.ROUND_EVAL,
    "SHOP": GamePhase.SHOP,
    "SMODS_BOOSTER_OPENED": GamePhase.PACK_OPENING,
    "GAME_OVER": GamePhase.GAME_OVER,
}


def action_to_rpc(action: Action, game_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert a jackdaw Action to a balatrobot JSON-RPC method + params.

    Returns ``{"method": str, "params": dict}`` suitable for sending
    as a JSON-RPC 2.0 request body (minus jsonrpc/id fields).

    *game_state* is required for swap actions so that the full
    permutation array can be constructed for the balatrobot RPC.
    """
    match action:
        case PlayHand(card_indices=indices):
            return {"method": "play", "params": {"cards": list(indices)}}

        case Discard(card_indices=indices):
            return {"method": "discard", "params": {"cards": list(indices)}}

        case SelectBlind():
            return {"method": "select", "params": {}}

        case SkipBlind():
            return {"method": "skip", "params": {}}

        case BuyCard(shop_index=idx):
            return {"method": "buy", "params": {"card": idx}}

        case SellCard(area=area, card_index=idx):
            if area == "jokers":
                return {"method": "sell", "params": {"joker": idx}}
            else:
                return {"method": "sell", "params": {"consumable": idx}}

        case UseConsumable(card_index=idx, target_indices=targets):
            params: dict[str, Any] = {"consumable": idx}
            if targets:
                params["cards"] = list(targets)
            return {"method": "use", "params": params}

        case RedeemVoucher(card_index=idx):
            return {"method": "buy", "params": {"voucher": idx}}

        case OpenBooster(card_index=idx):
            return {"method": "buy", "params": {"pack": idx}}

        case PickPackCard(card_index=idx, target_indices=targets):
            params = {"card": idx}
            if targets:
                params["targets"] = list(targets)
            return {"method": "pack", "params": params}

        case SkipPack():
            return {"method": "pack", "params": {"skip": True}}

        case Reroll():
            return {"method": "reroll", "params": {}}

        case NextRound():
            return {"method": "next_round", "params": {}}

        case CashOut():
            return {"method": "cash_out", "params": {}}

        case SortHand(mode=mode):
            return {"method": "rearrange", "params": {"sort": mode}}

        case SwapHandLeft(idx=idx) | SwapHandRight(idx=idx):
            assert game_state is not None, "game_state required for swap actions"
            n = len(game_state.get("hand", []))
            order = list(range(n))
            other = idx - 1 if isinstance(action, SwapHandLeft) else idx + 1
            order[idx], order[other] = order[other], order[idx]
            return {"method": "rearrange", "params": {"hand": order}}

        case SwapJokersLeft(idx=idx) | SwapJokersRight(idx=idx):
            assert game_state is not None, "game_state required for swap actions"
            n = len(game_state.get("jokers", []))
            order = list(range(n))
            other = idx - 1 if isinstance(action, SwapJokersLeft) else idx + 1
            order[idx], order[other] = order[other], order[idx]
            return {"method": "rearrange", "params": {"jokers": order}}

        case _:
            raise ValueError(f"Unknown action type: {type(action).__name__}")


# ---------------------------------------------------------------------------
# Balatrobot state → game_state
# ---------------------------------------------------------------------------





def _bot_card_to_card(bot_card: dict[str, Any]) -> Any:
    """Reconstruct an engine Card from a balatrobot card dict.
    The balatrobot card format::
        {
          "id": int,
          "key": str,            # "H_A" for playing cards, "j_joker" for jokers, etc.
          "set": str,            # "DEFAULT", "JOKER", "TAROT", "PLANET", ...
          "label": str,
          "value": {"suit": "H", "rank": "A", "effect": ""},
          "modifier": {"seal": null, "edition": null, "enhancement": null, ...},
          "state": {"debuff": false, "hidden": false, "highlight": false},
          "cost": {"sell": int, "buy": int},
        }
    """
    from jackdaw.engine.card import Card

    card = Card()

    bot_set = bot_card.get("set", "DEFAULT")
    key = bot_card.get("key", "")
    value = bot_card.get("value") or {}
    modifier = bot_card.get("modifier") or {}
    state = bot_card.get("state") or {}
    cost = bot_card.get("cost") or {}

    # Guard: some fields may be lists instead of dicts (empty Lua tables)
    if not isinstance(value, dict):
        value = {}
    if not isinstance(modifier, dict):
        modifier = {}
    if not isinstance(state, dict):
        state = {}
    if not isinstance(cost, dict):
        cost = {}

    # --- Suit letter → name ---
    _SUIT_FROM_LETTER: dict[str, str] = {
        "H": "Hearts",
        "D": "Diamonds",
        "C": "Clubs",
        "S": "Spades",
    }
    _RANK_FROM_LETTER: dict[str, str] = {
        "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
        "7": "7", "8": "8", "9": "9", "T": "10",
        "J": "Jack", "Q": "Queen", "K": "King", "A": "Ace",
    }
    _ENHANCEMENT_FROM_BOT: dict[str | None, str] = {
        None: "c_base",
        "BONUS": "m_bonus",
        "MULT": "m_mult",
        "WILD": "m_wild",
        "GLASS": "m_glass",
        "STEEL": "m_steel",
        "STONE": "m_stone",
        "GOLD": "m_gold",
        "LUCKY": "m_lucky",
    }
    _SEAL_FROM_BOT: dict[str | None, str | None] = {
        None: None,
        "GOLD": "Gold",
        "RED": "Red",
        "BLUE": "Blue",
        "PURPLE": "Purple",
    }

    is_playing_card = bot_set == "DEFAULT"

    if is_playing_card:
        suit_letter = value.get("suit", "S")
        rank_letter = value.get("rank", "A")
        suit_name = _SUIT_FROM_LETTER.get(suit_letter, "Spades")
        rank_name = _RANK_FROM_LETTER.get(rank_letter, "Ace")
        card.set_base(key, suit_name, rank_name)
        # Enhancement
        enhancement = modifier.get("enhancement")
        center_key = _ENHANCEMENT_FROM_BOT.get(enhancement, "c_base")
        card.center_key = center_key
        if center_key != "c_base":
            try:
                card.set_ability(center_key)
            except (KeyError, FileNotFoundError):
                card.ability = {"set": "Enhanced", "name": enhancement or ""}
    else:
        # Non-playing card (joker, tarot, planet, spectral, voucher)
        card.center_key = key
        try:
            card.set_ability(key)
        except (KeyError, FileNotFoundError):
            # Build a minimal ability dict from what we know
            _SET_FROM_BOT: dict[str, str] = {
                "JOKER": "Joker",
                "TAROT": "Tarot",
                "PLANET": "Planet",
                "SPECTRAL": "Spectral",
                "VOUCHER": "Voucher",
                "BOOSTER": "Booster",
            }
            card.ability = {
                "name": bot_card.get("label", ""),
                "set": _SET_FROM_BOT.get(bot_set, bot_set),
            }

    # Seal
    card.seal = _SEAL_FROM_BOT.get(modifier.get("seal"))

    # Edition
    edition_str = modifier.get("edition")
    if edition_str:
        card.edition = {edition_str.lower(): True}

    # Cost
    card.cost = cost.get("buy", 0)
    card.sell_cost = cost.get("sell", 0)

    # Status
    card.debuff = state.get("debuff", False)
    if state.get("hidden"):
        card.facing = "back"

    # Stickers
    card.eternal = bool(modifier.get("eternal"))
    perish = modifier.get("perishable")
    if perish is not None:
        card.perishable = True
        card.perish_tally = int(perish)
    card.rental = bool(modifier.get("rental"))

    # Assign a unique sort_id
    from jackdaw.engine.card import _next_sort_id
    card.sort_id = bot_card.get("id", _next_sort_id())

    return card


def bot_state_to_game_state(bot: dict[str, Any]) -> dict[str, Any]:
    """Convert a balatrobot gamestate response to our game_state dict.

    Maps the key fields for validation and comparison. Does NOT create
    a fully functional game_state (no RNG, no Card objects) — this is
    for read-only comparison purposes.
    """
    gs: dict[str, Any] = {}

    # Phase
    state_str = bot.get("state", "")
    gs["phase"] = _STATE_MAP.get(state_str, state_str)

    # Economy
    gs["dollars"] = bot.get("money", 0)

    # Ante / round
    gs["round_resets"] = {"ante": bot.get("ante_num", 1)}
    gs["round"] = bot.get("round_num", 0)

    # Round state
    br = bot.get("round", {})
    gs["current_round"] = {
        "hands_left": br.get("hands_left", 0),
        "discards_left": br.get("discards_left", 0),
        "hands_played": br.get("hands_played", 0),
        "discards_used": br.get("discards_used", 0),
        "reroll_cost": br.get("reroll_cost", 5),
    }
    gs["chips"] = br.get("chips", 0)

    # Cards — build Card objects from bot response
    gs["hand"] = [_bot_card_to_card(c) for c in bot.get("hand", {}).get("cards", [])]
    gs["hand_size"] = bot.get("hand", {}).get("limit", 8)

    gs["deck"] = [_bot_card_to_card(c) for c in bot.get("cards", {}).get("cards", [])]
    # Keep legacy keys for compatibility
    gs["hand_keys"] = [c.card_key or c.center_key for c in gs["hand"]]
    gs["hand_count"] = len(gs["hand"])
    gs["deck_size"] = len(gs["deck"])

    gs["jokers"] = [_bot_card_to_card(c) for c in bot.get("jokers", {}).get("cards", [])]
    j_area = bot.get("jokers", {})
    gs["joker_slots"] = j_area.get("limit", 5)
    gs["joker_count"] = len(gs["jokers"])
    gs["joker_keys"] = [c.center_key for c in gs["jokers"]]

    gs["consumables"] = [_bot_card_to_card(c) for c in bot.get("consumables", {}).get("cards", [])]
    c_area = bot.get("consumables", {})
    gs["consumable_slots"] = c_area.get("limit", 2)
    gs["consumable_keys"] = [c.center_key for c in gs["consumables"]]

    # Blinds — find the current blind on deck
    blinds = bot.get("blinds", {})
    gs["blind_info"] = {}
    blind_on_deck = None
    for btype in ("small", "big", "boss"):
        bi = blinds.get(btype, {})
        status = bi.get("status", "")
        name = bi.get("name", "")
        score = bi.get("score", 0)
        gs["blind_info"][btype] = {
            "name": name,
            "status": status,
            "score": score,
            "tag_name": bi.get("tag_name", ""),
        }
         # Determine blind_on_deck from status
        if status == "SELECT":
            blind_on_deck = btype.capitalize()  # "Small", "Big", "Boss"
        elif status == "CURRENT":
            # Create a Blind-like object with chips for score tracking
            from types import SimpleNamespace
            gs["blind"] = SimpleNamespace(chips=score, name=name)

    gs["blind_on_deck"] = blind_on_deck

    # Seed / deck / stake
    gs["seed"] = bot.get("seed", "")
    gs["deck_type"] = bot.get("deck", "")
    gs["stake_type"] = bot.get("stake", "")
    gs["won"] = bot.get("won", False)

    return gs


# ---------------------------------------------------------------------------
# Game state → balatrobot response (reverse direction)
# ---------------------------------------------------------------------------


def game_state_to_bot_response(gs: dict[str, Any]) -> dict[str, Any]:
    """Convert our game_state to balatrobot's JSON response format.

    Delegates to :func:`jackdaw.bridge.serializer.game_state_to_bot_response`.
    """
    from jackdaw.bridge.serializer import (
        game_state_to_bot_response as _serialize,
    )

    return _serialize(gs)


# ---------------------------------------------------------------------------
# Game state → comparison keys
# ---------------------------------------------------------------------------


def extract_comparison_keys(gs: dict[str, Any]) -> dict[str, Any]:
    """Extract the key fields from our game_state for comparison with bot state.

    Returns a flat dict suitable for field-by-field comparison with the
    output of :func:`bot_state_to_game_state`.
    """
    cr = gs.get("current_round", {})
    return {
        "dollars": gs.get("dollars", 0),
        "chips": gs.get("chips", 0),
        "ante": gs.get("round_resets", {}).get("ante", 1),
        "hands_left": cr.get("hands_left", 0),
        "discards_left": cr.get("discards_left", 0),
        "hand_count": len(gs.get("hand", [])),
        "deck_size": len(gs.get("deck", [])),
        "joker_count": len(gs.get("jokers", [])),
    }
