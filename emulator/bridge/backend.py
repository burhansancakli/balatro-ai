"""Swappable backend interface for the JSON-RPC server.

Two implementations:

- **SimBackend** — runs the jackdaw engine in-process. Fast, headless,
  deterministic, zero runtime deps.
- **LiveBackend** — proxies all requests to a real balatrobot instance
  over HTTP. Requires a running Balatro + balatrobot mod.
"""

from __future__ import annotations

from typing import Any, Protocol

from emulator.engine.actions import GamePhase

# ---------------------------------------------------------------------------
# Balatrobot enum → engine value maps
# ---------------------------------------------------------------------------

DECK_FROM_BOT: dict[str, str] = {
    "RED": "b_red",
    "BLUE": "b_blue",
    "YELLOW": "b_yellow",
    "GREEN": "b_green",
    "BLACK": "b_black",
    "MAGIC": "b_magic",
    "NEBULA": "b_nebula",
    "GHOST": "b_ghost",
    "ABANDONED": "b_abandoned",
    "CHECKERED": "b_checkered",
    "ZODIAC": "b_zodiac",
    "PAINTED": "b_painted",
    "ANAGLYPH": "b_anaglyph",
    "PLASMA": "b_plasma",
    "ERRATIC": "b_erratic",
}

STAKE_FROM_BOT: dict[str, int] = {
    "WHITE": 1,
    "RED": 2,
    "GREEN": 3,
    "BLACK": 4,
    "BLUE": 5,
    "PURPLE": 6,
    "ORANGE": 7,
    "GOLD": 8,
}


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class RPCError(Exception):
    """JSON-RPC error with code, message, and optional data."""

    def __init__(
        self,
        code: int,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.data = data or {}
        super().__init__(message)


# Error codes
BAD_REQUEST = -32001
INVALID_STATE = -32002
NOT_ALLOWED = -32003


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Backend(Protocol):
    """Structural interface for a JSON-RPC backend."""

    def handle(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        """Handle a JSON-RPC method call. Returns the result dict."""
        ...


# ---------------------------------------------------------------------------
# SimBackend
# ---------------------------------------------------------------------------

# Methods that map to engine game actions (handled via rpc_to_action + step).
_ACTION_METHODS = frozenset(
    {
        "play",
        "discard",
        "select",
        "skip",
        "buy",
        "sell",
        "use",
        "reroll",
        "next_round",
        "cash_out",
        "pack",
        "rearrange",
    }
)


class SimBackend:
    """Backend that runs the jackdaw engine in-process.

    Parameters
    ----------
    simplified:
        If True, apply simplified-env restrictions (joker whitelist,
        no vouchers/boosters, fixed 4 hands/4 discards, boss blinds
        disabled, SkipBlind removed from legal actions).
    """

    def __init__(self, *, simplified: bool = False, fast: bool = False) -> None:
        self._gs: dict[str, Any] | None = None
        self._simplified = simplified
        self._fast = fast

    def handle(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        if params is None:
            params = {}

        if method == "health":
            return {"status": "ok"}

        if method == "start":
            return self._handle_start(params)

        if method == "menu":
            self._gs = None
            return {"state": "MENU"}

        if method == "gamestate":
            return self._require_gamestate()

        if method == "add":
            return self._handle_add(params)

        if method == "set":
            return self._handle_set(params)

        if method in _ACTION_METHODS:
            return self._handle_action(method, params)

        raise RPCError(BAD_REQUEST, f"Unknown method: {method!r}")

    # -- internal -----------------------------------------------------------

    def _handle_start(self, params: dict[str, Any]) -> dict[str, Any]:
        from emulator.engine.run_init import initialize_run

        deck_str = params.get("deck", "RED")
        stake_str = params.get("stake", "WHITE")
        seed = params.get("seed", "DEFAULT")

        back_key = DECK_FROM_BOT.get(deck_str, "b_red")
        stake = STAKE_FROM_BOT.get(stake_str, 1)

        self._gs = initialize_run(back_key, stake, seed)
        self._gs["phase"] = GamePhase.BLIND_SELECT
        self._gs["blind_on_deck"] = "Small"

        if self._simplified:
            from emulator.engine.simplified import apply_to_run
            apply_to_run(self._gs)

        return self._serialize()

    def _handle_action(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._gs is None:
            raise RPCError(INVALID_STATE, "No active run — call 'start' first")

        from emulator.bridge.deserializer import rpc_to_action

        try:
            action = rpc_to_action(method, params)
        except ValueError as exc:
            raise RPCError(BAD_REQUEST, str(exc)) from exc

        if action is None:
            return self._serialize()

        from emulator.engine.game import IllegalActionError, step

        try:
            step(self._gs, action)
        except IllegalActionError as exc:
            raise RPCError(NOT_ALLOWED, str(exc)) from exc

        if self._simplified and self._gs.get("phase") == GamePhase.SHOP:
            from emulator.engine.simplified import apply_to_shop
            apply_to_shop(self._gs)

        return self._serialize()

    def _handle_add(self, params: dict[str, Any]) -> dict[str, Any]:
        """Insert a card into the game (debug method matching balatrobot API)."""
        if self._gs is None:
            raise RPCError(INVALID_STATE, "No active run — call 'start' first")

        key = params.get("key")
        if not key:
            raise RPCError(BAD_REQUEST, "'add' requires a 'key' parameter")

        # Validate phase (matches balatrobot API constraints)
        phase = self._gs.get("phase")
        if key.startswith("v_") or key.startswith("p_"):
            if phase != GamePhase.SHOP:
                raise RPCError(
                    INVALID_STATE,
                    f"Method 'add' requires SHOP state for vouchers/packs (got {phase})",
                )
        elif phase not in (GamePhase.SELECTING_HAND, GamePhase.SHOP, GamePhase.ROUND_EVAL):
            raise RPCError(
                INVALID_STATE,
                "Method 'add' requires one of these states: SELECTING_HAND, SHOP, ROUND_EVAL",
            )

        from emulator.engine.card_factory import (
            RANK_LETTER,
            SUIT_LETTER,
            create_consumable,
            create_joker,
            create_playing_card,
        )
        from emulator.engine.card_utils import poll_edition

        seal = params.get("seal")  # Gold, Red, Blue, Purple
        edition_key = params.get("edition")  # foil, holo, polychrome, negative
        enhancement = params.get("enhancement", "c_base")  # m_bonus, m_mult, etc.
        eternal = params.get("eternal", False)
        perishable = params.get("perishable", False)
        rental = params.get("rental", False)

        edition = {edition_key: True} if edition_key else None

        if key.startswith("j_"):
            card = create_joker(
                key,
                edition=edition,
                eternal=eternal,
                perishable=perishable,
                rental=rental,
                hands_played=self._gs.get("hands_played", 0),
            )
            # Poll for edition to match balatrobot's create_card() behaviour.
            # Balatrobot's add command goes through the full Lua create_card()
            # pipeline which calls poll_edition(), so we must do the same to
            # keep the RNG stream in sync.
            rng = self._gs.get("rng")
            if rng is not None:
                ante = self._gs.get("round_resets", {}).get("ante", 1)
                polled = poll_edition(
                    "edi" + str(ante),
                    rng,
                    rate=self._gs.get("edition_rate", 1.0),
                )
                if edition is None:
                    card.set_edition(polled)
            card.set_cost()
            self._gs["jokers"].append(card)
            # Apply passive effects (hand_size, discards, joker_slots, etc.)
            old_hand_size = self._gs.get("hand_size", 8)
            card.add_to_deck(self._gs)
            # Also update current round counters for mid-round adds.
            # Only discards are adjusted immediately — extra discards are
            # usable in the current round.  Hand count changes (h_plays)
            # take effect on future rounds via round_resets (already set
            # by add_to_deck), matching balatrobot behaviour.
            cr = self._gs.get("current_round")
            if cr is not None:
                d_size = card.ability.get("d_size", 0)
                if d_size > 0:
                    cr["discards_left"] = cr.get("discards_left", 0) + d_size
            # If hand_size increased mid-round, draw cards to fill the new
            # size — matches balatrobot which immediately fills on add.
            if self._gs.get("hand_size", 8) > old_hand_size and phase == GamePhase.SELECTING_HAND:
                from emulator.engine.game import _draw_hand

                _draw_hand(self._gs)
        elif key.startswith("c_"):
            card = create_consumable(key)
            self._gs["consumables"].append(card)
        elif len(key) == 3 and key[1] == "_":
            # Playing card key like "H_A", "S_2"
            suit_letter, rank_letter = key[0], key[2]
            if suit_letter not in SUIT_LETTER or rank_letter not in RANK_LETTER:
                raise RPCError(BAD_REQUEST, f"Invalid playing card key: {key!r}")
            suit = SUIT_LETTER[suit_letter]
            rank = RANK_LETTER[rank_letter]
            card = create_playing_card(
                suit=suit,
                rank=rank,
                enhancement=enhancement,
                edition=edition,
                seal=seal,
            )
            # Add to hand if in SELECTING_HAND, otherwise to deck
            phase = self._gs.get("phase")
            if phase == GamePhase.SELECTING_HAND:
                self._gs["hand"].append(card)
            else:
                self._gs["deck"].append(card)
        else:
            raise RPCError(BAD_REQUEST, f"Unrecognised card key prefix: {key!r}")

        return self._serialize()

    def _handle_set(self, params: dict[str, Any]) -> dict[str, Any]:
        """Modify game state values (debug method matching balatrobot API)."""
        if self._gs is None:
            raise RPCError(INVALID_STATE, "No active run — call 'start' first")

        if "money" in params:
            self._gs["dollars"] = params["money"]

        if "hands" in params:
            self._gs["current_round"]["hands_left"] = params["hands"]

        if "discards" in params:
            self._gs["current_round"]["discards_left"] = params["discards"]

        if "ante" in params:
            self._gs["round_resets"]["ante"] = params["ante"]

        if "round" in params:
            self._gs["round"] = params["round"]

        if "chips" in params:
            self._gs["chips"] = params["chips"]

        if "shop" in params and params["shop"]:
            from emulator.engine.shop import populate_shop

            populate_shop(self._gs)

        return self._serialize()

    def _require_gamestate(self) -> dict[str, Any]:
        if self._gs is None:
            raise RPCError(INVALID_STATE, "No active run — call 'start' first")
        return self._serialize()

    def _serialize(self) -> dict[str, Any]:
        if self._fast:
            return self._fast_serialize()
        from emulator.bridge.serializer import game_state_to_bot_response

        return game_state_to_bot_response(self._gs)  # type: ignore[arg-type]

    def _fast_serialize(self) -> dict[str, Any]:
        """Minimal state dict for training — skips full card serialization.

        Only includes the fields that ``env.py`` and ``observations.py``
        actually read: phase, ante/round/money, round info, blind
        status+score, hand cards (rank/suit), joker labels, and shop
        card labels+keys+cost.  Skips deck, consumables, hand levels,
        blind tags, modifiers, editions, seals, etc.
        """
        from emulator.engine.actions import GamePhase
        from emulator.engine.data.prototypes import BLINDS
        from emulator.engine.blind import Blind

        gs = self._gs
        if gs is None:
            return {}

        _PHASE_TO_STATE = {
            GamePhase.BLIND_SELECT: "BLIND_SELECT",
            GamePhase.SELECTING_HAND: "SELECTING_HAND",
            GamePhase.ROUND_EVAL: "ROUND_EVAL",
            GamePhase.SHOP: "SHOP",
            GamePhase.PACK_OPENING: "SMODS_BOOSTER_OPENED",
            GamePhase.GAME_OVER: "GAME_OVER",
        }
        _BLIND_STATUS = {"Select": "SELECT", "Current": "CURRENT",
                         "Skipped": "SKIPPED", "Defeated": "DEFEATED"}

        rr = gs.get("round_resets", {})
        cr = gs.get("current_round", {})

        def _light_card(card):
            is_playing = card.base is not None
            if is_playing:
                return {
                    "value": {"rank": card.base.rank.value, "suit": card.base.suit.value,
                              "effect": ""},
                    "state": {"debuff": card.debuff, "hidden": card.facing == "back",
                              "highlight": False},
                    "key": card.card_key or "",
                    "label": f"{card.base.rank.value} of {card.base.suit.value}",
                    "set": "",
                    "cost": {"sell": card.sell_cost, "buy": card.cost},
                }
            return {
                "value": {"rank": "", "suit": "", "effect": ""},
                "state": {"debuff": card.debuff, "hidden": card.facing == "back",
                          "highlight": False},
                "key": card.center_key,
                "label": card.ability.get("name", ""),
                "set": card.ability.get("set", ""),
                "cost": {"sell": card.sell_cost, "buy": card.cost},
            }

        def _light_area(cards, limit):
            return {"count": len(cards), "limit": limit,
                    "highlighted_limit": 0,
                    "cards": [_light_card(c) for c in cards]}

        def _score_blind(blind_type: str) -> int:
            status_raw = rr.get("blind_states", {}).get(blind_type, "")
            blind_key = rr.get("blind_choices", {}).get(blind_type, "")
            active = gs.get("blind")
            if active and status_raw == "Current":
                return active.chips
            proto = BLINDS.get(blind_key)
            if proto:
                return Blind.create(
                    blind_key,
                    rr.get("ante", 1),
                    gs.get("modifiers", {}).get("scaling", 1),
                    gs.get("starting_params", {}).get("ante_scaling", 1.0),
                ).chips
            return 0

        blind_states = rr.get("blind_states", {})
        blinds = {}
        for bt in ("Small", "Big", "Boss"):
            status_raw = blind_states.get(bt, "")
            blinds[bt.lower()] = {
                "type": bt.upper(),
                "status": _BLIND_STATUS.get(status_raw, "UPCOMING"),
                "score": _score_blind(bt),
            }

        hand = gs.get("hand", [])
        jokers = gs.get("jokers", [])
        shop_cards = gs.get("shop_cards", [])

        return {
            "state": _PHASE_TO_STATE.get(gs.get("phase", ""), str(gs.get("phase", ""))),
            "round_num": gs.get("round", 0),
            "ante_num": rr.get("ante", 1),
            "money": gs.get("dollars", 0),
            "won": gs.get("won", False),
            "round": {
                "hands_left": cr.get("hands_left", 0),
                "discards_left": cr.get("discards_left", 0),
                "chips": gs.get("chips", 0),
                "hands_played": cr.get("hands_played", 0),
                "discards_used": cr.get("discards_used", 0),
                "reroll_cost": cr.get("reroll_cost", 5),
            },
            "blinds": blinds,
            "hand": _light_area(hand, gs.get("hand_size", 8)),
            "jokers": _light_area(jokers, gs.get("joker_slots", 5)),
            "shop": _light_area(shop_cards, len(shop_cards)),
        }


# ---------------------------------------------------------------------------
# LiveBackend
# ---------------------------------------------------------------------------


class LiveBackend:
    """Backend that proxies requests to a real balatrobot instance."""

    # Internal → balatrobot format maps for the "add" method
    _ENHANCEMENT_TO_BOT: dict[str, str] = {
        "m_bonus": "BONUS",
        "m_mult": "MULT",
        "m_wild": "WILD",
        "m_glass": "GLASS",
        "m_steel": "STEEL",
        "m_stone": "STONE",
        "m_gold": "GOLD",
        "m_lucky": "LUCKY",
    }
    _EDITION_TO_BOT: dict[str, str] = {
        "foil": "FOIL",
        "holo": "HOLO",
        "polychrome": "POLYCHROME",
        "negative": "NEGATIVE",
    }
    _SEAL_TO_BOT: dict[str, str] = {
        "Gold": "GOLD",
        "Red": "RED",
        "Blue": "BLUE",
        "Purple": "PURPLE",
    }

    def __init__(self, host: str = "127.0.0.1", port: int = 12346) -> None:
        self._url = f"http://{host}:{port}"

    def _convert_add_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Convert internal format to balatrobot format for 'add' params."""
        params = dict(params)  # shallow copy
        if "enhancement" in params:
            params["enhancement"] = self._ENHANCEMENT_TO_BOT.get(
                params["enhancement"], params["enhancement"]
            )
        if "edition" in params:
            params["edition"] = self._EDITION_TO_BOT.get(params["edition"], params["edition"])
        if "seal" in params:
            params["seal"] = self._SEAL_TO_BOT.get(params["seal"], params["seal"])
        return params

    def handle(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        import httpx

        rpc_params = params or {}
        if method == "add" and rpc_params:
            rpc_params = self._convert_add_params(rpc_params)

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "params": rpc_params,
            "id": 1,
        }
        resp = httpx.post(self._url, json=payload, timeout=10.0)
        data = resp.json()

        if "error" in data:
            err = data["error"]
            raise RPCError(
                code=err.get("code", -32000),
                message=err.get("message", "Unknown error"),
                data=err.get("data", {}),
            )

        return data.get("result", {})
