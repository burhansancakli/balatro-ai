"""GameObserver — hooks into the game loop and broadcasts state to renderers.

Two modes
---------
watch   Agent plays automatically; renderers update after every step.
play    Human picks every action via the browser; game waits for input.
"""

from __future__ import annotations

import queue
import time
from collections.abc import Callable
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
    get_legal_actions,
)
from jackdaw.engine.game import IllegalActionError, step
from jackdaw.engine.run_init import initialize_run

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUIT_SYMBOLS: dict[str, str] = {
    "Spades": "♠",
    "Hearts": "♥",
    "Diamonds": "♦",
    "Clubs": "♣",
}

RANK_SHORT: dict[str, str] = {
    "Ace": "A", "King": "K", "Queen": "Q", "Jack": "J",
    "10": "10", "9": "9", "8": "8", "7": "7",
    "6": "6", "5": "5", "4": "4", "3": "3", "2": "2",
}

# ---------------------------------------------------------------------------
# State serialization
# ---------------------------------------------------------------------------


def _serialize_card(card: Any) -> dict | None:
    if card is None:
        return None
    ability = card.ability if isinstance(card.ability, dict) else {}
    result: dict[str, Any] = {
        "key": card.center_key,
        "cost": card.cost,
        "edition": card.edition,
        "seal": card.seal,
        "debuff": card.debuff,
        "eternal": getattr(card, "eternal", ability.get("eternal", False)),
        "perishable": getattr(card, "perishable", ability.get("perishable", False)),
        "rental": getattr(card, "rental", ability.get("rental", False)),
    }

    if card.base is not None:
        suit = card.base.suit.value
        rank = card.base.rank.value
        symbol = SUIT_SYMBOLS.get(suit, suit)
        short = RANK_SHORT.get(rank, rank)
        result.update({
            "type": "playing",
            "suit": suit,
            "rank": rank,
            "suit_symbol": symbol,
            "rank_short": short,
            "label": f"{short}{symbol}",
            "red": suit in ("Hearts", "Diamonds"),
            "name": f"{short}{symbol}",
            "enhancement": ability.get("name", "") if ability.get("name", "Base") != "Base" else "",
        })
    else:
        card_set = ability.get("set", "Unknown")
        extra = {
            k: v for k, v in ability.items()
            if k not in ("name", "desc", "set") and isinstance(v, (int, float, str))
        }
        result.update({
            "type": card_set.lower(),
            "name": ability.get("name", card.center_key),
            "desc": ability.get("desc", ""),
            "set": card_set,
            "extra": extra if extra else {},
        })

    return result


def _action_label(action: Action) -> str:
    match action:
        case PlayHand(card_indices=ci):
            return f"Play Hand ({len(ci)} cards)" if ci else "Play Hand"
        case Discard(card_indices=ci):
            return f"Discard ({len(ci)} cards)" if ci else "Discard"
        case SelectBlind():
            return "Select Blind"
        case SkipBlind():
            return "Skip Blind"
        case BuyCard(shop_index=i):
            return f"Buy Card #{i + 1}"
        case SellCard(area=a, card_index=i):
            return f"Sell {a} #{i + 1}"
        case UseConsumable(card_index=i):
            return f"Use Consumable #{i + 1}"
        case RedeemVoucher(card_index=i):
            return f"Redeem Voucher #{i + 1}"
        case OpenBooster(card_index=i):
            return f"Open Pack #{i + 1}"
        case PickPackCard(card_index=i):
            return f"Pick Card #{i + 1}"
        case SkipPack():
            return "Skip Pack"
        case Reroll():
            return "Reroll Shop"
        case NextRound():
            return "Next Round"
        case CashOut():
            return "Cash Out"
        case SortHand(mode=m):
            return f"Sort by {m}"
        case _:
            return type(action).__name__


def _serialize_action(action: Action) -> dict:
    name = type(action).__name__
    match action:
        case PlayHand(card_indices=ci):
            return {"type": "PlayHand", "card_indices": list(ci), "label": "Play Hand", "needs_selection": len(ci) == 0}
        case Discard(card_indices=ci):
            return {"type": "Discard", "card_indices": list(ci), "label": "Discard", "needs_selection": len(ci) == 0}
        case SelectBlind():
            return {"type": "SelectBlind", "label": "Select Blind"}
        case SkipBlind():
            return {"type": "SkipBlind", "label": "Skip Blind"}
        case BuyCard(shop_index=i):
            return {"type": "BuyCard", "shop_index": i, "label": f"Buy"}
        case SellCard(area=a, card_index=i):
            return {"type": "SellCard", "area": a, "card_index": i, "label": "Sell"}
        case UseConsumable(card_index=i):
            return {"type": "UseConsumable", "card_index": i, "label": "Use", "needs_selection": False}
        case RedeemVoucher(card_index=i):
            return {"type": "RedeemVoucher", "card_index": i, "label": "Redeem"}
        case OpenBooster(card_index=i):
            return {"type": "OpenBooster", "card_index": i, "label": "Open"}
        case PickPackCard(card_index=i):
            return {"type": "PickPackCard", "card_index": i, "label": "Pick"}
        case SkipPack():
            return {"type": "SkipPack", "label": "Skip Pack"}
        case Reroll():
            return {"type": "Reroll", "label": "Reroll"}
        case NextRound():
            return {"type": "NextRound", "label": "Next Round"}
        case CashOut():
            return {"type": "CashOut", "label": "Cash Out"}
        case SortHand(mode=m):
            return {"type": "SortHand", "mode": m, "label": f"Sort {m.capitalize()}"}
        case SwapHandLeft(idx=i):
            return {"type": "SwapHandLeft", "idx": i, "label": f"← {i}"}
        case SwapHandRight(idx=i):
            return {"type": "SwapHandRight", "idx": i, "label": f"{i} →"}
        case SwapJokersLeft(idx=i):
            return {"type": "SwapJokersLeft", "idx": i, "label": f"← Joker {i}"}
        case SwapJokersRight(idx=i):
            return {"type": "SwapJokersRight", "idx": i, "label": f"Joker {i} →"}
        case _:
            return {"type": name, "label": name}


def deserialize_action(data: dict) -> Action:
    """Convert a browser-sent action dict to an Action object."""
    t = data["type"]
    match t:
        case "PlayHand":
            return PlayHand(card_indices=tuple(data.get("card_indices", [])))
        case "Discard":
            return Discard(card_indices=tuple(data.get("card_indices", [])))
        case "SelectBlind":
            return SelectBlind()
        case "SkipBlind":
            return SkipBlind()
        case "BuyCard":
            return BuyCard(shop_index=data["shop_index"])
        case "SellCard":
            return SellCard(area=data["area"], card_index=data["card_index"])
        case "UseConsumable":
            ti = data.get("target_indices")
            return UseConsumable(card_index=data["card_index"], target_indices=tuple(ti) if ti else None)
        case "RedeemVoucher":
            return RedeemVoucher(card_index=data["card_index"])
        case "OpenBooster":
            return OpenBooster(card_index=data["card_index"])
        case "PickPackCard":
            ti = data.get("target_indices")
            return PickPackCard(card_index=data["card_index"], target_indices=tuple(ti) if ti else None)
        case "SkipPack":
            return SkipPack()
        case "Reroll":
            return Reroll()
        case "NextRound":
            return NextRound()
        case "CashOut":
            return CashOut()
        case "SortHand":
            return SortHand(mode=data["mode"])
        case "SwapHandLeft":
            return SwapHandLeft(idx=data["idx"])
        case "SwapHandRight":
            return SwapHandRight(idx=data["idx"])
        case "SwapJokersLeft":
            return SwapJokersLeft(idx=data["idx"])
        case "SwapJokersRight":
            return SwapJokersRight(idx=data["idx"])
        case _:
            raise ValueError(f"Unknown action type: {t!r}")


def serialize_state(
    gs: dict[str, Any],
    legal_actions: list[Action] | None = None,
    last_action: Action | None = None,
    last_score: int | None = None,
    history: list[str] | None = None,
) -> dict[str, Any]:
    cr = gs.get("current_round", {})
    rr = gs.get("round_resets", {})
    blind = gs.get("blind")
    chips = gs.get("chips", 0)
    blind_chips = getattr(blind, "chips", None) if blind else None
    last_hand = cr.get("current_hand", {})
    blind_states = rr.get("blind_states", {})
    blind_choices = rr.get("blind_choices", {})

    return {
        # Phase & control
        "phase": str(gs.get("phase", "")),
        "ante": rr.get("ante", 1),
        "round": gs.get("round", 0),
        "blind_on_deck": gs.get("blind_on_deck", "Small"),
        "blind_name": getattr(blind, "name", None) if blind else None,
        "blind_chips": blind_chips,
        "chips": chips,
        "chips_progress": round(chips / blind_chips, 4) if blind_chips else 0,
        "won": gs.get("won", False),
        "blind_states": blind_states,
        "blind_choices": blind_choices,
        # Economy
        "dollars": gs.get("dollars", 0),
        "hands_left": cr.get("hands_left", 0),
        "discards_left": cr.get("discards_left", 0),
        "reroll_cost": cr.get("reroll_cost", 5),
        "free_rerolls": cr.get("free_rerolls", 0),
        # Card areas
        "hand": [_serialize_card(c) for c in gs.get("hand", [])],
        "jokers": [_serialize_card(c) for c in gs.get("jokers", [])],
        "consumables": [_serialize_card(c) for c in gs.get("consumables", [])],
        "shop_cards": [_serialize_card(c) for c in gs.get("shop_cards", [])],
        "shop_vouchers": [_serialize_card(c) for c in gs.get("shop_vouchers", [])],
        "shop_boosters": [_serialize_card(c) for c in gs.get("shop_boosters", [])],
        "pack_cards": [_serialize_card(c) for c in gs.get("pack_cards", [])],
        "pack_choices_remaining": gs.get("pack_choices_remaining", 0),
        "pack_type": gs.get("pack_type", ""),
        # Last hand
        "last_hand": {
            "name": last_hand.get("handname", ""),
            "chips": last_hand.get("chips", 0),
            "mult": last_hand.get("mult", 0),
        } if last_hand else None,
        "last_score": last_score,
        "last_action": _action_label(last_action) if last_action else None,
        # Legal actions
        "legal_actions": [_serialize_action(a) for a in (legal_actions or [])],
        # History log
        "history": (history or [])[-30:],
        # Stats
        "deck_remaining": len(gs.get("deck", [])),
        "discard_pile": len(gs.get("discard_pile", [])),
    }


# ---------------------------------------------------------------------------
# GameObserver
# ---------------------------------------------------------------------------


class GameObserver:
    """Central hub that wraps the game loop and emits state to renderers.

    Parameters
    ----------
    use_terminal:
        Enable Rich terminal output.
    use_web:
        Enable browser dashboard (FastAPI + WebSocket).
    speed:
        Seconds between steps in watch mode (0 = as fast as possible).
    port:
        Port for the web dashboard server.
    """

    def __init__(
        self,
        *,
        use_terminal: bool = True,
        use_web: bool = True,
        speed: float = 0.8,
        port: int = 8765,
    ) -> None:
        self.use_terminal = use_terminal
        self.use_web = use_web
        self.speed = speed
        self.port = port

        self._rich: Any = None
        self._web: Any = None
        self._state_queue: queue.Queue[dict] = queue.Queue(maxsize=200)
        self._action_queue: queue.Queue[dict] = queue.Queue(maxsize=1)
        self._history: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "GameObserver":
        if self.use_terminal:
            from jackdaw.visualization.rich_renderer import RichRenderer
            self._rich = RichRenderer()

        if self.use_web:
            from jackdaw.visualization.web_server import WebServer
            self._web = WebServer(self._state_queue, self._action_queue, port=self.port)
            self._web.start()

        return self

    def stop(self) -> None:
        if self._web:
            self._web.stop()

    def __enter__(self) -> "GameObserver":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def emit(
        self,
        gs: dict[str, Any],
        action: Action | None = None,
        legal_actions: list[Action] | None = None,
        last_score: int | None = None,
    ) -> None:
        if action:
            self._history.append(_action_label(action))

        snapshot = serialize_state(
            gs,
            legal_actions=legal_actions,
            last_action=action,
            last_score=last_score,
            history=self._history,
        )

        if self._rich:
            self._rich.render(snapshot)

        if self._web:
            try:
                self._state_queue.put_nowait(snapshot)
            except queue.Full:
                # Drop oldest, put newest
                try:
                    self._state_queue.get_nowait()
                    self._state_queue.put_nowait(snapshot)
                except queue.Empty:
                    pass

    # ------------------------------------------------------------------
    # Watch mode
    # ------------------------------------------------------------------

    def simulate_watched(
        self,
        back_key: str,
        stake: int,
        seed: str,
        agent: Callable[[dict[str, Any], list[Action]], Action],
        *,
        challenge: dict[str, Any] | None = None,
        max_actions: int = 10_000,
        adapter: Any = None,
    ) -> dict[str, Any]:
        """Run simulate_run() with live rendering after every step.

        If *adapter* is provided (e.g. BridgeAdapter), use it instead of
        the in-process engine.  This enables live mode against real Balatro.
        """
        if adapter is not None:
            gs = adapter.raw_state
        else:
            gs = initialize_run(back_key, stake, seed, challenge=challenge)
        self.emit(gs, legal_actions=get_legal_actions(gs))

        actions_taken = 0
        while actions_taken < max_actions:
            phase = gs.get("phase")
            if phase == GamePhase.GAME_OVER:
                break
            if gs.get("won") and phase == GamePhase.SHOP:
                break

            legal = get_legal_actions(gs)
            if not legal:
                break

            action = agent(gs, legal)
            try:
                if adapter is not None:
                    adapter.step(action)
                    gs = adapter.raw_state
                else:
                    step(gs, action)
            except IllegalActionError:
                break

            actions_taken += 1
            self.emit(gs, action=action, legal_actions=get_legal_actions(gs))

            if self.speed > 0:
                time.sleep(self.speed)

        gs["actions_taken"] = actions_taken
        self.emit(gs)
        return gs

    # ------------------------------------------------------------------
    # Play mode
    # ------------------------------------------------------------------

    def simulate_play(
        self,
        back_key: str,
        stake: int,
        seed: str,
        *,
        challenge: dict[str, Any] | None = None,
        max_actions: int = 10_000,
        adapter: Any = None,
    ) -> dict[str, Any]:
        """Human-driven mode: game waits for the browser to send each action.

        If *adapter* is provided (e.g. BridgeAdapter), use it instead of
        the in-process engine.  This enables live mode against real Balatro.
        """
        if not self.use_web:
            raise RuntimeError("Play mode requires --web to be enabled.")

        if adapter is not None:
            gs = adapter.raw_state
        else:
            gs = initialize_run(back_key, stake, seed, challenge=challenge)

        actions_taken = 0
        while actions_taken < max_actions:
            phase = gs.get("phase")
            if phase == GamePhase.GAME_OVER:
                break
            if gs.get("won") and phase == GamePhase.SHOP:
                break

            legal = get_legal_actions(gs)
            if not legal:
                break

            self.emit(gs, legal_actions=legal)

            # Block until browser sends an action
            action_data = self._action_queue.get()
            try:
                action = deserialize_action(action_data)
                if adapter is not None:
                    adapter.step(action)
                    gs = adapter.raw_state
                else:
                    step(gs, action)
                actions_taken += 1
            except (ValueError, IllegalActionError):
                # Re-emit same state so browser can try again
                continue

        gs["actions_taken"] = actions_taken
        self.emit(gs)
        return gs
