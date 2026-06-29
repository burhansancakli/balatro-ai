"""
env.py — BalatroEnv Gymnasium wrapper
======================================
Single environment instance that wraps Balatrobot on one port.
Designed to be instantiated multiple times on different ports
for parallel training via stable-baselines3 SubprocVecEnv.
"""

import time
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple

from config import (
    SEED, DECK, STAKE, MAX_STEPS,
    POLL_INTERVAL, POLL_TIMEOUT,
    SURVIVAL_REWARD, PROGRESS_REWARD_SCALE, CASH_OUT_SETTLE_WAIT,
)
from emulator.bridge import LiveBackend, SimBackend
from strategy import (
    Strategy,
    pick_best_action,
    parse_cards_from_gamestate,
    parse_jokers_from_gamestate,
    strategy_coherence_reward,
)
from observations import gamestate_to_observation, OBS_SIZE, MAX_SHOP_SLOTS, MAX_JOKER_SLOTS



# ─────────────────────────────────────────────────────────────
# BALATROENV
# ─────────────────────────────────────────────────────────────

class BalatroEnv(gym.Env):
    """
    Gymnasium environment wrapping Balatrobot.

    Observation space:
        Flat float32 vector of size OBS_SIZE.
        Includes: ante, round, money, blind target, jokers, shop cards.

    Action space:
        Discrete(MAX_SHOP_SLOTS + MAX_JOKER_SLOTS + 1)
        0     = skip (don't do anything)
        1-4   = buy shop card at index 0-3
        5-9   = sell owned joker at index 0-4

    Strategy is fixed to MULT_BUILD.
    The RL agent decides which shop joker to buy (or skip) and
    which owned joker to sell.
    After the shop decision, the calculator plays through hands
    automatically until the next shop.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        port: int = 12346,
        save_path: str = Path.cwd() / "fresh_balatro.jkr",
        seed: str = SEED,
        render_mode: Optional[str] = None,
        backend=None,
    ):
        super().__init__()
        self.port      = port
        self.save_path = save_path
        self.seed      = seed
        self.backend   = backend or LiveBackend(port=port)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(MAX_SHOP_SLOTS + MAX_JOKER_SLOTS + 1)  # 0=skip, 1-4=buy, 5-9=sell

        self._current_ante: int  = 0
        self._steps: int         = 0
        self._episode_reward: float = 0.0
        self._last_gamestate: Optional[dict] = None
        self._jokers_bought_episode: int = 0

    # ─── Gymnasium interface ───────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Fast reset via load() — 6.75x faster than start()
        self._call("load", {"path": self.save_path})

        # Wait for a playable state — mod auto-handles BLIND_SELECT
        state = self._poll_until(["BLIND_SELECT", "SELECTING_HAND", "SHOP"])

        # If mod didn't auto-select blind, do it here
        if state.get("state") == "BLIND_SELECT":
            state = self._call("select")

        self._current_ante     = state.get("ante_num", 1)
        self._steps            = 0
        self._episode_reward   = 0.0
        self._last_gamestate   = state
        self._jokers_bought_episode = 0

        # If we landed in SHOP, skip to SELECTING_HAND so the first
        # step() gets a hand to play, not a buy decision on stale shop.
        if state.get("state") == "SHOP":
            state = self._call("next_round")
            if state.get("state") == "BLIND_SELECT":
                state = self._call("select")
            self._last_gamestate = state

        return gamestate_to_observation(state), {
            "state": state.get("state"),
            "ante": self._current_ante,
            "ante_reached": self._current_ante,
            "jokers_bought": self._jokers_bought_episode,
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"
        self._steps += 1

        total_reward = 0.0
        hand_logs    = []
        state        = self._last_gamestate

        # ── Phase 1: Handle shop if we're in one ─────────────
        if state.get("state") == "SHOP":
            self._execute_shop_action(state, action)
            total_reward += self._outcome_reward(state)
            
            state = self._call("next_round")
            if state.get("state") == "BLIND_SELECT":
                
                state = self._call("select")
               
            self._last_gamestate = state

            if state.get("state") == "GAME_OVER":
                total_reward += self._outcome_reward(state)
                self._episode_reward += total_reward
                obs = gamestate_to_observation(state)
                info = self._make_info(state, hand_logs, action)
                return obs, total_reward, True, False, info

        # ── Phase 2: Play hands until next shop (or game over)
        ante_reward, gamestate, done, hand_logs = self._play_ante(Strategy.MULT_BUILD)
        total_reward += ante_reward

        self._episode_reward += total_reward
        self._last_gamestate  = gamestate
        obs = gamestate_to_observation(gamestate)

        terminated = done
        truncated  = self._steps >= MAX_STEPS
        info = self._make_info(gamestate, hand_logs, action)

        return obs, total_reward, terminated, truncated, info

    def _make_info(self, gamestate: dict, hand_logs: list, action: int) -> dict:
        joker_labels = parse_jokers_from_gamestate(gamestate)
        shop = gamestate.get("shop", {}) or {}
        shop_cards = shop.get("cards", []) or []
        jokers = gamestate.get("jokers", {}) or {}
        joker_cards = jokers.get("cards", []) or []

        action_label = "skip"
        if 1 <= action <= MAX_SHOP_SLOTS and action - 1 < len(shop_cards):
            action_label = f"buy:{shop_cards[action - 1].get('label', f'card_{action-1}')}"
        elif MAX_SHOP_SLOTS + 1 <= action <= MAX_SHOP_SLOTS + MAX_JOKER_SLOTS:
            sell_idx = action - MAX_SHOP_SLOTS - 1
            if sell_idx < len(joker_cards):
                action_label = f"sell:{joker_cards[sell_idx].get('label', f'joker_{sell_idx}')}"
            else:
                action_label = f"sell:empty_{sell_idx}"

        return {
            "action":         action,
            "action_label":   action_label,
            "ante":           gamestate.get("ante_num", 0),
            "ante_reached":   gamestate.get("ante_num", 0),
            "round":          gamestate.get("round_num", 0),
            "won":            gamestate.get("won", False),
            "jokers_bought":  self._jokers_bought_episode,
            "episode_reward": self._episode_reward,
            "joker_labels":   joker_labels,
            "hand_logs":      hand_logs,
        }

    def close(self):
        pass

    # ─── Backend helpers ──────────────────────────────────────

    def _call(self, method: str, params: dict = {}) -> dict:
        """Call backend.handle() directly, with load→start translation for SimBackend."""
        if isinstance(self.backend, SimBackend) and method == "load":
            seed = Path(params.get("path", "")).stem.replace("fresh_", "")
            return self.backend.handle("start", {"deck": DECK, "stake": STAKE, "seed": seed or "DEFAULT"})
        return self.backend.handle(method, params)

    def _poll_until(self, target_states: list, timeout: float = POLL_TIMEOUT) -> dict:
        """Poll gamestate until one of the target states is reached."""
        if isinstance(self.backend, SimBackend):
            return self._call("gamestate")

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                state = self._call("gamestate")
                if state.get("state", "") in target_states:
                    return state
            except Exception:
                pass
            time.sleep(POLL_INTERVAL)

        raise TimeoutError(f"[port {self.port}] Timeout waiting for {target_states}")

    # ─── Internal: ante execution ─────────────────────────────

    def _play_ante(self, strategy: Strategy) -> Tuple[float, dict, bool, list]:
        """
        Play through one full ante (all rounds until shop).
        Returns (total_reward, final_gamestate, is_done, hand_logs).
        """
        total_reward = 0.0
        state        = self._last_gamestate
        hand_logs    = []
        max_loops    = 1000
        loop_count   = 0

        while loop_count < max_loops:
            loop_count += 1
            current = state.get("state", "")

            # ── Terminal ────────────────────────────────────────
            if current == "GAME_OVER":
                total_reward += self._outcome_reward(state)
                return total_reward, state, True, hand_logs

            # ── Blind select (fallback — mod should handle this) ─
            elif current == "BLIND_SELECT":
                state = self._call("select")

            # ── Play a hand ─────────────────────────────────────
            elif current == "SELECTING_HAND":
                state, reward, hand_log = self._play_hand(state, strategy)
                total_reward += reward
                if hand_log:
                    hand_logs.append(hand_log)

            # ── Cash out after round ────────────────────────────
            elif current == "ROUND_EVAL":
                time.sleep(CASH_OUT_SETTLE_WAIT)
                state = self._call("cash_out")
                if state.get("state") != "GAME_OVER":
                    total_reward += SURVIVAL_REWARD

            # ── Shop — return to agent for buy decision ────────
            elif current == "SHOP":
                return total_reward, state, False, hand_logs

            # ── Booster pack (mod should prevent this) ──────────
            elif current == "SMODS_BOOSTER_OPENED":
                state = self._call("pack", {"skip": True})

            # ── Transitional / unknown state ────────────────────
            else:
                time.sleep(POLL_INTERVAL)
                state = self._call("gamestate")

        # Safety: exceeded loop limit
        return total_reward, state, True, hand_logs

    def _play_hand(self, state: dict, strategy: Strategy) -> Tuple[dict, float, str]:
        """Use calculator to pick and play the best hand or discard cards."""
        hand_cards = parse_cards_from_gamestate(state)
        joker_labels = parse_jokers_from_gamestate(state)

        if not hand_cards:
            state = self._call("gamestate")
            return state, 0.0, ""

        round_info = state.get("round", {}) or {}
        discards_left = int(round_info.get("discards_left", 0) or 0)

        action_type, indices, hand_type = pick_best_action(
            hand_cards, strategy, discards_left, joker_labels=joker_labels
        )

        if action_type == "discard":
            discarded_cards = [f"{hand_cards[i]['rank']}{hand_cards[i]['suit'][0].upper()}" for i in indices]
            action_summary = (
                f"[seed {self.seed}] ante={state.get('ante_num','?')} "
                f"round={state.get('round_num','?')} strategy={strategy.name} "
                f"action=discard discarded=[{', '.join(discarded_cards)}]"
            )

            new_state = self._call("discard", {"cards": indices})
            return new_state, 0.0, action_summary
        else:
            played_cards = [f"{hand_cards[i]['rank']}{hand_cards[i]['suit'][0].upper()}" for i in indices]
            hand_summary = (
                f"[seed {self.seed}] ante={state.get('ante_num','?')} "
                f"round={state.get('round_num','?')} strategy={strategy.name} "
                f"hand={hand_type} played=[{', '.join(played_cards)}]"
            )

            new_state = self._call("play", {"cards": indices})

            coherence = strategy_coherence_reward(hand_type, strategy)
            reward    = coherence * 0.1 + self._progress_reward(new_state)

            return new_state, reward, hand_summary

    def _execute_shop_action(self, state: dict, action: int) -> float:
        """Execute the agent's shop action.
        action 0 = skip, 1-4 = buy, 5-9 = sell joker.
        """
        if action == 0:
            return 0.0  # skip

        # ── Buy actions (1-4) ──────────────────────────────────
        if 1 <= action <= MAX_SHOP_SLOTS:
            shop = state.get("shop", {}) or {}
            shop_cards = shop.get("cards", []) or []
            buy_idx = action - 1

            if buy_idx >= len(shop_cards):
                return 0.0  # invalid index

            card = shop_cards[buy_idx]
            label = card.get("label", "")
            key   = card.get("key", "")
            card_set = card.get("set", "")

            # Only allow buying actual jokers
            if card_set != "JOKER" and not str(key).startswith("j_"):
                return 0.0

            cost  = self._buy_cost(card)
            money = int(state.get("money", 0) or 0)
            if cost > money:
                return 0.0  # can't afford
            if not self._has_joker_slot(state):
                return 0.0  # no room

            self._call("buy", {"card": int(buy_idx)})
            self._jokers_bought_episode += 1
            return 0.0

        # ── Sell actions (5-9) ─────────────────────────────────
        if MAX_SHOP_SLOTS + 1 <= action <= MAX_SHOP_SLOTS + MAX_JOKER_SLOTS:
            sell_idx = action - MAX_SHOP_SLOTS - 1
            jokers = state.get("jokers", {}) or {}
            joker_cards = jokers.get("cards", []) or []

            if sell_idx >= len(joker_cards):
                return 0.0  # no joker at that slot

            label = joker_cards[sell_idx].get("label", f"joker_{sell_idx}")
            self._call("sell", {"joker": int(sell_idx)})
            return -0.05  # small penalty to discourage random selling

        return 0.0  # unknown action


    def _has_joker_slot(self, state: dict) -> bool:
        """Check whether buying another joker can fit in the joker area."""
        jokers = state.get("jokers", {}) or {}
        cards = jokers.get("cards", []) or []
        limit = int(jokers.get("limit", 5) or 5)
        return len(cards) < limit

    def _buy_cost(self, card: dict) -> int:
        """Read the Balatrobot card cost.buy field with safe fallbacks."""
        cost = card.get("cost", {}) or {}
        try:
            return int(cost.get("buy", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _progress_reward(self, state: dict) -> float:
        """Reward partial blind progress to reduce sparse early feedback."""
        target = self._blind_target(state)
        if target <= 0:
            return 0.0

        round_info = state.get("round", {}) or {}
        chips = float(round_info.get("chips", 0) or 0)
        progress = min(max(chips / target, 0.0), 1.0)
        return progress * PROGRESS_REWARD_SCALE

    def _blind_target(self, state: dict) -> float:
        """Return the active blind score target from the Balatrobot state."""
        blinds = state.get("blinds", {}) or {}
        for status in ["CURRENT", "SELECT"]:
            for blind_type in ["small", "big", "boss"]:
                blind = blinds.get(blind_type, {}) or {}
                if blind.get("status") == status:
                    return float(blind.get("score", 0) or 0)
        small = blinds.get("small", {}) or {}
        return float(small.get("score", 0) or 0)

    def _outcome_reward(self, state: dict) -> float:
        """
        Sparse outcome reward based on ante reached.

        +0.0  ante 1 baseline
        +0.2  ante 2 beaten
        +0.4  ante 3 beaten
        +0.6  ante 4 beaten  ← research minimum target
        +0.8  ante 5 beaten
        +1.0  ante 6 beaten
        +1.5  ante 7 beaten
        +2.0  ante 8 beaten / run won
        -0.5  game over at ante 1
        """
        ante = state.get("ante_num", 1)
        won  = state.get("won", False)

        if state.get("state") == "GAME_OVER" and not won:
            if ante <= 1:
                return -0.5
            return (ante - 1) * 0.2

        if won:
            return 2.0

        return max(0.0, (ante - 1) * 0.2)