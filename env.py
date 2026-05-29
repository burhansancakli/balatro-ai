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
import requests
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple

from strategy import (
    Strategy,
    NUM_STRATEGIES,
    pick_best_play,
    parse_cards_from_gamestate,
    parse_jokers_from_gamestate,
    strategy_coherence_reward,
)
from observations import gamestate_to_observation, OBS_SIZE
from jokers import JOKERS_BY_STRATEGY


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

DECK            = "RED"
STAKE           = "WHITE"
SEED            = "TRAIN01"
MAX_STEPS       = 500
RPC_TIMEOUT     = 60       # generous timeout for slow transitions
RPC_RETRIES     = 3        # retry count on timeout
RPC_RETRY_WAIT  = 0.5     # seconds between retries
POLL_INTERVAL   = 0.02      # seconds between state polls
POLL_TIMEOUT    = 30.0     # max seconds to wait for a state transition

SURVIVAL_REWARD       = 0.05
PROGRESS_REWARD_SCALE = 0.02
CASH_OUT_SETTLE_WAIT  = 0.05


# ─────────────────────────────────────────────────────────────
# TRANSITIONAL STATES
# States the game passes through briefly — we just poll past them
# ─────────────────────────────────────────────────────────────
TRANSITIONAL_STATES = {
    "HAND_PLAYED",
    "DRAW_TO_HAND",
    "PLAY_ANIM",
    "SCORING",
    "SCORED",
    "ROUND_TRANSITION",
}


# ─────────────────────────────────────────────────────────────
# RPC CLIENT
# ─────────────────────────────────────────────────────────────

class BalatrobotClient:
    """Thin JSON-RPC client with retry logic for one Balatrobot instance."""

    def __init__(self, port: int):
        self.url  = f"http://127.0.0.1:{port}"
        self.port = port

    def call(self, method: str, params: dict = {}) -> dict:
        """
        Call a Balatrobot RPC method with automatic retry on timeout.
        Raises RuntimeError if all retries fail.
        """
        last_error = None
        for attempt in range(RPC_RETRIES):
            try:
                response = requests.post(
                    self.url,
                    json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
                    timeout=RPC_TIMEOUT,
                )
                data = response.json()
                if "error" in data:
                    raise RuntimeError(f"[RPC:{method}] {data['error']['message']}")
                return data["result"]

            except requests.exceptions.ReadTimeout as e:
                last_error = e
                if attempt < RPC_RETRIES - 1:
                    time.sleep(RPC_RETRY_WAIT)
                    continue

            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < RPC_RETRIES - 1:
                    time.sleep(RPC_RETRY_WAIT)
                    continue

        raise RuntimeError(
            f"[port {self.port}] RPC '{method}' failed after {RPC_RETRIES} attempts: {last_error}"
        )

    def health(self) -> bool:
        try:
            return self.call("health").get("status") == "ok"
        except Exception:
            return False

    def poll_until(self, target_states: list, timeout: float = POLL_TIMEOUT) -> dict:
        """
        Poll gamestate until one of the target states is reached.
        Automatically skips through known transitional states.
        """
        deadline = time.time() + timeout
        last_state = None

        while time.time() < deadline:
            try:
                state = self.call("gamestate")
                current = state.get("state", "")
                last_state = current

                if current in target_states:
                    return state

                # If in a known transitional state just keep polling
                if current in TRANSITIONAL_STATES:
                    time.sleep(POLL_INTERVAL)
                    continue

                # Unknown state — poll but warn
                time.sleep(POLL_INTERVAL)

            except Exception:
                time.sleep(POLL_INTERVAL)

        raise TimeoutError(
            f"[port {self.port}] Timeout ({timeout}s) waiting for {target_states}. "
            f"Last state: {last_state}"
        )


# ─────────────────────────────────────────────────────────────
# BALATROENV
# ─────────────────────────────────────────────────────────────

class BalatroEnv(gym.Env):
    """
    Gymnasium environment wrapping Balatrobot.

    Observation space:
        Flat float32 vector of size OBS_SIZE.
        Includes: ante, round, money, blind target, hand cards, jokers.

    Action space:
        Discrete(NUM_STRATEGIES)
        0 = FLUSH_BUILD, 1 = PAIR_BUILD, 2 = MULT_BUILD

    The strategy is chosen ONCE PER ANTE by the RL agent.
    Within the ante, the calculator executes all plays automatically.
    Control returns to the agent at the start of each new ante.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        port: int = 12346,
        save_path: str = Path.cwd() / "fresh_balatro.jkr",
        seed: str = SEED,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.port      = port
        self.save_path = save_path
        self.seed      = seed
        self.client    = BalatrobotClient(port)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_STRATEGIES)

        self._current_strategy: Optional[Strategy] = None
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
        self.client.call("load", {"path": self.save_path})

        # Wait for a playable state — mod auto-handles BLIND_SELECT
        # but we accept it too as a fallback
        state = self.client.poll_until([
            "BLIND_SELECT", "SELECTING_HAND", "SHOP"
        ])

        # If mod didn't auto-select blind, do it here as fallback
        if state.get("state") == "BLIND_SELECT":
            try:
                state = self.client.call("select")
            except Exception:
                state = self.client.poll_until(["SELECTING_HAND", "SHOP"])

        self._current_strategy = None
        self._current_ante     = state.get("ante_num", 1)
        self._steps            = 0
        self._episode_reward   = 0.0
        self._last_gamestate   = state
        self._jokers_bought_episode = 0

        return gamestate_to_observation(state), {
            "state": state.get("state"),
            "ante": self._current_ante,
            "ante_reached": self._current_ante,
            "jokers_bought": self._jokers_bought_episode,
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        strategy = Strategy(action)
        self._current_strategy = strategy
        self._steps += 1

        total_reward, gamestate, done, hand_logs = self._play_ante(strategy)

        self._episode_reward += total_reward
        self._last_gamestate  = gamestate
        obs = gamestate_to_observation(gamestate)

        terminated = done
        truncated  = self._steps >= MAX_STEPS

        info = {
            "strategy":       strategy.name,
            "ante":           gamestate.get("ante_num", 0),
            "ante_reached":   gamestate.get("ante_num", 0),
            "round":          gamestate.get("round_num", 0),
            "won":            gamestate.get("won", False),
            "jokers_bought":  self._jokers_bought_episode,
            "episode_reward": self._episode_reward,
            "hand_logs":      hand_logs,
        }

        return obs, total_reward, terminated, truncated, info

    def close(self):
        pass

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
                try:
                    state = self.client.call("select")
                except Exception:
                    state = self.client.poll_until(["SELECTING_HAND"])

            # ── Play a hand ─────────────────────────────────────
            elif current == "SELECTING_HAND":
                state, reward, hand_log = self._play_hand(state, strategy)
                total_reward += reward
                if hand_log:
                    hand_logs.append(hand_log)

            # ── Cash out after round ────────────────────────────
            elif current == "ROUND_EVAL":
                # Poll briefly to let animations settle, then cash out
                time.sleep(CASH_OUT_SETTLE_WAIT)
                try:
                    state = self.client.call("cash_out")
                except Exception:
                    # If cash_out times out, poll until next state
                    state = self.client.poll_until([
                        "SHOP", "BLIND_SELECT", "GAME_OVER", "SELECTING_HAND"
                    ])
                if state.get("state") != "GAME_OVER":
                    total_reward += SURVIVAL_REWARD

            # ── Shop — ante complete ────────────────────────────
            elif current == "SHOP":
                state = self._handle_shop(state, strategy)
                total_reward += self._outcome_reward(state)
                # Update current ante tracker
                self._current_ante = state.get("ante_num", self._current_ante)
                return total_reward, state, False, hand_logs

            # ── Booster pack (mod should prevent this) ──────────
            elif current == "SMODS_BOOSTER_OPENED":
                try:
                    state = self.client.call("pack", {"skip": True})
                except Exception:
                    state = self.client.poll_until(["SHOP", "BLIND_SELECT"])

            # ── Transitional / unknown state ────────────────────
            else:
                time.sleep(POLL_INTERVAL)
                try:
                    state = self.client.call("gamestate")
                except Exception:
                    time.sleep(0.5)

        # Safety: exceeded loop limit
        return total_reward, state, True, hand_logs

    def _play_hand(self, state: dict, strategy: Strategy) -> Tuple[dict, float, str]:
        """Use calculator to pick and play the best hand."""
        hand_cards = parse_cards_from_gamestate(state)
        joker_labels = parse_jokers_from_gamestate(state)

        if not hand_cards:
            try:
                state = self.client.call("gamestate")
            except Exception:
                pass
            return state, 0.0, ""

        indices, hand_type, _ = pick_best_play(hand_cards, strategy)
        played_cards = [f"{hand_cards[i]['rank']}{hand_cards[i]['suit'][0].upper()}" for i in indices]
        hand_summary = (
            f"[seed {self.seed}] ante={state.get('ante_num','?')} "
            f"round={state.get('round_num','?')} strategy={strategy.name} "
            f"hand={hand_type} played=[{', '.join(played_cards)}]"
        )

        try:
            new_state = self.client.call("play", {"cards": indices})
        except Exception:
            # If play times out, poll until stable state
            new_state = self.client.poll_until([
                "SELECTING_HAND", "ROUND_EVAL", "GAME_OVER"
            ])

        coherence = strategy_coherence_reward(hand_type, strategy)
        reward    = coherence * 0.1 + self._progress_reward(new_state)

        return new_state, reward, hand_summary

    def _handle_shop(self, state: dict, strategy: Strategy) -> dict:
        """Buy one affordable strategy-matching shop joker, then leave shop."""
        if self._has_joker_slot(state):
            choice = self._choose_shop_joker(state, strategy)
            if choice is not None:
                try:
                    state = self.client.call("buy", {"card": choice})
                    self._jokers_bought_episode += 1
                except Exception:
                    # Buying is opportunistic; a stale shop state should not stop the run.
                    try:
                        state = self.client.call("gamestate")
                    except Exception:
                        pass

        try:
            state = self.client.call("next_round")
        except Exception:
            state = self.client.poll_until([
                "BLIND_SELECT", "SELECTING_HAND", "GAME_OVER"
            ])
        return state

    def _choose_shop_joker(self, state: dict, strategy: Strategy) -> Optional[int]:
        """Return the shop card index for the best affordable joker, if any."""
        strategy_jokers = JOKERS_BY_STRATEGY.get(strategy.name, [])
        strategy_rank = {name: rank for rank, name in enumerate(strategy_jokers)}
        money = int(state.get("money", 0) or 0)

        shop = state.get("shop", {}) or {}
        shop_cards = shop.get("cards", []) or []
        candidates = []

        for index, card in enumerate(shop_cards):
            label = card.get("label", "")
            key = card.get("key", "")
            card_set = card.get("set", "")

            if label not in strategy_rank:
                continue
            if card_set != "JOKER" and not str(key).startswith("j_"):
                continue

            cost = self._buy_cost(card)
            if cost > money:
                continue

            candidates.append((cost, -strategy_rank[label], index))

        if not candidates:
            return None

        # Higher cost is a simple rarity/impact proxy; registry order breaks ties.
        return max(candidates)[2]

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
