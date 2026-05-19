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

from strategy import Strategy, NUM_STRATEGIES, pick_best_play, parse_cards_from_gamestate, strategy_coherence_reward
from observations import gamestate_to_observation, OBS_SIZE


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

        return gamestate_to_observation(state), {
            "state": state.get("state"),
            "ante": self._current_ante,
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
            "round":          gamestate.get("round_num", 0),
            "won":            gamestate.get("won", False),
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
                time.sleep(0.2)
                try:
                    state = self.client.call("cash_out")
                except Exception:
                    # If cash_out times out, poll until next state
                    state = self.client.poll_until([
                        "SHOP", "BLIND_SELECT", "GAME_OVER", "SELECTING_HAND"
                    ])

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
        reward    = coherence * 0.1

        return new_state, reward, hand_summary

    def _handle_shop(self, state: dict, strategy: Strategy) -> dict:
        """Skip shop and go to next round."""
        try:
            state = self.client.call("next_round")
        except Exception:
            state = self.client.poll_until([
                "BLIND_SELECT", "SELECTING_HAND", "GAME_OVER"
            ])
        return state

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
