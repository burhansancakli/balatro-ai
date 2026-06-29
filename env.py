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


try:
    from emulator.env.balatro_env import BalatroEnvironment
    from emulator.env.game_interface import DirectAdapter
    JACKDAW_AVAILABLE = True
except Exception:
    JACKDAW_AVAILABLE = False


from strategy import (
    Strategy,
    pick_best_play,
    parse_cards_from_gamestate,
    parse_jokers_from_gamestate,
    strategy_coherence_reward,
    pick_best_action,
)
from observations import gamestate_to_observation, OBS_SIZE, MAX_SHOP_SLOTS, MAX_JOKER_SLOTS



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
        emulator: bool = False,
    ):
        super().__init__()
        self.port      = port
        self.save_path = save_path
        self.seed      = seed
        self.emulator = emulator
        
        if emulator:
            if not JACKDAW_AVAILABLE:
                raise RuntimeError("Jackdaw emulator not installed")

            self.sim = BalatroEnvironment(
                adapter_factory=DirectAdapter
            )
        else:
            self.client = BalatrobotClient(port)


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

        if self.emulator:
            # The emulator returns 3 values (obs, mask, info), not 2!
            try:
                obs, mask, info = self.sim.reset(seed=seed)
            except TypeError:
                obs, mask, info = self.sim.reset()
            
            # Access the raw state directly from the adapter
            state = self.sim._adapter.raw_state
            self.last_state = state
            
            obs_vector = gamestate_to_observation(state)
            return np.asarray(obs_vector, dtype=np.float32), info

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

        self._current_ante     = state.get("ante_num", 1)
        self._steps            = 0
        self._episode_reward   = 0.0
        self._last_gamestate   = state
        self._jokers_bought_episode = 0

        # If we landed in SHOP, skip to SELECTING_HAND so the first
        # step() gets a hand to play, not a buy decision on stale shop.
        if state.get("state") == "SHOP":
            try:
                state = self.client.call("next_round")
            except Exception:
                state = self.client.poll_until(["BLIND_SELECT", "SELECTING_HAND", "SHOP"])
            if state.get("state") == "BLIND_SELECT":
                try:
                    state = self.client.call("select")
                except Exception:
                    state = self.client.poll_until(["SELECTING_HAND", "SHOP"])
            self._last_gamestate = state

        return gamestate_to_observation(state), {
            "state": state.get("state"),
            "ante": self._current_ante,
            "ante_reached": self._current_ante,
            "jokers_bought": self._jokers_bought_episode,
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # --- SAFEGUARD: Convert NumPy array scalar / list action into a standard Python int ---
        if hasattr(action, "item"):
            action = int(action.item())
        elif isinstance(action, (list, np.ndarray)):
            action = int(action[0])
        else:
            action = int(action)
        
        if self.emulator:
            from emulator.engine.actions import PlayHand, Discard, SelectBlind, CashOut, NextRound, SkipPack
            strategy = Strategy(action)
            info = {}

            # Talk directly to the underlying DirectAdapter to avoid FactoredAction crashes
            while True:
                state = self.sim._adapter.raw_state
                self.last_state = state

                terminated = self.sim._adapter.done
                truncated = self.sim._step_count >= self.sim._max_steps
                won = self.sim._adapter.won

                # Terminal checks
                if terminated or truncated:
                    info["game_over"] = terminated
                    info["won"] = won
                    info["ante"] = self.sim.episode_ante
                    info["ante_reached"] = self.sim.episode_ante
                    reward = self._jackdaw_reward(info)
                    return np.asarray(gamestate_to_observation(state), dtype=np.float32), reward, terminated, truncated, info

                # DirectAdapter natively provides legal actions!
                legal_actions = self.sim._adapter.get_legal_actions()
                chosen_action = None

                # Automatically handle administrative selection phases
                for a in legal_actions:
                    if isinstance(a, (SelectBlind, CashOut, NextRound, SkipPack)):
                        chosen_action = a
                        break

                # Map strategy to cards
                if chosen_action is None:
                    hand_cards = parse_cards_from_gamestate(state)
                    jokers = parse_jokers_from_gamestate(state)
                    indices, hand_type, _ = pick_best_play(
                        hand_cards=hand_cards, 
                        strategy=strategy, 
                        joker_labels=jokers
                    )

                    if not indices and hand_cards:
                        indices = [0]

                    # jsut like for real game
                    chosen_action = PlayHand(card_indices=tuple(sorted(indices)))

                if chosen_action is None and legal_actions:
                    chosen_action = legal_actions[0]
                elif chosen_action is None:
                    break

                # Step the adapter directly using low-level Action objects
                self.sim._adapter.step(chosen_action)
                self.sim._step_count += 1
                
                next_state = self.sim._adapter.raw_state

                # Return control to PPO when entering a new selection phase
                phase = next_state.get("phase")
                if phase in ["SHOP", "BLIND_SELECT", "MAIN_MENU"] or self.sim._adapter.done:
                    info["game_over"] = self.sim._adapter.done
                    info["won"] = self.sim._adapter.won
                    info["ante"] = next_state.get("round_resets", {}).get("ante", 1)
                    info["ante_reached"] = info["ante"]
                    reward = self._jackdaw_reward(info)
                    return np.asarray(gamestate_to_observation(next_state), dtype=np.float32), reward, self.sim._adapter.done, truncated, info

        assert self.action_space.contains(action), f"Invalid action: {action}"
        self._steps += 1

        total_reward = 0.0
        hand_logs    = []
        state        = self._last_gamestate

        # ── Phase 1: Handle shop if we're in one ─────────────
        if state.get("state") == "SHOP":
            self._execute_shop_action(state, action)
            total_reward += self._outcome_reward(state)
            try:
                state = self.client.call("next_round")
            except Exception:
                state = self.client.poll_until([
                    "BLIND_SELECT", "SELECTING_HAND", "GAME_OVER"
                ])
            if state.get("state") == "BLIND_SELECT":
                try:
                    state = self.client.call("select")
                except Exception:
                    state = self.client.poll_until(["SELECTING_HAND", "GAME_OVER"])
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

            # ── Shop — return to agent for buy decision ────────
            elif current == "SHOP":
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
        """Use calculator to pick and play the best hand or discard cards."""
        hand_cards = parse_cards_from_gamestate(state)
        joker_labels = parse_jokers_from_gamestate(state)

        if not hand_cards:
            try:
                state = self.client.call("gamestate")
            except Exception:
                pass
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

            try:
                new_state = self.client.call("discard", {"cards": indices})
            except Exception:
                # If discard times out, poll until stable state
                new_state = self.client.poll_until([
                    "SELECTING_HAND", "ROUND_EVAL", "GAME_OVER"
                ])

            return new_state, 0.0, action_summary
        else:
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

            try:
                self.client.call("buy", {"card": int(buy_idx)})
                self._jokers_bought_episode += 1
                return 0.0 
            except Exception as e:
                print(f"[port {self.port}] Buy failed for '{label}' at index {buy_idx}: {e}")
                return 0.0

        # ── Sell actions (5-9) ─────────────────────────────────
        if MAX_SHOP_SLOTS + 1 <= action <= MAX_SHOP_SLOTS + MAX_JOKER_SLOTS:
            sell_idx = action - MAX_SHOP_SLOTS - 1
            jokers = state.get("jokers", {}) or {}
            joker_cards = jokers.get("cards", []) or []

            if sell_idx >= len(joker_cards):
                return 0.0  # no joker at that slot

            label = joker_cards[sell_idx].get("label", f"joker_{sell_idx}")
            try:
                self.client.call("sell", {"joker": int(sell_idx)})
                return -0.05  # small penalty to discourage random selling
            except Exception as e:
                print(f"[port {self.port}] Sell failed for '{label}' at index {sell_idx}: {e}")
                return 0.0

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
    

    def _jackdaw_reward(self, info: dict) -> float:
        """
        Calculates the reward for emulator steps. 
        Mirrors the sparse outcome reward logic from the actual game wrapper.
        """
        # Align these keys with whatever your Jackdaw sim actually outputs in `info`
        ante = info.get("ante_reached", info.get("ante", 1))
        won = info.get("won", False)
        game_over = info.get("game_over", False) or info.get("terminated", False)
        
        # If Jackdaw's wrapper natively calculates a step reward, grab it
        base_reward = info.get("reward", 0.0)

        # Terminal state rewards
        if game_over and not won:
            if ante <= 1:
                return base_reward - 0.5
            return base_reward + (ante - 1) * 0.2

        if won:
            return base_reward + 2.0

        # Ongoing step reward (optional: add progress logic if Jackdaw supports it)
        return base_reward
