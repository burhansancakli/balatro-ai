"""
observations.py — Balatro RL Observation Space
================================================
Converts raw Balatrobot gamestate JSON into a flat numpy vector.
Imports joker definitions from jokers.py — do not hardcode jokers here.

Observation layout:
  [0]     ante_num            normalized 0-1  (max ante = 8)
  [1]     round_num           normalized 0-1  (max round = 3)
  [2]     money               normalized 0-1  (clipped at $100). # Because saving more money doesn't serve any purpose.
  [3]     blind_target        normalized 0-1  (clipped at 100_000)
  [4]     chips_so_far        normalized 0-1  (clipped at 100_000)
  [5]     progress_ratio      chips / blind_target  (0-1)
  [6]     hands_left          normalized 0-1  (max = 4)
  [7]     discards_left       normalized 0-1  (max = 4)
  [8]     hand_count          normalized 0-1  (max = 8)
  [9-24]  8 card slots × (rank_idx, suit_idx)  — -1 if empty
  [25 - 25+NUM_JOKERS*2-1]  5 joker slots × (joker_idx, active)

OBS_SIZE is computed automatically from NUM_JOKERS.
Adding jokers in jokers.py automatically expands the vector.
"""

import numpy as np
from jokers import JOKER_INDEX, NUM_JOKERS

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["C", "D", "H", "S"]

SUIT_SYMBOLS = {"C": "♣", "D": "♦", "H": "♥", "S": "♠"}
RANK_LABELS  = {
    "A": "1", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9", "T": "10",
    "J": "J", "Q": "Q", "K": "K",
}

MAX_ANTE      = 8.0
MAX_ROUND     = 3.0
MAX_MONEY     = 100.0
MAX_BLIND     = 100_000.0
MAX_CHIPS     = 100_000.0
MAX_HANDS     = 4.0
MAX_DISCARDS  = 4.0
MAX_HAND_SIZE = 8.0
MAX_JOKER_SLOTS = 5

# Computed automatically — import this in env.py
OBS_SIZE = 9 + (8 * 2) + (MAX_JOKER_SLOTS * 2)  # = 35 for 5 joker slots
# ante, round, money, blind_target, chips, progress, hands_left, discards_left, hand_count. + 8 cards with rank and suit. + 5 jokers with idx and active.


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _clip_norm(value: float, max_val: float) -> float:
    if max_val == 0:
        return 0.0
    return float(np.clip(value / max_val, 0.0, 1.0))


def _get_blind_target(raw_state: dict) -> float:
    """
    Get active blind score target.
    Checks SELECT first then falls back to small blind score.
    """
    blinds = raw_state.get("blinds", {}) or {}
    for status in ["CURRENT", "SELECT"]:
        for blind_type in ["small", "big", "boss"]:
            blind = blinds.get(blind_type, {}) or {}
            if blind.get("status") == status:
                return float(blind.get("score", 0))
    small = blinds.get("small", {}) or {}
    return float(small.get("score", 0))


def card_to_string(card: dict) -> str:
    value  = card.get("value", {}) or {}
    rank   = value.get("rank", "?")
    suit   = value.get("suit", "?")
    state  = card.get("state", {}) or {}
    hidden = isinstance(state, dict) and bool(state.get("hidden", False))
    if hidden:
        return "??"
    return f"{RANK_LABELS.get(rank, rank)}{SUIT_SYMBOLS.get(suit, suit)}"


# ─────────────────────────────────────────────────────────────
# MAIN OBSERVATION FUNCTION
# ─────────────────────────────────────────────────────────────

def gamestate_to_observation(raw_state: dict) -> np.ndarray:
    """
    Convert raw Balatrobot gamestate into a flat float32 numpy vector.
    Shape: (OBS_SIZE,). All values normalized to approximately [0, 1].
    Empty slots encoded as -1.0.
    """
    obs = np.zeros(OBS_SIZE, dtype=np.float32)

    # ── Scalars [0-8] ─────────────────────────────────────────
    blind_target = _get_blind_target(raw_state)
    round_info   = raw_state.get("round", {}) or {}
    chips        = float(round_info.get("chips", 0))
    hand         = raw_state.get("hand", {}) or {}
    hand_cards   = hand.get("cards", []) or []

    obs[0] = _clip_norm(raw_state.get("ante_num",  0), MAX_ANTE)
    obs[1] = _clip_norm(raw_state.get("round_num", 0), MAX_ROUND)
    # Money is downweighted by 50% after ante 3, because it becomes less relevant to save money for future rounds.
    obs[2] = _clip_norm(raw_state.get("money",     0), MAX_MONEY) * 0.5 if raw_state.get("ante_num", 0) >= 3 else 1.0
    obs[3] = _clip_norm(blind_target,                  MAX_BLIND)
    obs[4] = _clip_norm(chips,                         MAX_CHIPS)
    obs[5] = float(np.clip(chips / blind_target, 0.0, 1.0)) if blind_target > 0 else 0.0
    obs[6] = _clip_norm(round_info.get("hands_left",    0), MAX_HANDS)
    obs[7] = _clip_norm(round_info.get("discards_left", 0), MAX_DISCARDS)
    obs[8] = _clip_norm(len(hand_cards),                    MAX_HAND_SIZE)

    # ── Card slots [9-24] ─────────────────────────────────────
    for i in range(8):
        base = 9 + i * 2
        if i < len(hand_cards):
            card     = hand_cards[i]
            value    = card.get("value", {}) or {}
            rank     = value.get("rank", "")
            suit     = value.get("suit", "")
            rank_idx = RANKS.index(rank) if rank in RANKS else -1
            suit_idx = SUITS.index(suit) if suit in SUITS else -1
            obs[base]     = float(rank_idx) / 12.0 if rank_idx >= 0 else -1.0
            obs[base + 1] = float(suit_idx) / 3.0  if suit_idx >= 0 else -1.0
        else:
            obs[base]     = -1.0
            obs[base + 1] = -1.0

    # ── Joker slots [25-34] ───────────────────────────────────
    jokers      = raw_state.get("jokers", {}) or {}
    joker_cards = jokers.get("cards", []) or []

    for i in range(MAX_JOKER_SLOTS):
        base = 25 + i * 2
        if i < len(joker_cards):
            label     = joker_cards[i].get("label", "")
            joker_idx = JOKER_INDEX.get(label, -1)
            obs[base]     = float(joker_idx) / max(NUM_JOKERS - 1, 1) if joker_idx >= 0 else -1.0
            obs[base + 1] = 1.0
        else:
            obs[base]     = -1.0
            obs[base + 1] = 0.0

    return obs


# ─────────────────────────────────────────────────────────────
# DEBUG
# ─────────────────────────────────────────────────────────────

def pretty_print_gamestate(raw_state: dict) -> None:
    hand       = raw_state.get("hand", {}) or {}
    hand_cards = hand.get("cards", []) or []
    hand_str   = "[" + ", ".join(card_to_string(c) for c in hand_cards) + "]"
    jokers      = raw_state.get("jokers", {}) or {}
    joker_cards = jokers.get("cards", []) or []
    joker_str   = "[" + ", ".join(c.get("label", "?") for c in joker_cards) + "]"
    round_info   = raw_state.get("round", {}) or {}
    blind_target = _get_blind_target(raw_state)
    chips        = int(round_info.get("chips", 0))
    print(f"Hand    : {hand_str}")
    print(f"Jokers  : {joker_str}")
    print(f"Score   : {chips} / {int(blind_target)}")
    print(f"Hands   : {round_info.get('hands_left', 0)}  |  Discards: {round_info.get('discards_left', 0)}")
    print(f"Ante    : {raw_state.get('ante_num', 0)}  |  Round: {raw_state.get('round_num', 0)}  |  Money: ${raw_state.get('money', 0)}")


def pretty_print_observation(obs: np.ndarray) -> None:
    labels = (
        ["ante", "round", "money", "blind_target", "chips",
         "progress", "hands_left", "discards_left", "hand_count"]
        + [f"card{i//2}_{'rank' if i%2==0 else 'suit'}" for i in range(16)]
        + [f"joker{i//2}_{'idx' if i%2==0 else 'active'}" for i in range(MAX_JOKER_SLOTS * 2)]
    )
    print(f"Observation vector (size={OBS_SIZE}, jokers={NUM_JOKERS}):")
    for i, (label, val) in enumerate(zip(labels, obs)):
        print(f"  [{i:02d}] {label:<25} {val:>8.4f}")
