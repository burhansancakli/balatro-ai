"""
strategy.py — Strategy definitions and low-level calculator
============================================================
The low-level executor. Given the current hand and an active
strategy label from the high-level RL agent, picks the optimal
5-card play deterministically using the Chips x Mult formula.

No RL here — this is pure math. The high-level agent decides
WHAT strategy to pursue; this module decides HOW to execute it.

Strategies:
  0 = FLUSH_BUILD   — maximize flush/straight flush hands
  1 = PAIR_BUILD    — maximize pair/two-pair/full house/four-of-a-kind
  2 = MULT_BUILD    — maximize raw chips x mult regardless of hand type
"""

from enum import IntEnum
from itertools import combinations
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────
# STRATEGY DEFINITIONS
# ─────────────────────────────────────────────────────────────

class Strategy(IntEnum):
    FLUSH_BUILD = 0
    PAIR_BUILD  = 1
    MULT_BUILD  = 2

STRATEGY_NAMES = {
    Strategy.FLUSH_BUILD: "Flush Build",
    Strategy.PAIR_BUILD:  "Pair Build",
    Strategy.MULT_BUILD:  "Mult Build",
}

NUM_STRATEGIES = len(Strategy)


# ─────────────────────────────────────────────────────────────
# BALATRO HAND SCORING (simplified, with whitelisted joker effects)
# Base: Chips x Mult as per vanilla Balatro level-1 hands
# ─────────────────────────────────────────────────────────────

# (chips, mult) for each hand type at level 1
HAND_SCORES = {
    "flush_five":    (160, 16),
    "flush_house":   (140, 14),
    "five_of_a_kind":(120, 12),
    "straight_flush":(100,  8),
    "four_of_a_kind": (60,  7),
    "full_house":     (40,  4),
    "flush":          (35,  4),
    "straight":       (30,  4),
    "three_of_a_kind":(30,  3),
    "two_pair":       (20,  2),
    "pair":           (10,  2),
    "high_card":       (5,  1),
}

RANK_VALUES = {
    "2": 2,  "3": 3,  "4": 4,  "5": 5,  "6": 6,
    "7": 7,  "8": 8,  "9": 9,  "T": 10, "J": 11,
    "Q": 12, "K": 13, "A": 14,
}


def _card_chip_value(rank: str) -> int:
    """Base chip contribution of a card when it scores."""
    return RANK_VALUES.get(rank, 0)


def _classify_hand(cards: List[dict]) -> Tuple[str, dict]:
    """Classify a played hand and return hand type plus metadata."""
    ranks = [c["rank"] for c in cards]
    suits = [c["suit"] for c in cards]
    rank_vals = sorted([RANK_VALUES.get(r, 0) for r in ranks], reverse=True)

    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    counts = sorted(rank_counts.values(), reverse=True)

    is_flush    = len(cards) == 5 and len(set(suits)) == 1
    is_straight = (len(cards) == 5 and len(set(rank_vals)) == 5 and
                   max(rank_vals) - min(rank_vals) == 4)
    # Ace-low straight
    if len(cards) == 5 and set(rank_vals) == {14, 2, 3, 4, 5}:
        is_straight = True

    # Determine hand type
    if len(cards) == 5 and is_flush and counts == [5]:
        hand_type = "flush_five"
    elif len(cards) == 5 and is_flush and counts == [3, 2]:
        hand_type = "flush_house"
    elif len(cards) == 5 and counts == [5]:
        hand_type = "five_of_a_kind"
    elif is_flush and is_straight:
        hand_type = "straight_flush"
    elif len(cards) >= 4 and counts[0] == 4:
        hand_type = "four_of_a_kind"
    elif len(cards) == 5 and counts == [3, 2]:
        hand_type = "full_house"
    elif is_flush:
        hand_type = "flush"
    elif is_straight:
        hand_type = "straight"
    elif counts[0] == 3:
        hand_type = "three_of_a_kind"
    elif counts[:2] == [2, 2]:
        hand_type = "two_pair"
    elif counts[0] == 2:
        hand_type = "pair"
    else:
        hand_type = "high_card"

    metadata = {
        "is_flush": is_flush,
        "rank_counts": rank_counts,
    }
    return hand_type, metadata


def _score_hand(cards: List[dict], joker_labels: Optional[List[str]] = None) -> Tuple[str, int]:
    """
    Score a played hand using simplified Balatro Chips x Mult arithmetic.

    The 13 whitelisted joker effects are modeled deterministically. This keeps
    the low-level executor non-RL while allowing shop purchases to influence
    card choice.
    """
    hand_type, metadata = _classify_hand(cards)
    chips, mult = HAND_SCORES[hand_type]
    chips += sum(_card_chip_value(c.get("rank", "")) for c in cards)

    for label in joker_labels or []:
        chips, mult = _apply_joker(label, cards, hand_type, metadata, chips, mult, joker_labels or [])

    return hand_type, chips * mult


def _evaluate_5card_hand(cards: List[dict]) -> Tuple[str, int]:
    """Backward-compatible helper for tests and debugging."""
    return _score_hand(cards, [])


def _apply_joker(
    label: str,
    cards: List[dict],
    hand_type: str,
    metadata: dict,
    chips: int,
    mult: int,
    joker_labels: List[str],
) -> Tuple[int, int]:
    """Apply one whitelisted joker effect to the current Chips/Mult totals."""
    rank_counts = metadata["rank_counts"]
    pair_count = sum(1 for count in rank_counts.values() if count >= 2)
    has_pair = any(count >= 2 for count in rank_counts.values())
    has_three = any(count >= 3 for count in rank_counts.values())
    has_two_pair = pair_count >= 2
    has_flush = metadata["is_flush"]

    if label == "Droll Joker" and has_flush:
        mult += 10
    elif label == "Crafty Joker" and has_flush:
        chips += 80
    elif label == "Lusty Joker":
        mult += 3 * sum(1 for card in cards if card.get("suit") == "H")
    elif label == "Greedy Joker":
        mult += 3 * sum(1 for card in cards if card.get("suit") == "D")
    elif label == "Jolly Joker" and has_pair:
        mult += 8
    elif label == "Zany Joker" and has_three:
        mult += 12
    elif label == "Mad Joker" and has_two_pair:
        mult += 10
    elif label == "Sly Joker" and has_pair:
        chips += 50
    elif label == "Wily Joker" and has_three:
        chips += 100
    elif label == "Joker":
        mult += 4
    elif label == "Abstract Joker":
        mult += 3 * len(joker_labels)
    elif label == "Half Joker" and len(cards) <= 3:
        mult += 20
    elif label == "Scary Face":
        chips += 30 * sum(1 for card in cards if card.get("rank") in {"J", "Q", "K"})

    return chips, mult


# Strategy-specific hand type preferences
STRATEGY_PREFERRED_HANDS = {
    Strategy.FLUSH_BUILD: {
        "flush_five": 1000, "flush_house": 900, "straight_flush": 800,
        "flush": 700, "four_of_a_kind": 200, "full_house": 100,
        "straight": 50, "three_of_a_kind": 20, "two_pair": 10,
        "pair": 5, "high_card": 1,
    },
    Strategy.PAIR_BUILD: {
        "flush_five": 500, "five_of_a_kind": 1000, "four_of_a_kind": 900,
        "full_house": 800, "flush_house": 700, "three_of_a_kind": 600,
        "two_pair": 500, "pair": 400, "straight_flush": 200,
        "flush": 100, "straight": 50, "high_card": 1,
    },
    Strategy.MULT_BUILD: {
        # Pure score maximizer — no preference, just highest chips*mult
        "flush_five": 1, "flush_house": 1, "five_of_a_kind": 1,
        "straight_flush": 1, "four_of_a_kind": 1, "full_house": 1,
        "flush": 1, "straight": 1, "three_of_a_kind": 1,
        "two_pair": 1, "pair": 1, "high_card": 1,
    },
}


def pick_best_play(
    hand_cards: List[dict],
    strategy: Strategy,
    n_play: int = 5,
    joker_labels: Optional[List[str]] = None,
) -> Tuple[List[int], str, int]:
    """
    Given a list of card dicts and a strategy, return the best play.

    Args:
        hand_cards: list of card dicts from Balatrobot gamestate.
                    Each dict has keys: rank (str), suit (str).
        strategy:   Strategy enum value from high-level agent.
        n_play:     number of cards to play (default 5).
        joker_labels: owned joker display names from the gamestate.

    Returns:
        (card_indices, hand_type, estimated_score)
        card_indices: list of indices into hand_cards to play.
    """
    n = len(hand_cards)
    if n == 0:
        return [], "high_card", 0

    joker_labels = joker_labels or []
    max_play = min(n_play, n)
    play_sizes = [max_play]
    if "Half Joker" in joker_labels:
        play_sizes = list(range(1, max_play + 1))

    best_indices  = list(range(max_play))
    best_score    = -1
    best_hand     = "high_card"
    prefs         = STRATEGY_PREFERRED_HANDS[strategy]

    for size in play_sizes:
        for combo in combinations(range(n), size):
            cards = [hand_cards[i] for i in combo]
            hand_type, base_score = _score_hand(cards, joker_labels)
            preference = prefs.get(hand_type, 1)

            if strategy == Strategy.MULT_BUILD:
                # Pure score maximizer
                weighted_score = base_score
            else:
                # Blend: heavily weight strategy preference, lightly weight score
                weighted_score = preference * 10000 + base_score

            if weighted_score > best_score:
                best_score   = weighted_score
                best_indices = list(combo)
                best_hand    = hand_type

    return best_indices, best_hand, best_score


def parse_cards_from_gamestate(gamestate: dict) -> List[dict]:
    """Extract a clean list of {rank, suit} dicts from raw gamestate."""
    hand = gamestate.get("hand", {}) or {}
    raw_cards = hand.get("cards", []) or []
    cards = []
    for c in raw_cards:
        value = c.get("value", {}) or {}
        rank  = value.get("rank", "")
        suit  = value.get("suit", "")
        if rank and suit:
            cards.append({"rank": rank, "suit": suit})
    return cards


def parse_jokers_from_gamestate(gamestate: dict) -> List[str]:
    """Extract owned joker display names from raw gamestate."""
    jokers = gamestate.get("jokers", {}) or {}
    raw_cards = jokers.get("cards", []) or []
    labels = []
    for card in raw_cards:
        label = card.get("label", "")
        if label:
            labels.append(label)
    return labels


def strategy_coherence_reward(
    hand_type: str,
    strategy: Strategy,
) -> float:
    """
    Shaped reward for the high-level agent: how well does the
    played hand type match the chosen strategy?
    Returns a value in [0.0, 1.0].
    """
    prefs = STRATEGY_PREFERRED_HANDS[strategy]
    max_pref = max(prefs.values())
    pref = prefs.get(hand_type, 1)
    return pref / max_pref
