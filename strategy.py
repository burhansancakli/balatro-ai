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

import numpy as np


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


# ─────────────────────────────────────────────────────────────
# FAST INTEGER-BASED HAND CLASSIFICATION (numpy)
# ─────────────────────────────────────────────────────────────

_RANK_ORDER = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
_RANK_TO_INT = {r: i for i, r in enumerate(_RANK_ORDER)}
_SUIT_TO_INT = {"C": 0, "D": 1, "H": 2, "S": 3}

# Valid 5-card straight patterns (set of rank indices)
_STRAIGHT_SETS = (
    frozenset(range(0, 5)),    # 2-6
    frozenset(range(1, 6)),    # 3-7
    frozenset(range(2, 7)),    # 4-8
    frozenset(range(3, 8)),    # 5-9
    frozenset(range(4, 9)),    # 6-T
    frozenset(range(5, 10)),   # 7-J
    frozenset(range(6, 11)),   # 8-Q
    frozenset(range(7, 12)),   # 9-K
    frozenset(range(8, 13)),   # T-A
    frozenset((12, 0, 1, 2, 3)),  # A-5 (ace-low)
)


def _cards_to_ints(hand_cards: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert card dicts to rank/suit integer arrays once."""
    n = len(hand_cards)
    ranks = np.empty(n, dtype=np.intp)
    suits = np.empty(n, dtype=np.intp)
    for i, c in enumerate(hand_cards):
        ranks[i] = _RANK_TO_INT.get(c["rank"], -1)
        suits[i] = _SUIT_TO_INT.get(c["suit"], -1)
    return ranks, suits


def _classify_ints(ranks: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rank counts from pre-computed rank integer array.

    Returns ``(counts, sorted_nonzero_counts)`` where ``counts`` is a
    length-13 array of rank frequencies and ``sorted_nonzero_counts``
    is the non-zero counts sorted descending.
    """
    # Flush: n==5 and all suits equal — checked via rank array slice in caller
    # (suits checked externally for speed; is_flush passed in)

    # Rank counts via bincount — no Python dict allocation
    counts = np.bincount(np.maximum(ranks, 0), minlength=13)
    nonzero = counts[counts > 0]
    sorted_counts = np.sort(nonzero)[::-1]
    return counts, sorted_counts


def _is_flush(suits: np.ndarray) -> bool:
    return len(suits) == 5 and bool(suits[0] == suits[1] == suits[2] == suits[3] == suits[4])


def _is_straight_fast(rank_set: frozenset) -> bool:
    return rank_set in _STRAIGHT_SETS


def _hand_type_from_counts(sorted_counts: np.ndarray, is_flush: bool, is_straight: bool, n: int) -> str:
    """Determine hand type from pre-sorted count array — pure integer comparison."""
    if n == 5 and is_flush:
        if sorted_counts[0] == 5:
            return "flush_five"
        if sorted_counts[0] == 3 and len(sorted_counts) == 2:
            return "flush_house"

    if n == 5 and sorted_counts[0] == 5:
        return "five_of_a_kind"

    if is_flush and is_straight:
        return "straight_flush"

    if n >= 4 and sorted_counts[0] == 4:
        return "four_of_a_kind"

    if n == 5 and sorted_counts[0] == 3 and len(sorted_counts) == 2:
        return "full_house"

    if is_flush:
        return "flush"
    if is_straight:
        return "straight"

    if sorted_counts[0] == 3:
        return "three_of_a_kind"
    if len(sorted_counts) >= 2 and sorted_counts[0] == 2 and sorted_counts[1] == 2:
        return "two_pair"
    if sorted_counts[0] == 2:
        return "pair"
    return "high_card"


def _apply_jokers_ints(
    label: str,
    sub_ranks: np.ndarray,
    sub_suits: np.ndarray,
    n: int,
    hand_type: str,
    is_flush: bool,
    counts: np.ndarray,
    chips: int,
    mult: int,
    n_jokers: int,
) -> Tuple[int, int]:
    """Apply one joker effect using integer arrays — avoids dict lookups."""
    has_pair = bool(np.any(counts >= 2))
    has_three = bool(np.any(counts >= 3))
    has_two_pair = bool(np.sum(counts >= 2) >= 2)

    if label == "Droll Joker" and is_flush:
        mult += 10
    elif label == "Crafty Joker" and is_flush:
        chips += 80
    elif label == "Lusty Joker":
        mult += 3 * int(np.sum(sub_suits == 2))  # Hearts
    elif label == "Greedy Joker":
        mult += 3 * int(np.sum(sub_suits == 1))  # Diamonds
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
        mult += 3 * n_jokers
    elif label == "Half Joker" and n <= 3:
        mult += 20
    elif label == "Scary Face":
        chips += 30 * int(np.sum(np.isin(sub_ranks, (9, 10, 11))))  # J, Q, K

    return chips, mult


def _score_hand_ints(
    all_ranks: np.ndarray,
    all_suits: np.ndarray,
    combo_idx: np.ndarray,
    joker_labels: Tuple[str, ...],
    n_jokers: int,
) -> Tuple[str, int]:
    """Score a combination using pre-computed integer arrays — zero dict allocation."""
    n = len(combo_idx)
    sub_ranks = all_ranks[combo_idx]
    sub_suits = all_suits[combo_idx]

    # Flush + straight
    fl = _is_flush(sub_suits)
    counts, sorted_counts = _classify_ints(sub_ranks, n)
    rank_set = frozenset(sub_ranks.tolist())
    st = _is_straight_fast(rank_set) if n == 5 and len(sorted_counts) == 5 else False

    hand_type = _hand_type_from_counts(sorted_counts, fl, st, n)
    chips, mult = HAND_SCORES[hand_type]

    # Sum chip values: rank index i → value i+2 (2..14)
    chips += int(np.sum(sub_ranks) + n * 2)

    # Apply jokers
    for label in joker_labels:
        chips, mult = _apply_jokers_ints(
            label, sub_ranks, sub_suits, n, hand_type, fl, counts, chips, mult, n_jokers,
        )

    return hand_type, chips * mult


def _pick_best_play_fast(
    hand_cards: List[dict],
    strategy: Strategy,
    n_play: int = 5,
    joker_labels: Optional[List[str]] = None,
) -> Tuple[List[int], str, int]:
    """Fast pick_best_play using pre-computed integer arrays."""
    n = len(hand_cards)
    if n == 0:
        return [], "high_card", 0

    all_ranks, all_suits = _cards_to_ints(hand_cards)
    jk = tuple(joker_labels or [])
    n_jk = len(jk)
    prefs = STRATEGY_PREFERRED_HANDS[strategy]

    max_play = min(n_play, n)
    play_sizes = [max_play]
    if "Half Joker" in jk:
        play_sizes = list(range(1, max_play + 1))

    best_indices = list(range(max_play))
    best_score = -1
    best_hand = "high_card"

    for size in play_sizes:
        for combo in combinations(range(n), size):
            combo_idx = np.asarray(combo, dtype=np.intp)
            hand_type, score = _score_hand_ints(all_ranks, all_suits, combo_idx, jk, n_jk)
            preference = prefs.get(hand_type, 1)

            if strategy == Strategy.MULT_BUILD:
                weighted_score = score
            else:
                weighted_score = preference * 10000 + score

            if weighted_score > best_score:
                best_score = weighted_score
                best_indices = list(combo)
                best_hand = hand_type

    return best_indices, best_hand, best_score


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

    Uses the fast integer-based path for performance.

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
    return _pick_best_play_fast(hand_cards, strategy, n_play, joker_labels)


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


# ─────────────────────────────────────────────────────────────
# MONTE CARLO DISCARDING HELPERS
# ─────────────────────────────────────────────────────────────

def get_remaining_deck(hand_cards: List[dict]) -> List[dict]:
    """
    Builds the remaining deck by removing the cards currently in hand
    from a standard 52-card deck.
    """
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = ["C", "D", "H", "S"]
    deck = [{"rank": r, "suit": s} for r in ranks for s in suits]

    hand_counts = {}
    for c in hand_cards:
        key = (c.get("rank", ""), c.get("suit", ""))
        hand_counts[key] = hand_counts.get(key, 0) + 1

    remaining = []
    for c in deck:
        key = (c["rank"], c["suit"])
        if hand_counts.get(key, 0) > 0:
            hand_counts[key] -= 1
        else:
            remaining.append(c)
    return remaining


def get_discard_candidates(
    hand_cards: List[dict],
    strategy: Strategy,
    joker_labels: Optional[List[str]] = None,
) -> List[List[int]]:
    """
    Generates a small list of candidate discard indices (subsets of hand indices).
    Each candidate has 1 to 5 indices.
    """
    candidates = []
    n = len(hand_cards)
    if n == 0:
        return candidates

    # Get indices of current best play
    best_play_indices, _, _ = pick_best_play(hand_cards, strategy, joker_labels=joker_labels)
    best_play_set = set(best_play_indices)

    # 1. Discard all unused cards (up to 5)
    unused_indices = [i for i in range(n) if i not in best_play_set]
    if 1 <= len(unused_indices) <= 5:
        candidates.append(unused_indices)
    elif len(unused_indices) > 5:
        unused_sorted = sorted(unused_indices, key=lambda i: RANK_VALUES.get(hand_cards[i].get("rank", ""), 0))
        candidates.append(unused_sorted[:5])

    # 2. Flush Hunt: For each suit, discard all cards in hand that do not belong to it
    suits = ["C", "D", "H", "S"]
    for s in suits:
        non_suit_indices = [i for i in range(n) if hand_cards[i].get("suit") != s]
        if 1 <= len(non_suit_indices) <= 5:
            candidates.append(non_suit_indices)
        elif len(non_suit_indices) > 5:
            non_suit_sorted = sorted(non_suit_indices, key=lambda i: RANK_VALUES.get(hand_cards[i].get("rank", ""), 0))
            candidates.append(non_suit_sorted[:5])

    # 3. Discard singletons (cards whose rank appears only once in hand)
    rank_counts = {}
    for c in hand_cards:
        r = c.get("rank", "")
        rank_counts[r] = rank_counts.get(r, 0) + 1
    singleton_indices = [i for i in range(n) if rank_counts.get(hand_cards[i].get("rank", "")) == 1]
    if 1 <= len(singleton_indices) <= 5:
        candidates.append(singleton_indices)
    elif len(singleton_indices) > 5:
        singleton_sorted = sorted(singleton_indices, key=lambda i: RANK_VALUES.get(hand_cards[i].get("rank", ""), 0))
        candidates.append(singleton_sorted[:5])

    # 4. Discard the 3, 4, or 5 lowest-ranking cards in hand
    hand_sorted_indices = sorted(range(n), key=lambda i: RANK_VALUES.get(hand_cards[i].get("rank", ""), 0))
    for k in [3, 4, 5]:
        if k <= n:
            candidates.append(hand_sorted_indices[:k])

    # Deduplicate candidates to avoid redundant simulations
    unique_candidates = []
    seen = set()
    for cand in candidates:
        cand_tuple = tuple(sorted(cand))
        if cand_tuple not in seen:
            seen.add(cand_tuple)
            unique_candidates.append(list(cand_tuple))

    return unique_candidates


def evaluate_discard(
    hand_cards: List[dict],
    discard_indices: List[int],
    remaining_deck: List[dict],
    strategy: Strategy,
    joker_labels: Optional[List[str]] = None,
    num_simulations: int = 50,
) -> float:
    """
    Simulates drawing cards to evaluate the EV of discarding a subset of cards.
    """
    import random
    k = len(discard_indices)
    if k == 0 or len(remaining_deck) < k:
        return -1.0

    kept_cards = [hand_cards[i] for i in range(len(hand_cards)) if i not in discard_indices]
    total_score = 0.0

    for _ in range(num_simulations):
        drawn = random.sample(remaining_deck, k)
        simulated_hand = kept_cards + drawn
        _, _, score = pick_best_play(simulated_hand, strategy, joker_labels=joker_labels)
        total_score += score

    return total_score / num_simulations


def pick_best_action(
    hand_cards: List[dict],
    strategy: Strategy,
    discards_left: int,
    joker_labels: Optional[List[str]] = None,
    num_simulations: int = 15,
) -> Tuple[str, List[int], Optional[str]]:
    """
    Decides whether to play or discard using Monte Carlo evaluation of discard candidates.
    Returns: (action_type, indices, hand_type) where action_type is "play" or "discard",
             indices are 0-based indices into hand_cards, and hand_type is the classified
             play hand type (or None if discarding).
    """
    joker_labels = joker_labels or []
    play_indices, play_hand, play_score = pick_best_play(hand_cards, strategy, joker_labels=joker_labels)

    if discards_left <= 0:
        return "play", play_indices, play_hand

    remaining_deck = get_remaining_deck(hand_cards)
    candidates = get_discard_candidates(hand_cards, strategy, joker_labels=joker_labels)

    best_discard_indices = []
    best_discard_ev = -1.0

    for cand in candidates:
        ev = evaluate_discard(
            hand_cards=hand_cards,
            discard_indices=cand,
            remaining_deck=remaining_deck,
            strategy=strategy,
            joker_labels=joker_labels,
            num_simulations=num_simulations,
        )
        if ev > best_discard_ev:
            best_discard_ev = ev
            best_discard_indices = cand

    # Discard if expected value of discarding is strictly greater than playing immediately
    if best_discard_ev > play_score:
        return "discard", best_discard_indices, None

    return "play", play_indices, play_hand

