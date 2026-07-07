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
        "counts": counts,
    }
    return hand_type, metadata


def _score_hand(cards: List[dict], joker_labels: Optional[List[str]] = None,
                _cache: Optional[dict] = None) -> Tuple[str, int]:
    """
    Score a played hand using simplified Balatro Chips x Mult arithmetic.

    The 13 whitelisted joker effects are modeled deterministically. This keeps
    the low-level executor non-RL while allowing shop purchases to influence
    card choice.

    Args:
        _cache: Optional per-call dict cache keyed by sorted card tuples.
                Avoids redundant scoring of the same 5-card combo across
                Monte Carlo simulations within a single pick_best_action call.
    """
    # Check per-call cache (cards sorted for canonical key)
    if _cache is not None:
        cache_key = tuple(sorted((c.get("rank", ""), c.get("suit", "")) for c in cards))
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

    hand_type, metadata = _classify_hand(cards)
    chips, mult = HAND_SCORES[hand_type]
    # Inline chip sum — avoids function call overhead per card
    card_chips = 0
    for c in cards:
        card_chips += RANK_VALUES.get(c.get("rank", ""), 0)
    chips += card_chips

    for label in joker_labels or []:
        chips, mult = _apply_joker(label, cards, hand_type, metadata, chips, mult, joker_labels or [])

    result = (hand_type, chips * mult)

    if _cache is not None:
        _cache[cache_key] = result

    return result


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
    # Use pre-computed sorted counts from metadata to avoid re-scanning
    # rank_counts.values() with any()/sum() on every joker call.
    counts = metadata["counts"]
    has_pair = counts[0] >= 2 if counts else False
    has_three = counts[0] >= 3 if counts else False
    has_two_pair = len(counts) >= 2 and counts[1] >= 2
    has_flush = metadata["is_flush"]

    if label == "Droll Joker":
        if has_flush:
            mult += 10
    elif label == "Crafty Joker":
        if has_flush:
            chips += 80
    elif label == "Lusty Joker":
        count = 0
        for card in cards:
            if card.get("suit") == "H":
                count += 1
        mult += 3 * count
    elif label == "Greedy Joker":
        count = 0
        for card in cards:
            if card.get("suit") == "D":
                count += 1
        mult += 3 * count
    elif label == "Jolly Joker":
        if has_pair:
            mult += 8
    elif label == "Zany Joker":
        if has_three:
            mult += 12
    elif label == "Mad Joker":
        if has_two_pair:
            mult += 10
    elif label == "Sly Joker":
        if has_pair:
            chips += 50
    elif label == "Wily Joker":
        if has_three:
            chips += 100
    elif label == "Joker":
        mult += 4
    elif label == "Abstract Joker":
        mult += 3 * len(joker_labels)
    elif label == "Half Joker" and len(cards) <= 3:
        mult += 20
    elif label == "Scary Face":
        count = 0
        for card in cards:
            if card.get("rank") in {"J", "Q", "K"}:
                count += 1
        chips += 30 * count

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
    _cache: Optional[dict] = None,
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
            hand_type, base_score = _score_hand(cards, joker_labels, _cache=_cache)
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

    # Cap at 4 candidates to limit Monte Carlo work — the most promising
    # candidates appear first (unused cards, flush hunts, singletons)
    # so truncating at 4 rarely loses the best option.
    return unique_candidates[:4]


def evaluate_discard(
    hand_cards: List[dict],
    discard_indices: List[int],
    remaining_deck: List[dict],
    strategy: Strategy,
    joker_labels: Optional[List[str]] = None,
    num_simulations: int = 50,
    play_score: float = 0.0,
    _cache: Optional[dict] = None,
) -> float:
    """
    Simulates drawing cards to evaluate the EV of discarding a subset of cards.

    Uses early-exit pruning: if after half the simulations the running EV
    is below the current play_score, the remaining simulations are skipped
    since this candidate is unlikely to beat the play threshold.
    """
    import random
    k = len(discard_indices)
    if k == 0 or len(remaining_deck) < k:
        return -1.0

    discard_set = set(discard_indices)
    kept_cards = [hand_cards[i] for i in range(len(hand_cards)) if i not in discard_set]
    total_score = 0.0
    half = num_simulations // 2

    for sim_i in range(num_simulations):
        drawn = random.sample(remaining_deck, k)
        simulated_hand = kept_cards + drawn
        _, _, score = pick_best_play(simulated_hand, strategy, joker_labels=joker_labels, _cache=_cache)
        total_score += score

        # Early exit: if halfway through and running EV is below play_score,
        # this candidate is unlikely to beat the 1.10x threshold — skip it.
        if sim_i == half and half > 0 and play_score > 0:
            running_ev = total_score / (sim_i + 1)
            if running_ev < play_score:
                return running_ev

    return total_score / num_simulations


def pick_best_action(
    hand_cards: List[dict],
    strategy: Strategy,
    discards_left: int,
    joker_labels: Optional[List[str]] = None,
    num_simulations: int = 20,
) -> Tuple[str, List[int], Optional[str]]:
    """
    Decides whether to play or discard using Monte Carlo evaluation of discard candidates.
    Returns: (action_type, indices, hand_type) where action_type is "play" or "discard",
             indices are 0-based indices into hand_cards, and hand_type is the classified
             play hand type (or None if discarding).
    """
    joker_labels = joker_labels or []
    # Per-call cache: avoids re-scoring the same 5-card combo across MC simulations.
    # Keyed by sorted (rank, suit) tuples. Discarded after this call returns.
    _cache: dict = {}
    play_indices, play_hand, play_score = pick_best_play(hand_cards, strategy, joker_labels=joker_labels, _cache=_cache)

    if discards_left <= 0:
        return "play", play_indices, play_hand

    # If the current best play is already a satisfactory target hand for the strategy, do not discard.
    satisfactory_hands = {
        Strategy.FLUSH_BUILD: {"flush", "straight_flush", "flush_house", "flush_five"},
        Strategy.PAIR_BUILD: {"three_of_a_kind", "full_house", "four_of_a_kind", "five_of_a_kind", "flush_house", "flush_five"},
        Strategy.MULT_BUILD: {"straight", "flush", "full_house", "four_of_a_kind", "straight_flush", "five_of_a_kind", "flush_house", "flush_five"},
    }
    if play_hand in satisfactory_hands.get(strategy, set()):
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
            play_score=play_score,
            _cache=_cache,
        )
        if ev > best_discard_ev:
            best_discard_ev = ev
            best_discard_indices = cand

    # Discard only if expected value of discarding is at least 10% greater than playing immediately
    if best_discard_ev > play_score * 1.10:
        return "discard", best_discard_indices, None

    return "play", play_indices, play_hand

