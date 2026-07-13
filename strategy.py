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
from functools import lru_cache
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


# ─────────────────────────────────────────────────────────────
# FAST CACHED SCORING CORE
#
# All hot-path scoring runs on hashable (rank, suit) tuples so
# results can be memoized. Monte Carlo discard simulation draws
# the same 5-card combos over and over — the cache turns those
# repeat evaluations into dict lookups (~100-300x speedup on
# the training hot path).
# ─────────────────────────────────────────────────────────────

def _cards_to_keys(cards: List[dict]) -> List[Tuple[str, str]]:
    """Convert card dicts to hashable (rank, suit) tuples."""
    return [(c.get("rank", ""), c.get("suit", "")) for c in cards]


def _classify_keys(cards_key: Tuple[Tuple[str, str], ...]) -> Tuple[str, bool, dict]:
    """Classify a played hand given (rank, suit) tuples.

    Returns (hand_type, is_flush, rank_counts).
    """
    ranks = [r for r, _ in cards_key]
    suits = [s for _, s in cards_key]
    n = len(cards_key)
    rank_vals = sorted((RANK_VALUES.get(r, 0) for r in ranks), reverse=True)

    rank_counts: dict = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    counts = sorted(rank_counts.values(), reverse=True)

    is_flush    = n == 5 and len(set(suits)) == 1
    is_straight = (n == 5 and len(set(rank_vals)) == 5 and
                   rank_vals[0] - rank_vals[4] == 4)
    if n == 5 and set(rank_vals) == {14, 2, 3, 4, 5}:
        is_straight = True

    if n == 5 and is_flush and counts == [5]:
        hand_type = "flush_five"
    elif n == 5 and is_flush and counts == [3, 2]:
        hand_type = "flush_house"
    elif n == 5 and counts == [5]:
        hand_type = "five_of_a_kind"
    elif is_flush and is_straight:
        hand_type = "straight_flush"
    elif n >= 4 and counts[0] == 4:
        hand_type = "four_of_a_kind"
    elif n == 5 and counts == [3, 2]:
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

    return hand_type, is_flush, rank_counts


_FACE_RANKS = frozenset({"J", "Q", "K"})


def _apply_joker_keys(
    label: str,
    cards_key: Tuple[Tuple[str, str], ...],
    hand_type: str,
    is_flush: bool,
    rank_counts: dict,
    chips: int,
    mult: int,
    n_jokers: int,
) -> Tuple[int, int]:
    """Apply one whitelisted joker effect (tuple-card fast path)."""
    pair_count   = sum(1 for count in rank_counts.values() if count >= 2)
    has_pair     = pair_count >= 1
    has_three    = any(count >= 3 for count in rank_counts.values())
    has_two_pair = pair_count >= 2

    if label == "Droll Joker" and is_flush:
        mult += 10
    elif label == "Crafty Joker" and is_flush:
        chips += 80
    elif label == "Lusty Joker":
        mult += 3 * sum(1 for _, s in cards_key if s == "H")
    elif label == "Greedy Joker":
        mult += 3 * sum(1 for _, s in cards_key if s == "D")
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
    elif label == "Half Joker" and len(cards_key) <= 3:
        mult += 20
    elif label == "Scary Face":
        chips += 30 * sum(1 for r, _ in cards_key if r in _FACE_RANKS)

    return chips, mult


@lru_cache(maxsize=262_144)
def _score_keys(
    cards_key: Tuple[Tuple[str, str], ...],
    jokers_key: Tuple[str, ...],
) -> Tuple[str, int]:
    """Memoized hand scorer. cards_key MUST be sorted for cache hits
    across card orderings (scoring is order-independent)."""
    hand_type, is_flush, rank_counts = _classify_keys(cards_key)
    chips, mult = HAND_SCORES[hand_type]
    chips += sum(RANK_VALUES.get(r, 0) for r, _ in cards_key)

    n_jokers = len(jokers_key)
    for label in jokers_key:
        chips, mult = _apply_joker_keys(
            label, cards_key, hand_type, is_flush, rank_counts,
            chips, mult, n_jokers,
        )

    return hand_type, chips * mult


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


def _pick_best_play_keys(
    keys: List[Tuple[str, str]],
    strategy: Strategy,
    jokers_key: Tuple[str, ...],
    play_sizes: List[int],
) -> Tuple[List[int], str, int]:
    """Core best-play search over hashable card keys (cache-backed)."""
    n = len(keys)
    best_indices  = list(range(min(play_sizes[-1], n)))
    best_score    = -1
    best_hand     = "high_card"
    prefs         = STRATEGY_PREFERRED_HANDS[strategy]
    is_mult       = strategy == Strategy.MULT_BUILD

    for size in play_sizes:
        for combo in combinations(range(n), size):
            combo_key = tuple(sorted(keys[i] for i in combo))
            hand_type, base_score = _score_keys(combo_key, jokers_key)

            if is_mult:
                weighted_score = base_score
            else:
                weighted_score = prefs.get(hand_type, 1) * 10000 + base_score

            if weighted_score > best_score:
                best_score   = weighted_score
                best_indices = list(combo)
                best_hand    = hand_type

    return best_indices, best_hand, best_score


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

    return _pick_best_play_keys(
        _cards_to_keys(hand_cards), strategy, tuple(joker_labels), play_sizes
    )


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

    # 2. Flush Hunt: only chase suits already holding >= 3 cards — hunting
    #    a suit with fewer is nearly always EV-negative and just burns
    #    Monte Carlo budget.
    suit_counts: dict = {}
    for c in hand_cards:
        s = c.get("suit", "")
        suit_counts[s] = suit_counts.get(s, 0) + 1
    for s, cnt in sorted(suit_counts.items(), key=lambda kv: -kv[1]):
        if cnt < 3:
            break
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

    # 4. Discard the 4 lowest-ranking cards in hand
    hand_sorted_indices = sorted(range(n), key=lambda i: RANK_VALUES.get(hand_cards[i].get("rank", ""), 0))
    if 4 <= n:
        candidates.append(hand_sorted_indices[:4])

    # Deduplicate candidates to avoid redundant simulations
    unique_candidates = []
    seen = set()
    for cand in candidates:
        cand_tuple = tuple(sorted(cand))
        if cand_tuple not in seen:
            seen.add(cand_tuple)
            unique_candidates.append(list(cand_tuple))

    # Cap the Monte Carlo budget — candidates are ordered by heuristic
    # quality (unused-cards first, then flush hunts, singletons, low cards)
    return unique_candidates[:4]


def evaluate_discard(
    hand_cards: List[dict],
    discard_indices: List[int],
    remaining_deck: List[dict],
    strategy: Strategy,
    joker_labels: Optional[List[str]] = None,
    num_simulations: int = 12,
) -> float:
    """
    Simulates drawing cards to evaluate the EV of discarding a subset of cards.

    Simulations use the fast path: 5-card plays only (no Half Joker
    size exploration) over memoized scoring — the final play decision
    still uses the full search.
    """
    import random
    k = len(discard_indices)
    if k == 0 or len(remaining_deck) < k:
        return -1.0

    discard_set = set(discard_indices)
    kept_keys = [
        (c.get("rank", ""), c.get("suit", ""))
        for i, c in enumerate(hand_cards) if i not in discard_set
    ]
    deck_keys = [(c.get("rank", ""), c.get("suit", "")) for c in remaining_deck]
    jokers_key = tuple(joker_labels or [])
    total_score = 0.0

    for _ in range(num_simulations):
        sim_keys = kept_keys + random.sample(deck_keys, k)
        play_size = min(5, len(sim_keys))
        _, _, score = _pick_best_play_keys(sim_keys, strategy, jokers_key, [play_size])
        total_score += score

    return total_score / num_simulations


def pick_best_action(
    hand_cards: List[dict],
    strategy: Strategy,
    discards_left: int,
    joker_labels: Optional[List[str]] = None,
    num_simulations: int = 12,
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

    # Early exit: the hand already achieves a top-tier result for this
    # strategy — discarding can only risk it, so skip the Monte Carlo
    # evaluation entirely.
    if STRATEGY_PREFERRED_HANDS[strategy].get(play_hand, 1) >= 700:
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