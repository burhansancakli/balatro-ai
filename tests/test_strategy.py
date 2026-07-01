from strategy import (
    Strategy,
    parse_jokers_from_gamestate,
    pick_best_play,
    strategy_coherence_reward,
    get_remaining_deck,
    get_discard_candidates,
    evaluate_discard,
    pick_best_action,
)


def test_flush_strategy_prefers_flush_hand():
    cards = [
        {"rank": "2", "suit": "H"},
        {"rank": "4", "suit": "H"},
        {"rank": "6", "suit": "H"},
        {"rank": "8", "suit": "H"},
        {"rank": "T", "suit": "H"},
        {"rank": "A", "suit": "S"},
        {"rank": "A", "suit": "C"},
        {"rank": "K", "suit": "D"},
    ]

    indices, hand_type, score = pick_best_play(cards, Strategy.FLUSH_BUILD)

    assert indices == [0, 1, 2, 3, 4]
    assert hand_type == "flush"
    assert score > 0


def test_pair_strategy_prefers_pair_over_high_card():
    cards = [
        {"rank": "A", "suit": "S"},
        {"rank": "A", "suit": "C"},
        {"rank": "K", "suit": "D"},
        {"rank": "Q", "suit": "H"},
        {"rank": "9", "suit": "S"},
        {"rank": "7", "suit": "C"},
        {"rank": "5", "suit": "D"},
        {"rank": "3", "suit": "H"},
    ]

    indices, hand_type, score = pick_best_play(cards, Strategy.PAIR_BUILD)

    assert 0 in indices and 1 in indices
    assert hand_type == "pair"
    assert score > 0


def test_strategy_coherence_reward_is_normalized():
    assert strategy_coherence_reward("flush", Strategy.FLUSH_BUILD) == 0.7
    assert 0.0 < strategy_coherence_reward("high_card", Strategy.FLUSH_BUILD) <= 1.0


def test_scary_face_prefers_face_cards_for_mult_build():
    cards = [
        {"rank": "K", "suit": "S"},
        {"rank": "Q", "suit": "C"},
        {"rank": "J", "suit": "D"},
        {"rank": "9", "suit": "H"},
        {"rank": "8", "suit": "S"},
        {"rank": "7", "suit": "C"},
    ]

    indices, hand_type, score = pick_best_play(
        cards,
        Strategy.MULT_BUILD,
        joker_labels=["Scary Face"],
    )

    assert {0, 1, 2}.issubset(indices)
    assert hand_type == "high_card"
    assert score > 0


def test_half_joker_allows_short_play():
    cards = [
        {"rank": "A", "suit": "S"},
        {"rank": "K", "suit": "C"},
        {"rank": "Q", "suit": "D"},
        {"rank": "2", "suit": "H"},
        {"rank": "3", "suit": "S"},
    ]

    indices, hand_type, score = pick_best_play(
        cards,
        Strategy.MULT_BUILD,
        joker_labels=["Half Joker"],
    )

    assert len(indices) <= 3
    assert hand_type == "high_card"
    assert score > 0


def test_parse_jokers_from_gamestate():
    state = {
        "jokers": {
            "cards": [
                {"label": "Joker"},
                {"label": "Abstract Joker"},
            ],
        },
    }

    assert parse_jokers_from_gamestate(state) == ["Joker", "Abstract Joker"]


def test_get_remaining_deck():
    hand = [
        {"rank": "A", "suit": "S"},
        {"rank": "K", "suit": "S"},
        {"rank": "Q", "suit": "S"},
        {"rank": "J", "suit": "S"},
        {"rank": "T", "suit": "S"},
    ]
    deck = get_remaining_deck(hand)
    assert len(deck) == 47
    for card in hand:
        assert card not in deck


def test_get_discard_candidates():
    hand = [
        {"rank": "A", "suit": "S"},
        {"rank": "K", "suit": "S"},
        {"rank": "Q", "suit": "S"},
        {"rank": "J", "suit": "S"},
        {"rank": "2", "suit": "H"},
        {"rank": "3", "suit": "C"},
        {"rank": "4", "suit": "D"},
        {"rank": "5", "suit": "S"},
    ]
    candidates = get_discard_candidates(hand, Strategy.FLUSH_BUILD)
    assert len(candidates) > 0
    for cand in candidates:
        assert isinstance(cand, list)
        assert 1 <= len(cand) <= 5
        for idx in cand:
            assert 0 <= idx < len(hand)


def test_pick_best_action_no_discards_left():
    hand = [
        {"rank": "A", "suit": "S"},
        {"rank": "A", "suit": "H"},
        {"rank": "Q", "suit": "S"},
        {"rank": "J", "suit": "S"},
        {"rank": "2", "suit": "H"},
        {"rank": "3", "suit": "C"},
        {"rank": "4", "suit": "D"},
        {"rank": "5", "suit": "S"},
    ]
    action_type, indices, hand_type = pick_best_action(hand, Strategy.PAIR_BUILD, discards_left=0)
    assert action_type == "play"
    assert hand_type == "pair"
    assert {0, 1}.issubset(set(indices))


def test_pick_best_action_flush_hunt():
    hand = [
        {"rank": "A", "suit": "H"},
        {"rank": "K", "suit": "H"},
        {"rank": "Q", "suit": "H"},
        {"rank": "J", "suit": "H"},
        {"rank": "2", "suit": "S"},
        {"rank": "3", "suit": "C"},
        {"rank": "4", "suit": "D"},
        {"rank": "5", "suit": "C"},
    ]
    action_type, indices, hand_type = pick_best_action(hand, Strategy.FLUSH_BUILD, discards_left=2, num_simulations=100)
    assert action_type == "discard"
    assert set(indices).issubset({4, 5, 6, 7})


def test_pick_best_action_flush_already_held():
    hand = [
        {"rank": "A", "suit": "H"},
        {"rank": "K", "suit": "H"},
        {"rank": "Q", "suit": "H"},
        {"rank": "J", "suit": "H"},
        {"rank": "9", "suit": "H"},
        {"rank": "2", "suit": "S"},
        {"rank": "3", "suit": "C"},
        {"rank": "4", "suit": "D"},
    ]
    action_type, indices, hand_type = pick_best_action(hand, Strategy.FLUSH_BUILD, discards_left=2, num_simulations=100)
    # Since we already have a flush, we should play it and not waste a discard.
    assert action_type == "play"
    assert hand_type == "flush"


def test_pick_best_action_pair_still_discarded_for_upgrade():
    hand = [
        {"rank": "A", "suit": "S"},
        {"rank": "A", "suit": "H"},
        {"rank": "2", "suit": "S"},
        {"rank": "3", "suit": "C"},
        {"rank": "4", "suit": "D"},
        {"rank": "5", "suit": "S"},
        {"rank": "6", "suit": "H"},
        {"rank": "7", "suit": "C"},
    ]
    action_type, indices, hand_type = pick_best_action(hand, Strategy.PAIR_BUILD, discards_left=2, num_simulations=100)
    # A Pair is not a satisfactory hand, and the EV of discarding to find Three-of-a-Kind/Full House
    # is at least 10% higher than just playing the Pair.
    assert action_type == "discard"


def test_pick_best_action_mult_build_no_minor_discard():
    # A Flush under MULT_BUILD
    hand = [
        {"rank": "A", "suit": "H"},
        {"rank": "K", "suit": "H"},
        {"rank": "Q", "suit": "H"},
        {"rank": "J", "suit": "H"},
        {"rank": "9", "suit": "H"},
        {"rank": "2", "suit": "S"},
        {"rank": "3", "suit": "C"},
        {"rank": "4", "suit": "D"},
    ]
    action_type, indices, hand_type = pick_best_action(hand, Strategy.MULT_BUILD, discards_left=2, num_simulations=100)
    # The expected improvement from discarding to turn the 9 of Hearts into a higher heart is minor (< 10%).
    # So we should play it immediately.
    assert action_type == "play"
    assert hand_type == "flush"


