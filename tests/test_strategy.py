from strategy import (
    Strategy,
    parse_jokers_from_gamestate,
    pick_best_play,
    strategy_coherence_reward,
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
