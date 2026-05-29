from env import BalatroEnv
from strategy import Strategy


def make_env():
    return BalatroEnv(port=12346, save_path="C:/tmp/unused_test_save.jkr")


def test_shop_selection_picks_strategy_matching_joker():
    env = make_env()
    state = {
        "money": 5,
        "jokers": {"cards": [], "limit": 5},
        "shop": {
            "cards": [
                {"label": "Jolly Joker", "key": "j_jolly", "set": "JOKER", "cost": {"buy": 4}},
                {"label": "Droll Joker", "key": "j_droll_joker", "set": "JOKER", "cost": {"buy": 4}},
            ],
        },
    }

    assert env._choose_shop_joker(state, Strategy.PAIR_BUILD) == 0
    assert env._choose_shop_joker(state, Strategy.FLUSH_BUILD) == 1


def test_shop_selection_skips_unaffordable_joker():
    env = make_env()
    state = {
        "money": 3,
        "jokers": {"cards": [], "limit": 5},
        "shop": {
            "cards": [
                {"label": "Jolly Joker", "key": "j_jolly", "set": "JOKER", "cost": {"buy": 4}},
            ],
        },
    }

    assert env._choose_shop_joker(state, Strategy.PAIR_BUILD) is None


def test_has_joker_slot_respects_limit():
    env = make_env()

    assert env._has_joker_slot({"jokers": {"cards": [{"label": "Joker"}], "limit": 5}})
    assert not env._has_joker_slot({"jokers": {"cards": [{}, {}, {}, {}, {}], "limit": 5}})
