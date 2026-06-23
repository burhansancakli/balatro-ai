from env import BalatroEnv
from strategy import Strategy


def make_env():
    return BalatroEnv(port=12346, save_path="C:/tmp/unused_test_save.jkr")


def test_has_joker_slot_respects_limit():
    env = make_env()

    assert env._has_joker_slot({"jokers": {"cards": [{"label": "Joker"}], "limit": 5}})
    assert not env._has_joker_slot({"jokers": {"cards": [{}, {}, {}, {}, {}], "limit": 5}})
