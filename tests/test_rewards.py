from env import BalatroEnv


def test_progress_reward_scales_with_blind_progress():
    env = BalatroEnv(port=12346, save_path="C:/tmp/unused_test_save.jkr")
    state = {
        "round": {"chips": 50},
        "blinds": {
            "small": {"status": "CURRENT", "score": 100},
        },
    }

    assert env._progress_reward(state) == 0.01


def test_progress_reward_clips_at_one_blind():
    env = BalatroEnv(port=12346, save_path="C:/tmp/unused_test_save.jkr")
    state = {
        "round": {"chips": 200},
        "blinds": {
            "small": {"status": "CURRENT", "score": 100},
        },
    }

    assert env._progress_reward(state) == 0.02
