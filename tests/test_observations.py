import numpy as np

from observations import OBS_SIZE, gamestate_to_observation


def test_empty_gamestate_observation_has_expected_shape():
    obs = gamestate_to_observation({})

    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_observation_encodes_known_joker_slot():
    state = {
        "jokers": {
            "cards": [
                {"label": "Joker"},
            ],
        },
    }

    obs = gamestate_to_observation(state)

    assert obs[4] >= 0.0
    assert obs[5] == 1.0


def test_observation_encodes_shop_cards():
    state = {
        "shop": {
            "cards": [
                {"label": "Joker", "cost": {"buy": 4}},
                {"label": "Abstract Joker", "cost": {"buy": 6}},
            ],
        },
    }

    obs = gamestate_to_observation(state)

    # Shop slots start at index 14
    assert obs[14] >= 0.0       # first shop joker idx
    assert obs[15] > 0.0        # first shop cost
    assert obs[16] >= 0.0       # second shop joker idx
    assert obs[17] > 0.0        # second shop cost
    # Empty slots should be -1
    assert obs[18] == -1.0
    assert obs[19] == 0.0
