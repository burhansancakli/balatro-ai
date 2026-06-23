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
