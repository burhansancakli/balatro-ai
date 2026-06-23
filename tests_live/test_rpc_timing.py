import os

import pytest

from tests_live.rpc_timing import time_rpc_call


@pytest.mark.live
def test_rpc_timing_gamestate():
    port = int(os.environ.get("BALATROBOT_PORT", "12346"))
    elapsed, state = time_rpc_call("gamestate", port=port, timeout=60)

    assert elapsed >= 0
    assert state is not None
