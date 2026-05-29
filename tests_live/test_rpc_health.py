import os

import pytest
import requests


@pytest.mark.live
def test_balatrobot_health():
    port = int(os.environ.get("BALATROBOT_PORT", "12346"))
    response = requests.post(
        f"http://127.0.0.1:{port}",
        json={"jsonrpc": "2.0", "method": "health", "params": {}, "id": 1},
        timeout=5,
    )

    assert response.json()["result"]["status"] == "ok"
