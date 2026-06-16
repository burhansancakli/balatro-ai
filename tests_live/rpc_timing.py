"""
RPC timing helper for live Balatrobot tests.
"""

import time

import requests


def time_rpc_call(method: str = "gamestate", port: int = 12346, timeout: int = 60) -> tuple[float, object]:
    url = f"http://127.0.0.1:{port}"
    payload = {"jsonrpc": "2.0", "method": method, "params": {}, "id": 1}

    start = time.perf_counter()
    response = requests.post(url, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - start
    data = response.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    state = data.get("result", {}).get("state")
    return elapsed, state
