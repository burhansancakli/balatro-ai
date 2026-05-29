"""
rpc_timing.py - Time one JSON-RPC call against a running Balatrobot server.
"""

import argparse
import time

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", nargs="?", default="gamestate")
    parser.add_argument("--port", type=int, default=12346)
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    payload = {"jsonrpc": "2.0", "method": args.method, "params": {}, "id": 1}

    start = time.perf_counter()
    response = requests.post(url, json=payload, timeout=60)
    elapsed = time.perf_counter() - start
    data = response.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    state = data.get("result", {}).get("state")
    print(f"{args.method} OK in {elapsed:.3f}s; state={state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
