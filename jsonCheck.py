import requests, json
r = requests.post("http://127.0.0.1:12346", json={
    "jsonrpc": "2.0", "method": "gamestate", "params": {}, "id": 1
}, timeout=10).json()["result"]
print(json.dumps(r.get("shop", {}), indent=2))
print("Money:", r.get("money"))