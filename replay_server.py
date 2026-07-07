"""Standalone replay server for balatro-ai episode logs.

Usage::

    python replay_server.py                        # reads logs/episodes/
    python replay_server.py --log-dir ./logs/episodes --port 8765

Architecture:
  HTTP  (port)      — static files + /api/episodes
  WebSocket (port+1)— replay control (browser ↔ server)

WebSocket protocol:
  Browser → server:
    {"cmd": "list"}
    {"cmd": "load", "episode": N}   — N is the array index in the merged list
    {"cmd": "step", "index": I}
    {"cmd": "next"}
    {"cmd": "prev"}

  Server → browser:
    {"type": "episodes", "episodes": [...]}
    {"type": "episode", ...}
    {"type": "cursor", "step_index": I, "total_steps": T}
"""

from __future__ import annotations

import argparse
import asyncio
import http.server
import json
import threading
from pathlib import Path
from typing import Any

STATIC_DIR = Path(__file__).parent / "static"


# ── JSONL loading ──────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    episodes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return episodes


def _load_all_episodes(log_dir: Path) -> list[dict]:
    """Load and merge all episodes_*.jsonl files, sorted by episode number."""
    all_eps: list[dict] = []
    for path in sorted(log_dir.glob("episodes_*.jsonl")):
        all_eps.extend(_load_jsonl(path))
    # Sort by recorded episode number so the list is chronological
    all_eps.sort(key=lambda e: e.get("episode", 0))
    return all_eps


# ── Payload builders ───────────────────────────────────────────

def _episode_summary(ep: dict, idx: int) -> dict:
    action_log = ep.get("action_log", [])
    hands_played = sum(1 for a in action_log if a.get("type") == "play")
    return {
        "index":         idx,
        "episode":       ep.get("episode", idx),
        "seed":          ep.get("seed", ""),
        "ante_reached":  ep.get("ante_reached", 0),
        "rounds_beaten": ep.get("rounds_beaten", 0),
        "won":           ep.get("won", False),
        "total_reward":  ep.get("total_reward", 0.0),
        "steps":         ep.get("steps", 0),
        "hands_played":  hands_played,
    }


def _build_episode_payload(ep: dict, idx: int) -> dict:
    action_log = ep.get("action_log", [])
    hand_history = ep.get("hand_history", [])
    total_steps = len(action_log)

    play_indices = [i for i, e in enumerate(action_log) if e.get("type") == "play"]

    # No explicit hand_history recorded — derive it from the play entries
    # in the action log so the hand-history strip still populates.
    if not hand_history:
        hand_counter: dict = {}
        for i in play_indices:
            e = action_log[i]
            key = (e.get("ante", 0), e.get("round", 0))
            hand_counter[key] = hand_counter.get(key, 0) + 1
            hand_history.append({
                "ante":      e.get("ante", 0),
                "round":     e.get("round", 0),
                "hand_num":  hand_counter[key],
                "hand_type": e.get("hand_type", "?"),
                "cards":     e.get("cards", []),
                "jokers":    e.get("jokers", []),
                "strategy":  e.get("strategy", ""),
            })

    enriched_history = []
    for i, h in enumerate(hand_history):
        step = play_indices[i] if i < len(play_indices) else max(total_steps - 1, 0)
        enriched_history.append({**h, "_step": step})

    return {
        "type":          "episode",
        "episode_index": ep.get("episode", idx),
        "seed":          ep.get("seed", ""),
        "total_steps":   total_steps,
        "won":           ep.get("won", False),
        "ante_reached":  ep.get("ante_reached", 0),
        "rounds_beaten": ep.get("rounds_beaten", 0),
        "action_log":    action_log,
        "hand_history":  enriched_history,
        "jokers_at_end": ep.get("jokers_at_end", []),
        "step_index":    0,
    }


def _build_cursor(step_index: int, total_steps: int) -> dict:
    return {"type": "cursor", "step_index": step_index, "total_steps": total_steps}


# ── Server ─────────────────────────────────────────────────────

class ReplayServer:
    def __init__(self, log_dir: Path, *, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._log_dir  = log_dir
        self._host     = host
        self._port     = port
        self._ws_port  = port + 1
        self._episodes: list[dict] = []
        self._started  = threading.Event()

    def start(self) -> None:
        self._episodes = _load_all_episodes(self._log_dir)
        print(f"\n  [balatro-ai] Loaded {len(self._episodes)} episode(s) from {self._log_dir}")
        threading.Thread(target=self._run_http, daemon=True, name="balatro-replay-http").start()
        threading.Thread(target=self._run_ws,   daemon=True, name="balatro-replay-ws").start()
        self._started.wait(timeout=8)
        print(f"  [balatro-ai] Replay dashboard -> http://{self._host}:{self._port}/?mode=replay\n")

    # ── HTTP ───────────────────────────────────────────────────

    def _run_http(self) -> None:
        static_dir = STATIC_DIR
        ws_port    = self._ws_port
        episodes   = self._episodes

        class Handler(http.server.BaseHTTPRequestHandler):
            _TYPES = {
                "/style.css": "text/css",
                "/app.js":    "application/javascript",
            }

            def do_GET(self) -> None:
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    content = (static_dir / "index.html").read_text(encoding="utf-8")
                    content = content.replace("__WS_PORT__", str(ws_port))
                    data = content.encode()
                    ct = "text/html; charset=utf-8"
                elif path == "/api/episodes":
                    summary = [_episode_summary(ep, i) for i, ep in enumerate(episodes)]
                    data = json.dumps(summary, ensure_ascii=False).encode()
                    ct = "application/json; charset=utf-8"
                elif path in self._TYPES:
                    data = (static_dir / path.lstrip("/")).read_bytes()
                    ct = self._TYPES[path]
                else:
                    self.send_response(404); self.end_headers(); return
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_: Any) -> None:
                pass

        server = http.server.HTTPServer((self._host, self._port), Handler)
        server.serve_forever()

    # ── WebSocket ──────────────────────────────────────────────

    def _run_ws(self) -> None:
        try:
            asyncio.run(self._serve_ws())
        except Exception as exc:
            print(f"\n[balatro-replay-ws] CRASH: {exc}\n")
            self._started.set()

    async def _serve_ws(self) -> None:
        try:
            import websockets
        except ImportError:
            print("\n[balatro-ai] websockets package missing — run: pip install websockets\n")
            self._started.set()
            return

        episodes = self._episodes

        async def handler(ws: Any) -> None:
            current_ep   = 0
            current_step = 0

            async def send_episodes() -> None:
                summary = [_episode_summary(ep, i) for i, ep in enumerate(episodes)]
                await ws.send(json.dumps({"type": "episodes", "episodes": summary}))

            async def send_episode(ep_idx: int) -> None:
                ep = episodes[ep_idx]
                await ws.send(json.dumps(_build_episode_payload(ep, ep_idx), default=str))

            async def send_cursor(step: int) -> None:
                total = len(episodes[current_ep].get("action_log", []))
                await ws.send(json.dumps(_build_cursor(step, total), default=str))

            try:
                await send_episodes()
                if episodes:
                    await send_episode(0)
                    await send_cursor(0)
            except Exception:
                return

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    cmd = msg.get("cmd", "")

                    if cmd == "list":
                        await send_episodes()

                    elif cmd == "load":
                        ep_idx = int(msg.get("episode", 0))
                        if 0 <= ep_idx < len(episodes):
                            current_ep   = ep_idx
                            current_step = 0
                            await send_episode(current_ep)
                            await send_cursor(current_step)

                    elif cmd == "step":
                        total = len(episodes[current_ep].get("action_log", []))
                        current_step = max(0, min(int(msg.get("index", 0)), total - 1))
                        await send_cursor(current_step)

                    elif cmd == "next":
                        total = len(episodes[current_ep].get("action_log", []))
                        if current_step < total - 1:
                            current_step += 1
                        await send_cursor(current_step)

                    elif cmd == "prev":
                        if current_step > 0:
                            current_step -= 1
                        await send_cursor(current_step)

                except Exception:
                    pass

        async with websockets.serve(handler, self._host, self._ws_port):
            self._started.set()
            await asyncio.Future()


# ── Entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balatro-AI replay server")
    parser.add_argument("--log-dir", default="./logs/episodes",
                        help="Directory containing episodes_*.jsonl files (default: ./logs/episodes)")
    parser.add_argument("--port", type=int, default=8765,
                        help="HTTP port (WebSocket runs on port+1, default: 8765)")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"[balatro-ai] Log directory not found: {log_dir}")
        print("             Run training first to generate episode logs.")
        raise SystemExit(1)

    server = ReplayServer(log_dir, host=args.host, port=args.port)
    server.start()

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\n[balatro-ai] Replay server stopped.")
