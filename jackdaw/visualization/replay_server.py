"""Replay server — serves pre-recorded JSONL episode logs to the browser.

Architecture (same two-port design as web_server.py):
  HTTP  (port)      — static files + /api/episodes list
  WebSocket (port+1)— replay control (browser sends nav commands, server
                      sends episode state snapshots)

WebSocket protocol (JSON messages):
  Browser → server:
    {"cmd": "list"}                          — get episode index
    {"cmd": "load", "episode": N}            — load episode N (0-indexed)
    {"cmd": "step", "index": I}              — seek to step I
    {"cmd": "next"}                          — advance one step
    {"cmd": "prev"}                          — go back one step

  Server → browser:
    {"type": "episodes", "episodes": [...]}  — episode list
    {"type": "state", ...}                   — step state snapshot
"""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
from pathlib import Path
from typing import Any

STATIC_DIR = Path(__file__).parent / "static"


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


def _episode_summary(ep: dict, idx: int) -> dict:
    return {
        "index": idx,
        "episode": ep.get("episode", idx),
        "seed": ep.get("seed", ""),
        "ante_reached": ep.get("ante_reached", 0),
        "rounds_beaten": ep.get("rounds_beaten", 0),
        "won": ep.get("won", False),
        "total_reward": ep.get("total_reward", 0.0),
        "steps": ep.get("steps", 0),
        "hands_played": len(ep.get("hand_history", [])),
    }


def _build_step_state(ep: dict, step_index: int) -> dict:
    """Build a frontend-compatible state snapshot for action_log step I."""
    action_log = ep.get("action_log", [])
    hand_history = ep.get("hand_history", [])
    total_steps = len(action_log)

    # Accumulate hand history up to current step
    # Map each hand_history entry to the action_log step where it was played
    play_step_map: dict[int, dict] = {}
    play_count = 0
    for i, entry in enumerate(action_log):
        if entry.get("type") == "play":
            if play_count < len(hand_history):
                play_step_map[i] = hand_history[play_count]
            play_count += 1

    visible_hands = [h for i, h in play_step_map.items() if i <= step_index]

    current = action_log[step_index] if step_index < total_steps else {}

    # Build joker list from the most recent hand before this step
    jokers: list[str] = []
    for i in range(step_index, -1, -1):
        if i in play_step_map:
            jokers = play_step_map[i].get("jokers", [])
            break

    return {
        "type": "state",
        "replay": True,
        "episode_index": ep.get("episode", 0),
        "seed": ep.get("seed", ""),
        "step_index": step_index,
        "total_steps": total_steps,
        "won": ep.get("won", False),
        "ante_reached": ep.get("ante_reached", 0),
        "rounds_beaten": ep.get("rounds_beaten", 0),
        # Current action context
        "current_action": current,
        "current_jokers": jokers,
        # Full accumulated hand history
        "hand_history": visible_hands,
        # Full action log for timeline
        "action_log": action_log,
    }


class ReplayServer:
    """HTTP + WebSocket server for replay mode."""

    def __init__(
        self,
        jsonl_path: Path,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self._path = jsonl_path
        self._host = host
        self._port = port
        self._ws_port = port + 1
        self._episodes: list[dict] = []
        self._started = threading.Event()

    def start(self) -> None:
        self._episodes = _load_jsonl(self._path)
        print(f"\n  [jackdaw] Loaded {len(self._episodes)} episode(s) from {self._path}")
        threading.Thread(target=self._run_http, daemon=True, name="jackdaw-replay-http").start()
        threading.Thread(target=self._run_ws,   daemon=True, name="jackdaw-replay-ws").start()
        self._started.wait(timeout=8)
        print(f"  [jackdaw] Replay dashboard → http://{self._host}:{self._port}/?mode=replay\n")

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _run_http(self) -> None:
        static_dir = STATIC_DIR
        ws_port = self._ws_port
        episodes = self._episodes

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

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    def _run_ws(self) -> None:
        try:
            asyncio.run(self._serve_ws())
        except Exception as exc:
            print(f"\n[jackdaw-replay-ws] CRASH: {exc}\n")
            self._started.set()

    async def _serve_ws(self) -> None:
        try:
            import websockets
        except ImportError:
            print("\n[jackdaw] websockets package missing — run: pip install 'jackdaw[viz]'\n")
            self._started.set()
            return

        episodes = self._episodes

        async def handler(ws: Any) -> None:
            # Per-connection state
            current_ep: int = 0
            current_step: int = 0

            async def send_episodes() -> None:
                summary = [_episode_summary(ep, i) for i, ep in enumerate(episodes)]
                await ws.send(json.dumps({"type": "episodes", "episodes": summary}))

            async def send_step(ep_idx: int, step: int) -> None:
                ep = episodes[ep_idx]
                snap = _build_step_state(ep, step)
                await ws.send(json.dumps(snap, default=str))

            # Send episode list on connect
            try:
                await send_episodes()
                if episodes:
                    await send_step(0, 0)
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
                            current_ep = ep_idx
                            current_step = 0
                            await send_step(current_ep, current_step)

                    elif cmd == "step":
                        step = int(msg.get("index", 0))
                        total = len(episodes[current_ep].get("action_log", []))
                        current_step = max(0, min(step, total - 1))
                        await send_step(current_ep, current_step)

                    elif cmd == "next":
                        total = len(episodes[current_ep].get("action_log", []))
                        if current_step < total - 1:
                            current_step += 1
                        await send_step(current_ep, current_step)

                    elif cmd == "prev":
                        if current_step > 0:
                            current_step -= 1
                        await send_step(current_ep, current_step)

                except Exception:
                    pass

        async with websockets.serve(handler, self._host, self._ws_port):
            self._started.set()
            await asyncio.Future()  # run forever
