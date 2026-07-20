"""HTTP + WebSocket server for the browser dashboard.

Two threads, two ports:
  HTTP  (port)      — serves index.html / style.css / app.js
  WebSocket (port+1)— real-time game state push / play-mode actions

Per-connection sessions: each WebSocket client gets its own independent
game state, state queue, and action queue — so multiple players can play
simultaneously without affecting each other.
"""
from __future__ import annotations

import asyncio
import http.server
import json
import queue
import threading
import uuid
from pathlib import Path
from typing import Any

STATIC_DIR = Path(__file__).parent / "static"


class Session:
    """One player's isolated game session."""

    def __init__(self) -> None:
        self.id: str = uuid.uuid4().hex[:8]
        self.state_queue: queue.Queue = queue.Queue(maxsize=2)
        self.action_queue: queue.Queue[dict] = queue.Queue()
        self.last_snapshot: dict | None = None
        self._thread: threading.Thread | None = None

    def start_game_loop(
        self,
        back_key: str,
        stake: int,
        simplified: bool = False,
    ) -> None:
        """Spawn a background thread running the play-mode game loop."""
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(back_key, stake, simplified),
            daemon=True,
            name=f"game-{self.id}",
        )
        self._thread.start()

    def _run_loop(self, back_key: str, stake: int, simplified: bool) -> None:
        """Game loop — runs in its own thread with its own queues."""
        import random as _rng
        import time
        import traceback

        try:
            from emulator.engine.actions import (
                GamePhase,
                get_legal_actions,
            )
            from emulator.engine.game import IllegalActionError, step
            from emulator.engine.run_init import initialize_run
            from emulator.visualization.observer import deserialize_action, serialize_state

            if simplified:
                from emulator.engine.simplified import (
                    apply_to_run,
                    apply_to_shop,
                    filter_legal_actions,
                )

            seed = f"SEED{_rng.randint(1000, 9999)}"
            gs = initialize_run(back_key, stake, seed)
            if simplified:
                apply_to_run(gs)

            def _emit(gs: dict, legal: list | None = None) -> None:
                snapshot = serialize_state(gs, legal_actions=legal)
                self.last_snapshot = snapshot
                try:
                    self.state_queue.put_nowait(json.dumps(snapshot, default=str))
                except queue.Full:
                    try:
                        self.state_queue.get_nowait()
                        self.state_queue.put_nowait(json.dumps(snapshot, default=str))
                    except queue.Empty:
                        pass

            def _legal(gs: dict) -> list:
                actions = get_legal_actions(gs)
                return filter_legal_actions(actions) if simplified else actions

            history: list[str] = []
            actions_taken = 0

            while True:
                phase = gs.get("phase")
                if phase == GamePhase.GAME_OVER or (gs.get("won") and phase == GamePhase.SHOP):
                    gs["actions_taken"] = actions_taken
                    _emit(gs, legal=[{"type": "NewGame", "label": "New Game"}])
                    while True:
                        try:
                            cmd = self.action_queue.get(timeout=60)
                        except queue.Empty:
                            return  # Session timed out
                        if isinstance(cmd, dict) and cmd.get("type") == "NewGame":
                            seed = f"SEED{_rng.randint(1000, 9999)}"
                            gs = initialize_run(back_key, stake, seed)
                            if simplified:
                                apply_to_run(gs)
                            actions_taken = 0
                            break
                        elif isinstance(cmd, dict) and cmd.get("type") == "__close__":
                            return  # Client disconnected
                    continue

                legal = _legal(gs)
                if not legal:
                    break

                _emit(gs, legal=legal)

                # Wait for action from this client
                try:
                    raw = self.action_queue.get(timeout=120)
                except queue.Empty:
                    return  # Session timed out

                if isinstance(raw, str):
                    action_data = json.loads(raw)
                else:
                    action_data = raw

                if isinstance(action_data, dict) and action_data.get("type") == "__close__":
                    return  # Client disconnected

                try:
                    action = deserialize_action(action_data)
                    step(gs, action)
                    if simplified and gs.get("phase") == GamePhase.SHOP:
                        apply_to_shop(gs)
                    actions_taken += 1
                except (ValueError, IllegalActionError):
                    continue

            gs["actions_taken"] = actions_taken
            _emit(gs)
        except Exception as e:
            print(f"\n[emulator-session-{self.id}] GAME LOOP CRASHED: {e}")
            traceback.print_exc()

    def send(self, data: dict) -> None:
        """Queue an action from the client."""
        self.action_queue.put_nowait(data)

    def close(self) -> None:
        """Signal the game loop to stop."""
        try:
            self.action_queue.put_nowait({"type": "__close__"})
        except queue.Full:
            pass


class WebServer:
    def __init__(
        self,
        state_queue: "queue.Queue[dict]",
        action_queue: "queue.Queue[dict]",
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        # Legacy single-session queues (used by watch mode / replay)
        self._state_queue = state_queue
        self._action_queue = action_queue
        self._host = host
        self._port = port       # HTTP
        self._ws_port = port + 1  # WebSocket
        self._connections: set[Any] = set()
        self._last_snapshot: str | None = None
        self._started = threading.Event()
        # Per-session mode (play mode)
        self._sessions: dict[Any, Session] = {}  # ws -> Session
        self._game_params: dict[str, Any] = {}  # set by main.py before start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        threading.Thread(target=self._run_http, daemon=True, name="emulator-http").start()
        threading.Thread(target=self._run_ws,   daemon=True, name="emulator-ws").start()
        self._started.wait(timeout=8)
        print(f"\n  [emulator] Dashboard → http://{self._host}:{self._port}\n")

    def stop(self) -> None:
        pass  # daemon threads die with the main process

    # ------------------------------------------------------------------
    # HTTP thread — serves static files
    # ------------------------------------------------------------------
    def _run_http(self) -> None:
        try:
            self._run_http_inner()
        except Exception as exc:
            print(f"\n[emulator-http] CRASH: {exc}\n")

    def _run_http_inner(self) -> None:
        static_dir = STATIC_DIR
        ws_port = self._ws_port

        class Handler(http.server.BaseHTTPRequestHandler):
            _TYPES = {
                "/style.css": "text/css",
                "/app.js":    "application/javascript",
            }

            def do_GET(self) -> None:
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    # Inject WS port so app.js can connect to the right port
                    content = (static_dir / "index.html").read_text(encoding="utf-8")
                    content = content.replace("__WS_PORT__", str(ws_port))
                    data = content.encode()
                    ct = "text/html; charset=utf-8"
                elif path in self._TYPES:
                    data = (static_dir / path.lstrip("/")).read_bytes()
                    ct = self._TYPES[path]
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_: Any) -> None:
                pass  # silence request logs

        server = http.server.HTTPServer((self._host, self._port), Handler)
        server.serve_forever()

    # ------------------------------------------------------------------
    # WebSocket thread
    # ------------------------------------------------------------------
    def _run_ws(self) -> None:
        try:
            asyncio.run(self._serve_ws())
        except Exception as exc:
            print(f"\n[emulator-ws] CRASH: {exc}\n")
            self._started.set()  # unblock main thread

    async def _serve_ws(self) -> None:
        try:
            import websockets
        except ImportError:
            print("\n[emulator] websockets package missing — run: pip install 'emulator[viz]'\n")
            self._started.set()
            return

        async def handler(ws: Any) -> None:
            self._connections.add(ws)

            # Determine mode from first message or default to legacy
            is_play_mode = self._game_params.get("mode") == "play"

            if is_play_mode:
                # Create a new session for this connection
                session = Session()
                self._sessions[ws] = session
                session.start_game_loop(
                    back_key=self._game_params.get("back_key", "b_red"),
                    stake=self._game_params.get("stake", 1),
                    simplified=self._game_params.get("simplified", False),
                )
                print(f"  [emulator] Player connected → session {session.id}")

                # Forward state from session to this WS client
                async def forward_state() -> None:
                    while True:
                        try:
                            snap = session.state_queue.get_nowait()
                            await ws.send(snap)
                        except queue.Empty:
                            await asyncio.sleep(0.04)
                        except Exception:
                            break

                # Forward actions from this WS client to session
                async def receive_actions() -> None:
                    try:
                        async for msg in ws:
                            try:
                                data = json.loads(msg)
                                session.send(data)
                            except Exception:
                                pass
                    except Exception:
                        pass

                forward_task = asyncio.create_task(forward_state())
                receive_task = asyncio.create_task(receive_actions())

                # Wait until either finishes (client disconnect or error)
                done, pending = await asyncio.wait(
                    [forward_task, receive_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()

                # Clean up session
                session.close()
                self._sessions.pop(ws, None)
                print(f"  [emulator] Player disconnected → session {session.id}")

            else:
                # Legacy mode (watch/replay): broadcast shared state
                if self._last_snapshot:
                    try:
                        await ws.send(self._last_snapshot)
                    except Exception:
                        pass
                try:
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                            self._action_queue.put_nowait(data)
                        except Exception:
                            pass
                except Exception:
                    pass

            self._connections.discard(ws)

        async with websockets.serve(handler, self._host, self._ws_port):
            self._started.set()
            # Broadcast loop for legacy mode
            while True:
                batch: list[str] = []
                while True:
                    try:
                        snap = self._state_queue.get_nowait()
                        batch.append(json.dumps(snap, default=str))
                    except queue.Empty:
                        break

                if batch:
                    payload = batch[-1]
                    self._last_snapshot = payload
                    # Only broadcast to legacy (non-session) connections
                    dead: set[Any] = set()
                    for ws in list(self._connections):
                        if ws not in self._sessions:
                            try:
                                await ws.send(payload)
                            except Exception:
                                dead.add(ws)
                    self._connections -= dead

                await asyncio.sleep(0.04)  # ~25 fps
