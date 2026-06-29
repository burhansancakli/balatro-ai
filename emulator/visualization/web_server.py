"""HTTP + WebSocket server for the browser dashboard.

Two threads, two ports:
  HTTP  (port)      — serves index.html / style.css / app.js
  WebSocket (port+1)— real-time game state push / play-mode actions

The split eliminates routing conflicts between HTTP and WebSocket upgrades.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import queue
import threading
from pathlib import Path
from typing import Any

STATIC_DIR = Path(__file__).parent / "static"


class WebServer:
    def __init__(
        self,
        state_queue: "queue.Queue[dict]",
        action_queue: "queue.Queue[dict]",
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self._state_queue = state_queue
        self._action_queue = action_queue
        self._host = host
        self._port = port       # HTTP
        self._ws_port = port + 1  # WebSocket
        self._connections: set[Any] = set()
        self._last_snapshot: str | None = None
        self._started = threading.Event()

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
            # Replay last state immediately so late-joining browsers see something
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
            finally:
                self._connections.discard(ws)

        async with websockets.serve(handler, self._host, self._ws_port):
            self._started.set()
            # Broadcast loop — runs for the lifetime of the WS server
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
                    dead: set[Any] = set()
                    for ws in list(self._connections):
                        try:
                            await ws.send(payload)
                        except Exception:
                            dead.add(ws)
                    self._connections -= dead

                await asyncio.sleep(0.04)  # ~25 fps
