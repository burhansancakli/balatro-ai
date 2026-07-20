"""
Reverse proxy for Balatro AI — Cloudflare Tunnel compatible.
Single port (8770) serves both HTTP (→8765) and WebSocket (→8766).

Cloudflare config:
  - hostname: balatro.burhansancakli.com
    service: http://localhost:8770
"""
import asyncio
import socket

import websockets
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

HTTP_UPSTREAM = ("127.0.0.1", 8765)
WS_UPSTREAM = "ws://127.0.0.1:8766"
PROXY_PORT = 8770


def _forward_http(method: str, path: str, req_headers: Headers) -> tuple[int, Headers, bytes]:
    """Forward an HTTP request to the upstream HTTP server and return (status, headers, body)."""
    try:
        with socket.create_connection(HTTP_UPSTREAM, timeout=10) as s:
            lines = [f"{method} {path} HTTP/1.1"]
            for k, v in req_headers.raw_items():
                lines.append(f"{k}: {v}")
            lines.append("")
            lines.append("")
            s.sendall("\r\n".join(lines).encode())

            resp = b""
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                resp += chunk
                if b"\r\n\r\n" in resp:
                    hdr_end = resp.index(b"\r\n\r\n") + 4
                    body_len = 0
                    for line in resp[:hdr_end].decode("latin-1", errors="replace").split("\r\n"):
                        if line.lower().startswith("content-length:"):
                            body_len = int(line.split(":", 1)[1].strip())
                    if len(resp) >= hdr_end + body_len:
                        break

            # Parse status
            status_line = resp.split(b"\r\n", 1)[0].decode("latin-1", errors="replace")
            parts = status_line.split(" ")
            status_code = int(parts[1]) if len(parts) > 1 else 200

            # Parse headers
            hdr_section = resp[:hdr_end].decode("latin-1", errors="replace")
            resp_headers = Headers()
            for line in hdr_section.split("\r\n")[1:]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    resp_headers[k.strip()] = v.strip()

            body = resp[hdr_end:]
            return status_code, resp_headers, body
    except Exception as e:
        return 502, Headers([("Content-Type", "text/plain")]), f"Proxy error: {e}".encode()


async def process_request(
    connection: websockets.ServerConnection,
    request: Request,
) -> Response | None:
    """
    websockets callback: if NOT a WS upgrade, serve HTTP inline and
    return a Response to short-circuit the WS handshake.
    """
    upgrade = request.headers.get("Upgrade", "").lower()
    if upgrade == "websocket":
        return None  # let websockets handle the upgrade

    # Regular HTTP — forward to upstream
    status, headers, body = _forward_http("GET", request.path, request.headers)
    return Response(status, "OK" if status == 200 else "Error", headers, body)


async def ws_bridge(client: websockets.ServerConnection) -> None:
    """Bridge a browser WS connection to the upstream WS server."""
    try:
        upstream = await websockets.connect(WS_UPSTREAM)
    except Exception:
        await client.close(1011, "upstream unavailable")
        return

    async def c2u():
        try:
            async for msg in client:
                await upstream.send(msg)
        except Exception:
            pass
        finally:
            try:
                await upstream.close()
            except Exception:
                pass

    async def u2c():
        try:
            async for msg in upstream:
                await client.send(msg)
        except Exception:
            pass
        finally:
            try:
                await client.close()
            except Exception:
                pass

    await asyncio.gather(c2u(), u2c())


async def main() -> None:
    print(f"[balatro-proxy] HTTP upstream → {HTTP_UPSTREAM}")
    print(f"[balatro-proxy] WS upstream  → {WS_UPSTREAM}")
    print(f"[balatro-proxy] Listening on 0.0.0.0:{PROXY_PORT} (HTTP + WS)")

    async with websockets.serve(
        ws_bridge,
        "0.0.0.0",
        PROXY_PORT,
        process_request=process_request,
    ):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
