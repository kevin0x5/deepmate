"""Minimal Enterprise WeChat AI bot WebSocket client."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse


WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_WEBSOCKET_FRAME_BYTES = 1_000_000
MAX_WEBSOCKET_MESSAGE_BYTES = 2_000_000


class WeComPayloadError(ValueError):
    """A complete WebSocket message was not a usable WeCom JSON object."""


class WeComProtocolError(ValueError):
    """The WebSocket stream is invalid and the connection should be reset."""


@dataclass(frozen=True, slots=True)
class WeComClientConfig:
    """WeCom websocket credentials and endpoint."""

    bot_id: str
    secret: str
    url: str = "wss://openws.work.weixin.qq.com"


class WeComWsClient:
    """Small blocking WebSocket client for WeCom AI bot messages."""

    def __init__(self, config: WeComClientConfig) -> None:
        self.config = config
        self._sock: ssl.SSLSocket | socket.socket | None = None
        self._send_lock = threading.Lock()

    def connect(self) -> None:
        parsed = urlparse(self.config.url)
        if parsed.scheme not in {"ws", "wss"}:
            raise ValueError(f"unsupported WeCom websocket scheme: {parsed.scheme}")
        host = parsed.hostname or ""
        if not host:
            raise ValueError("WeCom websocket host is required")
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"
        raw = socket.create_connection((host, port), timeout=30)
        if parsed.scheme == "wss":
            sock: ssl.SSLSocket | socket.socket = ssl.create_default_context().wrap_socket(
                raw,
                server_hostname=host,
            )
        else:
            sock = raw
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        headers = [
            f"GET {path} HTTP/1.1",
            f"Host: {host}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
            "",
            "",
        ]
        sock.sendall("\r\n".join(headers).encode("ascii"))
        response = _read_http_response(sock)
        expected_accept = base64.b64encode(
            hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()
        ).decode("ascii")
        if " 101 " not in response.splitlines()[0]:
            raise RuntimeError(f"WeCom websocket upgrade failed: {response.splitlines()[0]}")
        if f"sec-websocket-accept: {expected_accept.lower()}" not in response.lower():
            raise RuntimeError("WeCom websocket upgrade returned invalid accept key")
        self._sock = sock

    def subscribe(self) -> None:
        self.send_json(
            {
                "cmd": "aibot_subscribe",
                "bot_id": self.config.bot_id,
                "botid": self.config.bot_id,
                "secret": self.config.secret,
            }
        )

    def send_json(self, payload: dict[str, object]) -> None:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self._send_frame(text.encode("utf-8"), opcode=0x1)

    def ping(self) -> None:
        self.send_json({"cmd": "ping", "timestamp": int(time.time())})

    def recv_json(self) -> dict[str, object]:
        payload = self._recv_frame()
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WeComPayloadError("WeCom websocket payload must be JSON") from exc
        if not isinstance(decoded, dict):
            raise WeComPayloadError("WeCom websocket payload must be a JSON object")
        return decoded

    def recv_json_timeout(self, timeout: float) -> dict[str, object] | None:
        """Receive one JSON payload with a temporary socket timeout."""
        if self._sock is None:
            raise RuntimeError("WeCom websocket is not connected")
        sock = self._sock
        previous_timeout = sock.gettimeout()
        try:
            sock.settimeout(max(0.0, timeout))
            return self.recv_json()
        except socket.timeout:
            return None
        finally:
            sock.settimeout(previous_timeout)

    def close(self) -> None:
        with self._send_lock:
            sock = self._sock
            self._sock = None
        if sock is None:
            return
        try:
            sock.close()
        except OSError:
            pass

    def _send_frame(self, payload: bytes, *, opcode: int) -> None:
        with self._send_lock:
            if self._sock is None:
                raise RuntimeError("WeCom websocket is not connected")
            mask = os.urandom(4)
            length = len(payload)
            header = bytearray([0x80 | opcode])
            if length < 126:
                header.append(0x80 | length)
            elif length <= 0xFFFF:
                header.append(0x80 | 126)
                header.extend(struct.pack("!H", length))
            else:
                header.append(0x80 | 127)
                header.extend(struct.pack("!Q", length))
            masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
            self._sock.sendall(bytes(header) + mask + masked)

    def _recv_frame(self) -> bytes:
        if self._sock is None:
            raise RuntimeError("WeCom websocket is not connected")
        fragments: list[bytes] = []
        total = 0
        fragmented_opcode: int | None = None
        while True:
            first = _recv_exact(self._sock, 2)
            fin = bool(first[0] & 0x80)
            if first[0] & 0x70:
                raise WeComProtocolError(
                    "WeCom websocket frame uses unsupported extensions"
                )
            opcode = first[0] & 0x0F
            masked = bool(first[1] & 0x80)
            length = first[1] & 0x7F
            if length == 126:
                length = struct.unpack("!H", _recv_exact(self._sock, 2))[0]
            elif length == 127:
                length = struct.unpack("!Q", _recv_exact(self._sock, 8))[0]
            if length > MAX_WEBSOCKET_FRAME_BYTES:
                raise WeComProtocolError(
                    "WeCom websocket frame is too large "
                    f"({length} bytes > {MAX_WEBSOCKET_FRAME_BYTES})"
                )
            mask = _recv_exact(self._sock, 4) if masked else b""
            payload = _recv_exact(self._sock, length)
            if masked:
                payload = bytes(
                    byte ^ mask[index % 4] for index, byte in enumerate(payload)
                )
            if opcode == 0x8:
                raise EOFError("WeCom websocket closed")
            if opcode == 0x9:
                if not fin or len(payload) > 125:
                    raise WeComProtocolError("WeCom websocket ping frame is invalid")
                self._send_frame(payload, opcode=0xA)
                continue
            if opcode == 0xA:
                if not fin or len(payload) > 125:
                    raise WeComProtocolError("WeCom websocket pong frame is invalid")
                continue
            if opcode in {0x1, 0x2}:
                if fragmented_opcode is not None:
                    raise WeComProtocolError(
                        "WeCom websocket started a new message before completing "
                        "the previous fragmented message"
                    )
                if fin:
                    return payload
                fragmented_opcode = opcode
                fragments = [payload]
                total = len(payload)
            elif opcode == 0x0:
                if fragmented_opcode is None:
                    raise WeComProtocolError(
                        "WeCom websocket continuation without a message"
                    )
                fragments.append(payload)
                total += len(payload)
                if total > MAX_WEBSOCKET_MESSAGE_BYTES:
                    raise WeComProtocolError(
                        "WeCom websocket message is too large "
                        f"({total} bytes > {MAX_WEBSOCKET_MESSAGE_BYTES})"
                    )
                if fin:
                    return b"".join(fragments)
            else:
                raise WeComProtocolError(
                    f"WeCom websocket opcode is unsupported: 0x{opcode:x}"
                )


def _read_http_response(sock: socket.socket) -> str:
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > 64_000:
            raise RuntimeError("websocket upgrade response is too large")
    return bytes(data).decode("iso-8859-1", errors="replace")


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError("websocket connection closed")
        data.extend(chunk)
    return bytes(data)
