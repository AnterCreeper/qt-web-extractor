#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 Zhou Qiankang <wszqkzqk@qq.com>
#
# This file is part of Qt Web Extractor.
#
# Qt Web Extractor is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Qt Web Extractor is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Qt Web Extractor. If not, see <https://www.gnu.org/licenses/>.

import json
import logging
import os
import queue
import select
import signal
import socket
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    from importlib.metadata import version
    _server_version = version("qt-web-extractor")
except Exception:
    _server_version = "0.1.0dev"

from PySide6.QtCore import QTimer

from qt_web_extractor.extractor import QtWebExtractor, _ExtractionResult
from qt_web_extractor.summarier import summarize_markdown, summary_configured

log = logging.getLogger("qt-web-extractor")

_MCP_PROTOCOL_VERSION = "2024-11-05"
_MCP_MAX_RESULT_CHARS = 500000


class _ExtractRequest:
    __slots__ = ("url", "pdf", "result", "done")

    def __init__(self, url: str, pdf: bool = False):
        self.url = url
        self.pdf = pdf
        self.result: _ExtractionResult | None = None
        self.done = threading.Event()


class _Handler(BaseHTTPRequestHandler):
    extract_queue: "queue.Queue[_ExtractRequest | None]"
    timeout_s: int = 40
    api_key: str = ""
    extractor: QtWebExtractor

    def log_message(self, fmt, *args):
        log.info(fmt, *args)

    def _check_auth(self) -> bool:
        if not self.api_key:
            return True
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and auth[7:].strip() == self.api_key:
            return True
        self._send_json({"error": "unauthorized"}, 401)
        return False

    def _send_json(self, data, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, status: int = 204):
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_mcp_result(self, request_id, result):
        self._send_json({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _send_mcp_error(self, request_id, code: int, message: str, data=None):
        error = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        self._send_json({"jsonrpc": "2.0", "id": request_id, "error": error})

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        if not self._check_auth():
            return
        self._send_json({"error": "not found"}, 404)

    def _read_json_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send_json({"error": "empty body"}, 400)
            return None
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, 400)
            return None

    @staticmethod
    def _is_pdf(url: str, extractor: QtWebExtractor) -> bool:
        return extractor.detect_pdf_url(url)

    def _extract_one(self, url: str, pdf: bool = False) -> _ExtractionResult | None:
        start = time.monotonic()
        req = _ExtractRequest(url, pdf=pdf)
        self.extract_queue.put(req)
        if not req.done.wait(timeout=self.timeout_s):
            log.warning(
                "Extract timed out: url=%s pdf=%s duration_ms=%d",
                url,
                pdf,
                int((time.monotonic() - start) * 1000),
            )
            return None
        duration_ms = int((time.monotonic() - start) * 1000)
        error = req.result.error if req.result is not None else "no result"
        log.info(
            "Extract finished: url=%s pdf=%s duration_ms=%d text_chars=%d error=%s",
            url,
            pdf,
            duration_ms,
            len(req.result.text or "") if req.result is not None else 0,
            error or "",
        )
        return req.result

    @staticmethod
    def _client_connected(sock) -> bool:
        try:
            error = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if error != 0:
                return False
            try:
                data = sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                return len(data) > 0
            except BlockingIOError:
                return True
        except OSError:
            return False

    @staticmethod
    def _mcp_error_result(url: str, error: str, *, title: str = "", markdown: str = "", extra: dict | None = None) -> dict:
        response = {
            "url": url,
            "title": title,
            "markdown": markdown,
            "error": error,
        }
        if extra:
            response.update(extra)
        return {
            "content": [{"type": "text", "text": f"Error: {error}"}],
            "structuredContent": response,
            "isError": True,
        }

    @staticmethod
    def _mcp_success_result(markdown: str, response: dict, error: str = "") -> dict:
        text = markdown
        if error:
            text = f"{markdown}\n\n[warning] {error}" if markdown else f"[warning] {error}"
        return {
            "content": [{"type": "text", "text": text}],
            "structuredContent": response,
            "isError": False,
        }

    def _build_tool_response(self, name: str, result: _ExtractionResult, url: str, prompt: str) -> tuple[dict, str, str]:
        output_url = result.url or url
        if name == "summary_url":
            cancel = threading.Event()
            if not self._client_connected(self.connection):
                return (
                    {"url": output_url, "title": result.title, "markdown": "", "error": "client disconnected before LLM call",
                     "summary": True, "llm_summary_applied": False, "llm_summary_truncated": False},
                    "",
                    "client disconnected before LLM call",
                )

            result_holder = []
            exception_holder = []

            pipe_r, pipe_w = os.pipe()

            def _run_llm():
                try:
                    summary = summarize_markdown(
                        result.text or "",
                        url=output_url,
                        title=result.title,
                        prompt=prompt,
                        cancel=cancel,
                    )
                    response = {
                        "url": output_url,
                        "title": result.title,
                        "markdown": summary.markdown,
                        "error": result.error,
                        "summary": True,
                        "llm_summary_applied": summary.cleaned,
                        "llm_summary_truncated": summary.truncated,
                        **({"llm_summary_error": summary.error} if summary.error else {}),
                    }
                    result_holder.append((response, summary.markdown, result.error or ""))
                except Exception as e:
                    exception_holder.append(e)
                finally:
                    try:
                        os.write(pipe_w, b"\x00")
                    except OSError:
                        pass

            llm_thread = threading.Thread(target=_run_llm, daemon=True)
            llm_thread.start()

            try:
                while llm_thread.is_alive():
                    r, _, _ = select.select([self.connection, pipe_r], [], [])
                    if pipe_r in r:
                        break
                    cancel.set()
                    llm_thread.join(timeout=5)
                    break
            finally:
                os.close(pipe_r)
                os.close(pipe_w)

            if exception_holder:
                raise exception_holder[0]

            if result_holder:
                summary_response, summary_markdown, summary_error = result_holder[0]
                log.info(
                    "MCP summary_url summary finished: url=%s applied=%s truncated=%s output_chars=%d error=%s",
                    output_url,
                    summary_response.get("llm_summary_applied"),
                    summary_response.get("llm_summary_truncated"),
                    len(summary_markdown or ""),
                    summary_error,
                )
                return summary_response, summary_markdown, summary_error

            cancel_error = "request cancelled by client timeout"
            cancel_response = {
                "url": output_url,
                "title": result.title,
                "markdown": "",
                "error": cancel_error,
                "summary": True,
                "llm_summary_applied": False,
                "llm_summary_truncated": False,
            }
            return cancel_response, "", cancel_error

        markdown = result.text or ""
        response = {
            "url": output_url,
            "title": result.title,
            "markdown": markdown,
            "error": result.error,
        }
        return response, markdown, result.error or ""

    @staticmethod
    def _mcp_tools() -> list[dict]:
        tools = [
            {
                "name": "fetch_url",
                "description": (
                    "Extracts full content from websites, including dynamic pages, to clean Markdown."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to extract content from.",
                        }
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
                "_meta": {
                    "anthropic/maxResultSizeChars": _MCP_MAX_RESULT_CHARS,
                },
            },
        ]
        if summary_configured():
            tools.append({
                "name": "summary_url",
                "description": (
                    "Extracts a web page and returns a concise Markdown summary. "
                    "Use the optional prompt to focus on specific information."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to extract and summarize.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Optional information need to focus the summary.",
                        },
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
                "_meta": {
                    "anthropic/maxResultSizeChars": _MCP_MAX_RESULT_CHARS,
                },
            })
        return tools

    def _mcp_call_tool(self, params: dict) -> dict:
        tool_start = time.monotonic()
        name = params.get("name")
        if name == "summary_url" and not summary_configured():
            raise ValueError("summary_url is unavailable because LLM_BASE_URL or LLM_MODEL is not configured")
        if name not in {"fetch_url", "summary_url"}:
            raise ValueError("unknown tool name")

        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be an object")

        url = arguments.get("url")
        if not isinstance(url, str):
            raise ValueError("arguments.url must be a string")

        url = url.strip()
        if not url:
            raise ValueError("arguments.url is required")

        prompt = arguments.get("prompt", "")
        if prompt is None:
            prompt = ""
        if not isinstance(prompt, str):
            raise ValueError("arguments.prompt must be a string")
        prompt = prompt.strip()

        pdf = self._is_pdf(url, self.extractor)
        log.info("MCP %s: %s (pdf=%s)", name, url, pdf)
        result = self._extract_one(url, pdf=pdf)

        if result is None:
            timeout_error = "extraction timed out"
            log.warning(
                "MCP %s failed: url=%s duration_ms=%d error=%s",
                name,
                url,
                int((time.monotonic() - tool_start) * 1000),
                timeout_error,
            )
            return self._mcp_error_result(url, timeout_error)

        response, markdown, error = self._build_tool_response(name, result, url, prompt)

        if error and not markdown:
            log.warning(
                "MCP %s failed: url=%s duration_ms=%d error=%s",
                name,
                url,
                int((time.monotonic() - tool_start) * 1000),
                error,
            )
            return self._mcp_error_result(
                response["url"],
                error,
                title=response.get("title", ""),
                markdown=response.get("markdown", ""),
                extra={k: v for k, v in response.items() if k not in {"url", "title", "markdown", "error"}},
            )

        log.info(
            "MCP %s finished: url=%s duration_ms=%d output_chars=%d error=%s",
            name,
            url,
            int((time.monotonic() - tool_start) * 1000),
            len(markdown or ""),
            error,
        )
        return self._mcp_success_result(markdown, response, error)

    def _handle_mcp(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send_mcp_error(None, -32600, "Invalid Request", {"reason": "empty body"})
            return

        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._send_mcp_error(None, -32700, "Parse error")
            return

        if not isinstance(body, dict):
            self._send_mcp_error(None, -32600, "Invalid Request")
            return

        has_id = "id" in body
        request_id = body.get("id")
        method = body.get("method")
        params = body.get("params", {})

        if body.get("jsonrpc") != "2.0" or not isinstance(method, str) or not method:
            self._send_mcp_error(request_id if has_id else None, -32600, "Invalid Request")
            return

        if not isinstance(params, dict):
            if has_id:
                self._send_mcp_error(request_id, -32602, "Invalid params", {"reason": "params must be an object"})
            else:
                self._send_empty()
            return

        # Ignore JSON-RPC notifications unless explicitly needed.
        if not has_id:
            self._send_empty()
            return

        if method == "initialize":
            self._send_mcp_result(
                request_id,
                {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "qt-web-extractor",
                        "version": _server_version,
                    },
                },
            )
            return

        if method == "ping":
            self._send_mcp_result(request_id, {})
            return

        if method == "notifications/initialized":
            self._send_mcp_result(request_id, {})
            return

        if method == "tools/list":
            self._send_mcp_result(request_id, {"tools": self._mcp_tools()})
            return

        if method == "tools/call":
            try:
                result = self._mcp_call_tool(params)
            except ValueError as e:
                self._send_mcp_error(request_id, -32602, "Invalid params", {"reason": str(e)})
                return
            self._send_mcp_result(request_id, result)
            return

        self._send_mcp_error(request_id, -32601, "Method not found")

    def do_POST(self):
        if not self._check_auth():
            return

        if self.path in ("/mcp", "/mcp/"):
            self._handle_mcp()
            return

        body = self._read_json_body()
        if body is None:
            return

        # Open WebUI external web loader format: POST / with {"urls": [...]}
        if self.path in ("/", "") and "urls" in body:
            urls = body.get("urls", [])
            if not isinstance(urls, list) or not urls:
                self._send_json({"error": "urls must be a non-empty array"}, 400)
                return

            log.info("Batch extract request: %d URLs", len(urls))
            documents = []
            for url in urls:
                url = url.strip()
                if not url:
                    continue
                pdf = self._is_pdf(url, self.extractor)
                log.info("  -> %s (pdf=%s)", url, pdf)
                result = self._extract_one(url, pdf=pdf)
                if result is None:
                    documents.append({
                        "page_content": "",
                        "metadata": {"source": url, "error": "extraction timed out"},
                    })
                else:
                    documents.append({
                        "page_content": result.text,
                        "metadata": {
                            "source": result.url or url,
                            "title": result.title,
                            **({"error": result.error} if result.error else {}),
                        },
                    })
            self._send_json(documents)
            return

        # Legacy single-URL format: POST /extract with {"url": "..."}
        if self.path == "/extract":
            url = body.get("url", "").strip()
            if not url:
                self._send_json({"error": "url is required"}, 400)
                return

            pdf = body.get("pdf", None)
            if pdf is None:
                pdf = self._is_pdf(url, self.extractor)

            log.info("Extract request: %s (pdf=%s)", url, pdf)
            result = self._extract_one(url, pdf=pdf)

            if result is None:
                self._send_json({"error": "extraction timed out"}, 504)
                return

            self._send_json(result.to_dict())
            return

        self._send_json({"error": "not found"}, 404)


def serve(
    host: str = "127.0.0.1",
    port: int = 8766,
    timeout_ms: int = 30000,
    user_agent: str | None = None,
    api_key: str = "",
    proxy: str | None = None,
):
    """Start the extraction server. Blocks forever (runs Qt event loop)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    extractor = QtWebExtractor(timeout_ms=timeout_ms, user_agent=user_agent, proxy=proxy)
    app = extractor._app

    extract_queue: queue.Queue[_ExtractRequest | None] = queue.Queue()

    _Handler.extract_queue = extract_queue
    _Handler.timeout_s = timeout_ms // 1000 + 10
    _Handler.api_key = api_key
    _Handler.extractor = extractor

    server = HTTPServer((host, port), _Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    log.info("Listening on http://%s:%d", host, port)
    log.info(
        "  timeout: %dms, auth: %s, proxy: %s",
        timeout_ms,
        "on" if api_key else "off",
        extractor.proxy_summary,
    )

    shutting_down = False

    def handle_signal(*_):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        log.info("Shutting down...")
        extract_queue.put(None)  # poison pill
        server.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Qt WebEngine must run on the main thread; poll queue from Qt event loop.
    poll_timer = QTimer()
    poll_timer.setInterval(50)

    def poll_queue():
        try:
            req = extract_queue.get_nowait()
        except queue.Empty:
            return
        if req is None:
            poll_timer.stop()
            app.quit()
            return
        result = extractor.extract_pdf(req.url) if req.pdf else extractor.extract(req.url)
        req.result = result
        req.done.set()

    poll_timer.timeout.connect(poll_queue)
    poll_timer.start()

    app.exec()

    server.server_close()
    server_thread.join(timeout=2)
    del extractor
    log.info("Shutdown complete")
