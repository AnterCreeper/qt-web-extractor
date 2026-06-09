#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

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
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("qt-web-extractor")

_CLEANUP_RULES_PATH = Path(__file__).with_name("llm_cleanup.txt")
_DEFAULT_LLM_TIMEOUT = 60
_DEFAULT_SUMMARY_MAX_CHARS = 16384
_DEFAULT_SUMMARY_MAX_TOKENS = 2048


@dataclass(frozen=True)
class _SummaryResult:
    markdown: str
    cleaned: bool = False
    error: str = ""
    truncated: bool = False


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        log.warning("Ignoring invalid %s=%r", name, value)
        return default


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def _env_int_first(*names: str, default: int) -> int:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return _env_int(name, default)
    return default


def _build_chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def summary_configured() -> bool:
    return bool(_env_first("LLM_BASE_URL") and _env_first("LLM_MODEL"))


def _summary_system_prompt() -> str:
    return (
        "You extract and summarize useful web page content for an AI agent.\n\n"
        "Rules:\n"
        "- Prioritize the user's information need when provided.\n"
        "- Focus on purpose, setup, usage, configuration, warnings, and important links.\n"
        "- Preserve key links and image Markdown links when relevant.\n"
        "- Remove navigation, duplicated menus, login/signup chrome, footers, and boilerplate.\n"
        "- Do not invent facts.\n"
        "- Do not include reasoning, analysis, or channel markers.\n"
        "- Return concise Markdown only."
    )


def _summary_user_prompt(markdown: str, *, url: str = "", title: str = "", prompt: str = "") -> str:
    parts = ["Summarize or extract the useful content from this web page Markdown."]
    if url:
        parts.append(f"Source URL: {url}")
    if title:
        parts.append(f"Title: {title}")
    if prompt:
        parts.append(f"User information need: {prompt}")
    parts.append("Markdown:")
    parts.append(markdown)
    return "\n\n".join(parts)


def _call_chat_completion(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens_env: str,
    default_max_tokens: int,
) -> tuple[str, str]:
    base_url = _env_first("LLM_BASE_URL")
    model = _env_first("LLM_MODEL")
    if not base_url or not model:
        return "", "LLM_BASE_URL or LLM_MODEL is not configured"

    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(_env_first("LLM_TEMPERATURE", default="0") or 0),
    }

    max_tokens = _env_int(max_tokens_env, default_max_tokens)
    if max_tokens > 0:
        request_payload["max_tokens"] = max_tokens

    body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        _build_chat_completions_url(base_url),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    api_key = _env_first("LLM_API_KEY")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    timeout = _env_int_first("LLM_TIMEOUT", default=_DEFAULT_LLM_TIMEOUT)
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            response_payload = json.loads(resp.read().decode(charset, errors="replace"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        log.warning(
            "LLM chat completion HTTP error: duration_ms=%d status=%s detail=%s",
            int((time.monotonic() - start) * 1000),
            e.code,
            detail,
        )
        return "", f"LLM HTTP {e.code}: {detail}"
    except Exception as e:
        log.warning(
            "LLM chat completion failed: duration_ms=%d error=%s",
            int((time.monotonic() - start) * 1000),
            e,
        )
        return "", f"LLM failed: {e}"

    content = _strip_model_artifacts(_extract_message_content(response_payload))
    if not content:
        log.warning(
            "LLM chat completion returned empty content: duration_ms=%d input_chars=%d",
            int((time.monotonic() - start) * 1000),
            len(user_prompt),
        )
        return "", "LLM returned empty content"
    log.info(
        "LLM chat completion finished: duration_ms=%d input_chars=%d output_chars=%d",
        int((time.monotonic() - start) * 1000),
        len(user_prompt),
        len(content),
    )
    return content, ""


def _extract_message_content(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        return content.strip() if isinstance(content, str) else ""
    text = first.get("text")
    return text.strip() if isinstance(text, str) else ""


def _read_cleanup_rules() -> list[tuple[str, str]]:
    rules_path = Path(os.environ.get("LLM_OUTPUT_CLEANUP_RULES") or _CLEANUP_RULES_PATH)
    try:
        lines = rules_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    rules: list[tuple[str, str]] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        action, value = line.split("=", 1)
        action = action.strip()
        value = value.strip()
        if action and value:
            rules.append((action, value))
    return rules


def _strip_model_artifacts(text: str) -> str:
    text = text.strip()
    fences: list[str] = []
    for action, value in _read_cleanup_rules():
        if action == "split_after" and value in text:
            text = text.rsplit(value, 1)[1]
        elif action == "remove":
            text = text.replace(value, "")
        elif action == "regex_remove":
            try:
                text = re.sub(value, "", text)
            except re.error as e:
                log.warning("Ignoring invalid cleanup regex %r: %s", value, e)
        elif action == "fence":
            fences.append(value)

    text = text.strip()
    for fence in fences:
        if text.startswith(fence) and text.endswith("```"):
            text = text[len(fence): -len("```")].strip()
            break
    return text


def summarize_markdown(
    markdown: str,
    *,
    url: str = "",
    title: str = "",
    prompt: str = "",
) -> _SummaryResult:
    """Summarize Markdown through the configured OpenAI-compatible endpoint.

    Long inputs are truncated before the LLM call. This keeps MCP calls bounded
    and avoids introducing multi-turn chunking into the first summary tool.
    """
    markdown = markdown or ""
    if not markdown.strip():
        return _SummaryResult(markdown=markdown)

    max_chars = _env_int("LLM_SUMMARY_MAX_CHARS", _DEFAULT_SUMMARY_MAX_CHARS)
    truncated = max_chars > 0 and len(markdown) > max_chars
    llm_input = markdown[:max_chars] if truncated else markdown
    start = time.monotonic()

    summary, error = _call_chat_completion(
        system_prompt=_summary_system_prompt(),
        user_prompt=_summary_user_prompt(llm_input, url=url, title=title, prompt=prompt),
        max_tokens_env="LLM_SUMMARY_MAX_TOKENS",
        default_max_tokens=_DEFAULT_SUMMARY_MAX_TOKENS,
    )
    if error:
        log.warning(
            "LLM summary failed: duration_ms=%d raw_chars=%d input_chars=%d truncated=%s error=%s",
            int((time.monotonic() - start) * 1000),
            len(markdown),
            len(llm_input),
            truncated,
            error,
        )
        return _SummaryResult(markdown=llm_input, error=f"LLM summary {error}", truncated=truncated)
    log.info(
        "LLM summary finished: duration_ms=%d raw_chars=%d input_chars=%d output_chars=%d truncated=%s",
        int((time.monotonic() - start) * 1000),
        len(markdown),
        len(llm_input),
        len(summary),
        truncated,
    )
    return _SummaryResult(markdown=summary, cleaned=True, truncated=truncated)
