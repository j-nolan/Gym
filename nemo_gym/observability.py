# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-rollout trajectory capture for model servers.

Default-on (opt out with ``observability_enabled: false``). A FastAPI middleware on
the model server records every ``/v1/responses``, ``/v1/chat/completions`` and
``/v1/messages`` exchange — full request + response, with token-ids when
``return_token_id_information`` is on — into a per-rollout
:class:`~nemo_gym.trajectory_capture.CaptureStore`, correlated via a request header.
The captured exchanges assemble into a full ordered trajectory (content, tool calls,
token-ids) via :func:`nemo_gym.trajectory_capture.assemble_rollout`.

:func:`summarize_response` additionally offers a compact per-call telemetry view.
Capture is best-effort and never affects the response returned to the caller.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi.responses import Response

from nemo_gym.trajectory_capture import CaptureStore

logger = logging.getLogger(__name__)

# A launcher or agent sets this header so exchanges can be grouped per task / rollout.
ROLLOUT_HEADER = "x-nemo-gym-rollout-id"

_OBSERVED_PATHS = {
    "/v1/responses": "responses",
    "/v1/chat/completions": "chat",
    "/v1/messages": "messages",
}


# ----------------------------------------------------------------------------
# Compact per-call telemetry (utility; the canonical record is the full exchange)
# ----------------------------------------------------------------------------
def _usage(usage: Any) -> Optional[dict[str, Any]]:
    """Normalize token usage across Responses, Chat Completions, and Anthropic Messages."""
    if not usage:
        return None
    tokens_in = usage.get("input_tokens")
    if tokens_in is None:
        tokens_in = usage.get("prompt_tokens")
    tokens_out = usage.get("output_tokens")
    if tokens_out is None:
        tokens_out = usage.get("completion_tokens")
    tokens_total = usage.get("total_tokens")
    if tokens_total is None and tokens_in is not None and tokens_out is not None:
        tokens_total = tokens_in + tokens_out
    details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    return {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_total": tokens_total,
        "tokens_reasoning": details.get("reasoning_tokens"),
    }


def summarize_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Compact CLU telemetry for a response payload (token stats + tool/message/reasoning signals).

    Handles all three response shapes: Responses (``output``), Chat Completions
    (``choices``), and Anthropic Messages (``content``).
    """
    summary: dict[str, Any] = {
        "model": payload.get("model"),
        "usage": _usage(payload.get("usage")),
        "num_tool_calls": 0,
        "tool_names": [],
        "num_messages": 0,
        "has_reasoning": False,
    }

    output = payload.get("output")
    if output is not None:  # Responses
        tool_calls = [item for item in output if item.get("type") == "function_call"]
        summary.update(
            num_tool_calls=len(tool_calls),
            tool_names=[call.get("name") for call in tool_calls],
            num_messages=sum(item.get("type") == "message" for item in output),
            has_reasoning=any(item.get("type") == "reasoning" for item in output),
        )
        return summary

    choices = payload.get("choices")
    if choices is not None:  # Chat Completions
        messages = [c.get("message") for c in choices if isinstance(c, dict) and c.get("message")]
        tool_calls = [tc for message in messages for tc in (message.get("tool_calls") or [])]
        summary.update(
            num_tool_calls=len(tool_calls),
            tool_names=[(tc.get("function") or {}).get("name") for tc in tool_calls],
            num_messages=len(messages),
            has_reasoning=any(message.get("reasoning_content") for message in messages),
        )
        return summary

    content = payload.get("content")
    if isinstance(content, list):  # Anthropic Messages
        tool_calls = [block for block in content if block.get("type") == "tool_use"]
        text_blocks = [block for block in content if block.get("type") == "text"]
        thinking = [block for block in content if block.get("type") in ("thinking", "redacted_thinking")]
        summary.update(
            num_tool_calls=len(tool_calls),
            tool_names=[block.get("name") for block in tool_calls],
            num_messages=1 if text_blocks else 0,
            has_reasoning=bool(thinking),
        )
        return summary

    return summary


# ----------------------------------------------------------------------------
# Full per-rollout exchange capture (the canonical trajectory source)
# ----------------------------------------------------------------------------
def _default_capture_dir(server_name: str) -> str:
    env_dir = os.environ.get("NEMO_GYM_TRAJECTORY_DIR")
    if env_dir:
        return env_dir
    return str(Path(tempfile.gettempdir()) / "nemo_gym_trajectories" / server_name)


def make_capture_store(config: Any) -> Optional[CaptureStore]:
    """Build a CaptureStore when observability is enabled (default on); otherwise None."""
    if not getattr(config, "observability_enabled", True):
        return None
    root = getattr(config, "trajectory_capture_dir", None) or _default_capture_dir(
        getattr(config, "name", None) or "model_server"
    )
    try:
        return CaptureStore(root)
    except Exception:
        logger.warning("Could not initialize trajectory capture at %s; disabling it.", root, exc_info=True)
        return None


def install_trajectory_capture(app: Any, config: Any) -> None:
    """Add the per-rollout exchange-capture middleware to a model-server app (no-op when disabled).

    Records the full request + response for each observed call into a rollout-keyed
    CaptureStore. Independent of each server's handler signature; never alters the
    response. Buffers the (non-streaming) JSON bodies and replays the request body so
    the downstream route still reads it.
    """
    store = make_capture_store(config)
    if store is None:
        return

    @app.middleware("http")
    async def _capture(request: Any, call_next: Any) -> Response:
        dialect = _OBSERVED_PATHS.get(request.url.path)
        if dialect is None:
            return await call_next(request)

        request_bytes = await request.body()

        # Replay the buffered body so the route handler can still read it.
        async def _receive() -> dict[str, Any]:
            return {"type": "http.request", "body": request_bytes, "more_body": False}

        request._receive = _receive

        response = await call_next(request)
        response_bytes = b"".join([chunk async for chunk in response.body_iterator])

        try:
            if response.status_code < 400 and response_bytes:
                rollout_id = request.headers.get(ROLLOUT_HEADER) or "rollout"
                store.record(
                    rollout_id,
                    {
                        "dialect": dialect,
                        "request": json.loads(request_bytes) if request_bytes else None,
                        "response": json.loads(response_bytes),
                    },
                )
        except Exception:
            logger.warning("Trajectory capture failed for one %s call.", dialect, exc_info=True)

        headers = dict(response.headers)
        headers.pop("content-length", None)  # Response() recomputes it from the buffered body
        return Response(
            content=response_bytes,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )
