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
from types import SimpleNamespace

from fastapi import Body, FastAPI
from fastapi.testclient import TestClient

from nemo_gym.observability import install_trajectory_capture, make_capture_store, summarize_response
from nemo_gym.trajectory_capture import (
    CaptureStore,
    StepRecord,
    aggregate_rollout_metrics,
    assemble_rollout,
    assemble_step_records,
    build_step_record,
    has_token_ids,
)

_RESPONSES_PAYLOAD = {
    "model": "m",
    "usage": {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "output_tokens_details": {"reasoning_tokens": 3},
    },
    "output": [
        {"type": "reasoning"},
        {"type": "message", "content": [{"type": "output_text", "text": "hi there"}]},
        {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": '{"city": "SF"}'},
    ],
}


# --- summarize_response telemetry utility (all three response shapes) ---
def test_summarize_responses_shape():
    summary = summarize_response(_RESPONSES_PAYLOAD)
    assert summary["usage"] == {"tokens_in": 10, "tokens_out": 5, "tokens_total": 15, "tokens_reasoning": 3}
    assert summary["num_tool_calls"] == 1 and summary["tool_names"] == ["get_weather"]
    assert summary["num_messages"] == 1 and summary["has_reasoning"] is True


def test_summarize_chat_completions_shape():
    payload = {
        "model": "m",
        "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
        "choices": [{"message": {"content": "hi", "tool_calls": [{"function": {"name": "f"}}], "reasoning_content": "r"}}],
    }
    summary = summarize_response(payload)
    assert summary["usage"]["tokens_in"] == 7
    assert summary["num_tool_calls"] == 1 and summary["tool_names"] == ["f"] and summary["has_reasoning"] is True


def test_summarize_anthropic_messages_shape():
    payload = {
        "model": "m",
        "usage": {"input_tokens": 8, "output_tokens": 6},
        "content": [
            {"type": "thinking", "thinking": "let me think"},
            {"type": "text", "text": "hi there"},
            {"type": "tool_use", "name": "get_weather", "input": {"city": "SF"}},
        ],
    }
    summary = summarize_response(payload)
    assert summary["usage"] == {"tokens_in": 8, "tokens_out": 6, "tokens_total": 14, "tokens_reasoning": None}
    assert summary["num_tool_calls"] == 1 and summary["num_messages"] == 1 and summary["has_reasoning"] is True


def test_make_capture_store_disabled_returns_none():
    assert make_capture_store(SimpleNamespace(observability_enabled=False)) is None


# --- #1483 StepRecord contract ---
def test_build_step_record_from_exchange():
    exchange = {
        "dialect": "responses",
        "model_server": "srv",
        "trial_index": 2,
        "turn_index": 1,
        "latency_ms": 18.4,
        "request": {"input": "hi"},
        "response": {
            "model": "m",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "output_tokens_details": {"reasoning_tokens": 3},
                "prompt_tokens_details": {"cached_tokens": 4},
            },
            "output": [
                {"type": "reasoning", "summary": [{"text": "thinking..."}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
                {"type": "function_call", "call_id": "c1", "name": "calc", "arguments": '{"x": 1}'},
            ],
        },
    }
    rec = build_step_record(exchange, step_index=3, run_id="run-1")
    assert (rec.step_index, rec.trial_index, rec.turn_index) == (3, 2, 1)
    assert rec.run_id == "run-1" and rec.model_server == "srv" and rec.dialect == "responses"
    assert (rec.tokens_in, rec.tokens_out, rec.tokens_total, rec.tokens_reasoning) == (10, 5, 15, 3)
    assert rec.cache_hit is True and rec.cached_tokens == 4
    assert rec.reasoning_content == "thinking..."
    assert rec.tool_calls == [{"call_id": "c1", "name": "calc", "arguments": {"x": 1}}]
    assert rec.latency_total_ms == 18.4


def test_step_record_json_schema_has_contract_fields():
    props = StepRecord.model_json_schema()["properties"]
    for field in (
        "step_index",
        "trial_index",
        "turn_index",
        "tokens_in",
        "tokens_out",
        "tokens_reasoning",
        "tokens_total",
        "request",
        "response",
        "tool_calls",
        "reasoning_content",
        "cache_hit",
        "error_category",
        "latency_total_ms",
        "latency_ttft_ms",
    ):
        assert field in props


# --- full per-rollout capture + assembly (trajectory + StepRecords) ---
def test_capture_assembles_trajectory_and_step_records(tmp_path):
    """End-to-end: capture two model calls (tool call + tool result + token-ids + usage) and
    assemble both the NeMoGym trajectory and the typed #1483 StepRecords. Also proves the
    request-body replay works (the route still receives its body through the middleware)."""
    turns = [
        {
            "model": "m",
            "usage": {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19, "prompt_tokens_details": {"cached_tokens": 4}},
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "let me check"}],
                    "prompt_token_ids": [9],
                    "generation_token_ids": [1, 2, 3],
                    "generation_log_probs": [-0.1, -0.2, -0.3],
                },
                {"type": "function_call", "name": "calc", "call_id": "c1", "arguments": '{"x": 1}'},
            ],
        },
        {
            "model": "m",
            "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer is 42"}],
                    "prompt_token_ids": [9, 1, 2, 3],
                    "generation_token_ids": [4, 5],
                    "generation_log_probs": [-0.1, -0.2],
                }
            ],
        },
    ]
    seen_requests: list[dict] = []

    app = FastAPI()

    @app.post("/v1/responses")
    async def _responses(body: dict = Body()) -> dict:
        seen_requests.append(body)
        return turns[len(seen_requests) - 1]

    config = SimpleNamespace(observability_enabled=True, trajectory_capture_dir=str(tmp_path), name="srv")
    install_trajectory_capture(app, config)
    client = TestClient(app)
    base_headers = {"x-nemo-gym-rollout-id": "rollout-x", "x-nemo-gym-trial-index": "2"}

    r1 = client.post("/v1/responses", json={"input": "solve it"}, headers={**base_headers, "x-nemo-gym-turn-index": "0"})
    r2 = client.post(
        "/v1/responses",
        json={"input": [{"type": "function_call_output", "call_id": "c1", "output": "42"}]},
        headers={**base_headers, "x-nemo-gym-turn-index": "1"},
    )

    assert r1.status_code == 200 and r2.status_code == 200
    assert seen_requests[0] == {"input": "solve it"}  # request-body replay worked

    # NeMoGym trajectory (ordered output items).
    items = assemble_rollout(CaptureStore(tmp_path), "rollout-x")
    assert [type(i).__name__ for i in items] == [
        "NeMoGymResponseOutputMessageForTraining",
        "NeMoGymResponseFunctionToolCall",
        "NeMoGymFunctionCallOutput",
        "NeMoGymResponseOutputMessageForTraining",
    ]
    assert has_token_ids(items) is True

    # Typed #1483 StepRecords.
    steps = assemble_step_records(CaptureStore(tmp_path), "rollout-x", run_id="run-1")
    assert [s.step_index for s in steps] == [0, 1]
    assert [s.turn_index for s in steps] == [0, 1]
    assert all(s.trial_index == 2 and s.run_id == "run-1" and s.model_server == "srv" for s in steps)
    assert (steps[0].tokens_in, steps[0].tokens_out, steps[0].tokens_total) == (12, 7, 19)
    assert steps[0].cache_hit is True and steps[0].cached_tokens == 4
    assert steps[0].tool_calls == [{"call_id": "c1", "name": "calc", "arguments": {"x": 1}}]
    assert steps[0].latency_total_ms is not None
    assert steps[1].tokens_in == 20

    # Per-rollout aggregates for the rollout record.
    agg = aggregate_rollout_metrics(CaptureStore(tmp_path), "rollout-x")
    assert agg["tokens_in"] == 32 and agg["tokens_out"] == 12
    assert agg["num_turns"] == 2 and agg["num_steps"] == 2


def test_failed_call_is_captured_with_error_category(tmp_path):
    """A non-2xx model call is captured (replacing generic exception catching) with a
    normalized error_category + status_code on the StepRecord."""
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.post("/v1/responses")
    async def _boom(body: dict = Body()) -> JSONResponse:
        return JSONResponse(content={"error": "boom"}, status_code=500)

    config = SimpleNamespace(observability_enabled=True, trajectory_capture_dir=str(tmp_path), name="srv")
    install_trajectory_capture(app, config)
    client = TestClient(app)

    r = client.post("/v1/responses", json={"input": "x"}, headers={"x-nemo-gym-rollout-id": "r-err"})
    assert r.status_code == 500  # response unchanged

    steps = assemble_step_records(CaptureStore(tmp_path), "r-err")
    assert len(steps) == 1
    assert steps[0].error_category == "upstream_error"
    assert steps[0].status_code == 500
