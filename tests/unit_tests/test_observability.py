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
from nemo_gym.trajectory_capture import CaptureStore, assemble_rollout, has_token_ids

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
    assert summary["num_tool_calls"] == 1
    assert summary["tool_names"] == ["get_weather"]
    assert summary["num_messages"] == 1
    assert summary["has_reasoning"] is True


def test_summarize_chat_completions_shape():
    payload = {
        "model": "m",
        "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
        "choices": [{"message": {"content": "hi", "tool_calls": [{"function": {"name": "f"}}], "reasoning_content": "r"}}],
    }
    summary = summarize_response(payload)
    assert summary["usage"]["tokens_in"] == 7
    assert summary["num_tool_calls"] == 1
    assert summary["tool_names"] == ["f"]
    assert summary["has_reasoning"] is True


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
    assert summary["num_tool_calls"] == 1
    assert summary["num_messages"] == 1
    assert summary["has_reasoning"] is True


def test_make_capture_store_disabled_returns_none():
    assert make_capture_store(SimpleNamespace(observability_enabled=False)) is None


# --- full per-rollout capture + assembly (the report) ---
def test_capture_assembles_full_two_turn_trajectory(tmp_path):
    """End-to-end: capture two model calls (with a tool call + tool result + token-ids)
    and assemble the full ordered trajectory. Also proves the request-body replay works
    (the route still receives its body through the capture middleware)."""
    turns = [
        {
            "model": "m",
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
    headers = {"x-nemo-gym-rollout-id": "rollout-x"}

    r1 = client.post("/v1/responses", json={"input": "solve it"}, headers=headers)
    r2 = client.post(
        "/v1/responses",
        json={"input": [{"type": "function_call_output", "call_id": "c1", "output": "42"}]},
        headers=headers,
    )

    # Responses preserved through the capture middleware.
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["output"][1]["name"] == "calc"
    # Request-body replay worked: the route received both bodies.
    assert seen_requests[0] == {"input": "solve it"}
    assert seen_requests[1]["input"][0]["call_id"] == "c1"

    # Full ordered trajectory assembled from the captured exchanges.
    items = assemble_rollout(CaptureStore(tmp_path), "rollout-x")
    assert [type(i).__name__ for i in items] == [
        "NeMoGymResponseOutputMessageForTraining",  # turn 1 assistant
        "NeMoGymResponseFunctionToolCall",  # turn 1 tool call
        "NeMoGymFunctionCallOutput",  # turn 2 tool result (from request input)
        "NeMoGymResponseOutputMessageForTraining",  # turn 2 assistant
    ]
    assert items[0].content[0].text == "let me check"
    assert items[0].generation_token_ids == [1, 2, 3]
    assert items[1].name == "calc"
    assert items[2].output == "42"
    assert items[3].content[0].text == "answer is 42"
    assert has_token_ids(items) is True
