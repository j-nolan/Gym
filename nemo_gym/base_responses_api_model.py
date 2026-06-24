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
from abc import abstractmethod
from typing import Optional

from fastapi import Body, FastAPI
from pydantic import Field

from nemo_gym.observability import install_trajectory_capture
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import BaseRunServerInstanceConfig, BaseServer, SimpleServer


class BaseResponsesAPIModelConfig(BaseRunServerInstanceConfig):
    observability_enabled: bool = Field(
        default=True,
        description=(
            "Emit one CLU trajectory record per model call (token stats, tool calls, "
            "messages, reasoning). Default on; set false to opt out."
        ),
    )
    trajectory_capture_dir: Optional[str] = Field(
        default=None,
        description=(
            "Directory for per-rollout trajectory-capture JSONL. Defaults to $NEMO_GYM_TRAJECTORY_DIR, "
            "else a per-server dir under the system temp dir."
        ),
    )


class BaseResponsesAPIModel(BaseServer):
    config: BaseResponsesAPIModelConfig


class SimpleResponsesAPIModel(BaseResponsesAPIModel, SimpleServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)

        app.post("/v1/chat/completions")(self.chat_completions)

        app.post("/v1/responses")(self.responses)

        # Default-on per-rollout trajectory capture (opt out via observability_enabled=false).
        # Installed as an exchange-capturing middleware, so it is independent of each server's
        # handler signature and never alters the response.
        install_trajectory_capture(app, self.config)

        return app

    @abstractmethod
    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        pass

    @abstractmethod
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        pass
