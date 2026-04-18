# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    def _build_reward_inputs(self, data: DataProto) -> Tuple[list[dict], torch.Tensor]:
        """Decode responses and assemble reward input dicts for a full batch.

        Returns the list of reward inputs and the per-sample response lengths (needed
        by compute_reward to place the scalar reward at the last valid token position).
        """
        response_ids = data.batch["responses"]
        response_lengths = torch.sum(data.batch["response_mask"], dim=-1)
        has_anchors = "anchor_correct" in data.non_tensor_batch

        reward_inputs = []
        for i in range(len(data)):
            length = int(response_lengths[i].item())
            reward_input = {
                "response": self.tokenizer.decode(
                    response_ids[i, :length], skip_special_tokens=self.config.skip_special_tokens
                ),
                "response_length": length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            if has_anchors:
                reward_input["anchor_correct"] = data.non_tensor_batch["anchor_correct"][i]
                reward_input["anchor_position"] = data.non_tensor_batch["anchor_position"][i]
            reward_inputs.append(reward_input)

        return reward_inputs, response_lengths

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs, response_lengths = self._build_reward_inputs(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, reward_input in enumerate(reward_inputs):
            score = self.reward_fn(reward_input)
            reward_tensor[i, reward_input["response_length"] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)
        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs, response_lengths = self._build_reward_inputs(data)
        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            reward_tensor[i, reward_inputs[i]["response_length"] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)
        return reward_tensor, reward_metrics
