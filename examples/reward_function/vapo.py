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

import re
from typing import Any


import numpy as np
from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def anchor_reward(response: str, anchor_correct: np.ndarray, anchor_position: np.ndarray, beta: float = 1.5) -> float:
    """Return a perception score in [0, 1] based on how well the model verifies visual claims.

    Implements Eq. (4) from the paper: a weighted average of per-anchor correctness where
    weights are exp(beta * position/length), giving greater emphasis to later anchors.
    A raw score of 0.5 (random chance) maps to 0; perfect accuracy maps to 1.
    """
    if anchor_correct.size == 0:
        return 0.0
    length_ratios = anchor_position / len(response)
    weights = np.exp(beta * length_ratios)
    weighted_score = np.sum(anchor_correct * weights) / np.sum(weights)
    return float(max((weighted_score - 0.5) / 0.5, 0.0))


def compute_score(reward_input: dict[str, Any], format_weight: float = 0.1, anchor_weight: float = 0.1) -> dict[str, float]:
    if not isinstance(reward_input, dict):
        raise ValueError("Please use `reward_type=sequential` for vapo reward function.")

    format_score = format_reward(reward_input["response"])
    accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
    if "anchor_correct" in reward_input:
        anchor_score = anchor_reward(reward_input["response"], reward_input["anchor_correct"], reward_input["anchor_position"])
        conditioned_anchor_score = anchor_score if accuracy_score == 1.0 else 0.0
        overall = (
            (1 - format_weight - anchor_weight) * accuracy_score
            + format_weight * format_score
            + anchor_weight * conditioned_anchor_score
        )
    else:
        anchor_score = 0.0
        overall = (
            (1 - format_weight) * accuracy_score
            + format_weight * format_score
        )
    return {
        "overall": overall,
        "format": format_score,
        "accuracy": accuracy_score,
        "anchor_score": anchor_score,
    }
