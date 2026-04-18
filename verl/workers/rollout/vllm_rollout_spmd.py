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

import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


# Number of visual claims pre-generated per training example (matches dataset field count claim_1..claim_N).
_NUM_CLAIMS = 20

# Responses approaching this token count have their anchor search window shortened to avoid overflow.
_ANCHOR_TOKEN_LIMIT = 6000
# Characters to leave free at the tail of a long response when computing the search window.
_ANCHOR_TAIL_BUFFER = 128


def _extract_think_part(response: str) -> str:
    """Return the text before </think>; fall back to text before \\boxed{}, or the full response."""
    think_end = response.find("</think>")
    if think_end != -1:
        return response[:think_end]
    boxed_start = response.find("\\boxed")
    if boxed_start != -1:
        return response[:boxed_start]
    return response


def _find_cutoff_positions(think_part: str, max_pos: int, separators: set[str], min_sep_dist: int) -> np.ndarray:
    """Return character positions of sentence boundaries within think_part.

    Only positions before max_pos are considered, and consecutive positions must be
    at least min_sep_dist apart to avoid over-dense anchoring.
    """
    filtered, last_pos = [], -min_sep_dist - 1
    for pos, char in enumerate(think_part):
        if pos >= max_pos:
            break
        if char in separators and pos - last_pos >= min_sep_dist:
            filtered.append(pos)
            last_pos = pos
    return np.array(filtered, dtype=np.int32)


def _sample_k_positions(
    positions: np.ndarray, k: int, rng: np.random.Generator, fallback_pos: int
) -> np.ndarray:
    """Sample k sorted anchor positions from available cutoffs.

    If there are no valid positions, all k anchors are placed at fallback_pos.
    If there are fewer than k positions, the last position is repeated to fill the remainder.
    """
    n_pos = len(positions)
    if n_pos == 0:
        return np.full(k, fallback_pos, dtype=np.int32)
    if k <= n_pos:
        return np.sort(rng.choice(positions, size=k, replace=False))
    # Pad with the last available position when cutoffs are scarce.
    return np.sort(np.concatenate([positions, np.full(k - n_pos, positions[-1], dtype=np.int32)]))


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        self.tokenizer = tokenizer
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )
            raw_response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                raw_response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_raw_prompt_ids = _repeat_interleave(batch_raw_prompt_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {
                "multi_modal_data": batch_multi_modal_data,
                "raw_prompt_ids": batch_raw_prompt_ids,
                "raw_response_ids": np.array(raw_response_ids, dtype=object),
                }
        else:
            non_tensor_batch = {
                "raw_prompt_ids": batch_raw_prompt_ids,
                "raw_response_ids": np.array(raw_response_ids, dtype=object),
            }

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    @torch.no_grad()
    def generate_anchors(self, prompts: DataProto) -> DataProto:
        """Insert visual claim anchors into rollout responses and score model perception.

        For each response, k positions are sampled from sentence boundaries within the
        <think>...</think> section. At each position, a visual claim is appended and the
        model is prompted for a binary Yes/No judgment (single token, greedy). Predictions
        are compared against ground-truth labels to produce per-example anchor_correct
        (bool array of shape (n, k)) and anchor_position (int array of shape (n, k)).
        """
        tensor_batch = prompts.batch
        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_raw_response_ids = non_tensor_batch.pop("raw_response_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)

        decoded_responses = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch_raw_response_ids
        ]

        separators = {",", ".", "\n", "!", "?", ";", ":"}
        min_sep_dist = 2
        k = self.config.anchor_k
        rng = np.random.default_rng()

        # --- Step 1: Find valid anchor cutoff positions within <think>...</think> ---
        # Responses near the token limit have a shortened search window to avoid overflow.
        token_counts = tensor_batch["attention_mask"].eq(1).sum(dim=-1)  # (batch_size,)
        cutoff_positions = []
        for i, response in enumerate(decoded_responses):
            think_part = _extract_think_part(response)
            max_pos = (
                len(think_part) - _ANCHOR_TAIL_BUFFER
                if token_counts[i] > _ANCHOR_TOKEN_LIMIT
                else len(think_part)
            )
            cutoff_positions.append(_find_cutoff_positions(think_part, max_pos, separators, min_sep_dist))

        # --- Step 2: Sample k sorted anchor positions per response ---
        anchor_positions = [
            _sample_k_positions(positions, k, rng, fallback_pos=len(decoded_responses[i]) - 1)
            for i, positions in enumerate(cutoff_positions)
        ]

        # --- Step 3: Build anchored sequences (prompt + truncated response + claim question) ---
        # For each response, k new sequences are created, each truncated at a different anchor
        # position and appended with a randomly chosen visual claim as a Yes/No question.
        anchored_sequences = []
        for i, (response, anchors) in enumerate(zip(decoded_responses, anchor_positions)):
            claims = [non_tensor_batch[f"claim_{j + 1}"][i] for j in range(_NUM_CLAIMS)]
            chosen_indices = rng.choice(_NUM_CLAIMS, size=len(anchors), replace=False)
            decoded_prompt = self.tokenizer.decode(batch_raw_prompt_ids[i], skip_special_tokens=False)
            for anchor_idx, anchor_pos in enumerate(anchors):
                claim_text = claims[chosen_indices[anchor_idx]]
                anchor_question = f"\n<anchor>{claim_text} Is this claim correct? Answer (Yes/No): "
                anchored_sequences.append(decoded_prompt + response[:anchor_pos + 1] + anchor_question)

        # --- Step 4: Generate single-token Yes/No predictions via vLLM ---
        # Each example contributes k sequences; replicate its multi-modal data to match.
        repeated_multi_modal_data = [data for data in batch_multi_modal_data for _ in range(k)]
        vllm_inputs = [
            {
                "prompt_token_ids": list(self.tokenizer.encode(prompt, add_special_tokens=False)),
                "multi_modal_data": _process_multi_modal_data(
                    mm_data,
                    prompts.meta_info["min_pixels"],
                    prompts.meta_info["max_pixels"],
                    prompts.meta_info["video_fps"],
                ),
            }
            for prompt, mm_data in zip(anchored_sequences, repeated_multi_modal_data)
        ]

        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )

        # --- Step 5: Decode predictions into binary labels ---
        # allowed_token_ids is split evenly: first half = Yes tokens, second half = No tokens.
        allowed_token_ids = prompts.meta_info["allowed_token_ids"]
        half = len(allowed_token_ids) // 2
        yes_token_ids = set(allowed_token_ids[:half])
        no_token_ids = set(allowed_token_ids[half:])

        anchor_pred_labels = []
        for completion in completions:
            for output in completion.outputs:
                token_id = output.token_ids[0]
                if token_id in yes_token_ids:
                    anchor_pred_labels.append(1)
                elif token_id in no_token_ids:
                    anchor_pred_labels.append(0)
                else:
                    raise ValueError(f"Token id {token_id} is not in allowed_token_ids.")

        # --- Step 6: Compare predictions against ground-truth claim labels ---
        # label_{j+1}[i] is "CORRECT" (→1) or "WRONG" (→0) for the j-th anchor of example i.
        n = len(prompts)
        gt_labels = np.array(
            [
                [1 if non_tensor_batch[f"label_{j + 1}"][i] == "CORRECT" else 0 for j in range(k)]
                for i in range(n)
            ],
            dtype=int,
        )  # shape (n, k)
        anchor_pred_labels = np.array(anchor_pred_labels, dtype=int).reshape(n, k)
        anchor_correct = anchor_pred_labels == gt_labels  # shape (n, k)

        return DataProto(
            batch=prompts.batch,
            non_tensor_batch={"anchor_correct": anchor_correct, "anchor_position": np.array(anchor_positions)},
            meta_info=None,
        )