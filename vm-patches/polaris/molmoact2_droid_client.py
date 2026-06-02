import os
import re

import numpy as np
from PIL import Image
from polaris.config import PolicyArgs
from polaris.policy.abstract_client import InferenceClient
import torch
from transformers.tokenization_utils_base import SpecialTokensMixin
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor


def _as_pil_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _patch_transformers_extra_special_tokens() -> None:
    original = SpecialTokensMixin._set_model_specific_special_tokens
    if getattr(original, "_molmoact2_patched", False):
        return

    def patched(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {
                re.sub(r"[^0-9a-zA-Z]+", "_", token).strip("_").lower()
                or f"extra_token_{idx}": token
                for idx, token in enumerate(special_tokens)
            }
        return original(self, special_tokens)

    patched._molmoact2_patched = True
    SpecialTokensMixin._set_model_specific_special_tokens = patched


@InferenceClient.register(client_name="MolmoAct2Droid")
class MolmoAct2DroidClient(InferenceClient):
    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        self.repo_id = os.environ.get("MOLMOACT2_REPO_ID", "allenai/MolmoAct2-DROID")
        self.norm_tag = os.environ.get("MOLMOACT2_NORM_TAG", "franka_droid")
        self.num_steps = int(os.environ.get("MOLMOACT2_NUM_STEPS", "10"))
        self.enable_cuda_graph = _env_bool("MOLMOACT2_ENABLE_CUDA_GRAPH", True)
        self.open_loop_horizon = args.open_loop_horizon or int(
            os.environ.get("MOLMOACT2_OPEN_LOOP_HORIZON", "8")
        )

        dtype_name = os.environ.get("MOLMOACT2_DTYPE", "float32").lower()
        dtype = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }.get(dtype_name, torch.float32)

        _patch_transformers_extra_special_tokens()
        self.processor = AutoProcessor.from_pretrained(self.repo_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.repo_id,
            trust_remote_code=True,
            dtype=dtype,
        ).to("cuda").eval()

        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    @property
    def rerender(self) -> bool:
        return (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        )

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    def infer(
        self, obs, instruction, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        viz = None
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            curr_obs = self._extract_observation(obs)
            with torch.inference_mode():
                out = self.model.predict_action(
                    processor=self.processor,
                    images=[curr_obs["right_image"], curr_obs["wrist_image"]],
                    task=instruction,
                    state=curr_obs["robot_state"],
                    norm_tag=self.norm_tag,
                    inference_action_mode="continuous",
                    enable_depth_reasoning=False,
                    num_steps=self.num_steps,
                    normalize_language=True,
                    enable_cuda_graph=self.enable_cuda_graph,
                )
            actions = _as_numpy(out.actions).astype(np.float32)
            if actions.ndim == 3 and actions.shape[0] == 1:
                actions = actions[0]
            if actions.ndim == 1:
                actions = actions[None, :]
            if actions.ndim == 1:
                actions = actions[None, :]
            if actions.shape[-1] != 8:
                raise ValueError(f"Expected 8D DROID action, got {actions.shape}")

            self.pred_action_chunk = actions
            self.actions_from_chunk_completed = 0
            viz = np.concatenate(
                [
                    np.asarray(curr_obs["right_image"].resize((224, 224))),
                    np.asarray(curr_obs["wrist_image"].resize((224, 224))),
                ],
                axis=1,
            )

        if self.pred_action_chunk is None:
            raise ValueError("No MolmoAct2 action chunk predicted")

        idx = min(self.actions_from_chunk_completed, len(self.pred_action_chunk) - 1)
        action = self.pred_action_chunk[idx]
        self.actions_from_chunk_completed += 1

        action = np.asarray(action, dtype=np.float32)
        action[-1] = 1.0 if action[-1] > 0.5 else 0.0
        return action, viz

    def _extract_observation(self, obs_dict):
        right_image = _as_pil_rgb(obs_dict["splat"]["external_cam"])
        wrist_image = _as_pil_rgb(obs_dict["splat"]["wrist_cam"])

        robot_state = obs_dict["policy"]
        joint_position = np.squeeze(_as_numpy(robot_state["arm_joint_pos"]))
        gripper_position = np.squeeze(_as_numpy(robot_state["gripper_pos"]))
        joint_position = np.atleast_1d(joint_position)
        gripper_position = np.atleast_1d(gripper_position)
        state = np.concatenate([joint_position, gripper_position]).astype(np.float32)

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "robot_state": state,
        }
