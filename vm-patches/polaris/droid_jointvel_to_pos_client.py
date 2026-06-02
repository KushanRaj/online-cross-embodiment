import os

import numpy as np
from openpi_client import image_tools, websocket_client_policy
from polaris.config import PolicyArgs
from polaris.policy.abstract_client import InferenceClient


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@InferenceClient.register(client_name="DroidJointVelToPos")
class DroidJointVelToPosClient(InferenceClient):
    """Adapter for public pi05_droid joint-velocity outputs in joint-position envs."""

    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        if args.open_loop_horizon is None:
            raise ValueError("open_loop_horizon must be set for DroidJointVelToPosClient")

        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )
        self.dt = float(os.environ.get("DROID_JOINT_VEL_TO_POS_DT", "0.0666666667"))
        self.debug_chunks = int(
            os.environ.get("DROID_JOINT_VEL_TO_POS_DEBUG_CHUNKS", "3")
        )
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.open_loop_horizon = args.open_loop_horizon

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
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        curr_obs = self._extract_observation(obs)
        both = None

        if self.rerender:
            exterior_image = image_tools.resize_with_pad(
                curr_obs["right_image"], 224, 224
            )
            wrist_image = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
            request_data = {
                "observation/exterior_image_1_left": exterior_image,
                "observation/wrist_image_left": wrist_image,
                "observation/joint_position": curr_obs["joint_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": instruction,
            }
            server_response = self.client.infer(request_data)
            actions = np.asarray(server_response["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None, :]
            if actions.shape[-1] != 8:
                raise ValueError(f"Expected 8D DROID action chunk, got {actions.shape}")

            self.pred_action_chunk = actions
            self.actions_from_chunk_completed = 0
            both = np.concatenate([exterior_image, wrist_image], axis=1)

        if return_viz and both is None:
            both = np.concatenate(
                [
                    image_tools.resize_with_pad(curr_obs["right_image"], 224, 224),
                    image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                ],
                axis=1,
            )

        if self.pred_action_chunk is None:
            raise ValueError("No action chunk predicted")

        idx = min(self.actions_from_chunk_completed, len(self.pred_action_chunk) - 1)
        raw_action = self.pred_action_chunk[idx]
        action = self._velocity_to_joint_position(raw_action, curr_obs["joint_position"])

        if self.debug_chunks > 0 and idx == 0:
            print(
                "DroidJointVelToPos raw first action:",
                np.array2string(raw_action, precision=5, suppress_small=False),
                flush=True,
            )
            print(
                "DroidJointVelToPos adapted first action:",
                np.array2string(action, precision=5, suppress_small=False),
                flush=True,
            )
            self.debug_chunks -= 1

        self.actions_from_chunk_completed += 1
        return action, both

    def _velocity_to_joint_position(
        self, raw_action: np.ndarray, joint_position: np.ndarray
    ) -> np.ndarray:
        raw_action = np.asarray(raw_action, dtype=np.float32)
        joint_position = np.asarray(joint_position, dtype=np.float32)
        target_joints = joint_position[:7] + raw_action[:7] * self.dt
        gripper = np.array([1.0 if raw_action[-1] > 0.5 else 0.0], dtype=np.float32)
        return np.concatenate([target_joints.astype(np.float32), gripper])

    def _extract_observation(self, obs_dict):
        right_image = obs_dict["splat"]["external_cam"]
        wrist_image = obs_dict["splat"]["wrist_cam"]

        robot_state = obs_dict["policy"]
        joint_position = np.squeeze(_as_numpy(robot_state["arm_joint_pos"]))
        gripper_position = np.squeeze(_as_numpy(robot_state["gripper_pos"]))
        joint_position = np.atleast_1d(joint_position).astype(np.float32)
        gripper_position = np.atleast_1d(gripper_position).astype(np.float32)

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }
