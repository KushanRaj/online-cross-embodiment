import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.tokenization_utils_base import SpecialTokensMixin
import tyro


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    repo_id: str = "allenai/MolmoAct2-LIBERO"
    task_suite_name: str = "libero_spatial"
    task_ids: str | None = "0"
    num_trials_per_task: int = 1
    num_steps_wait: int = 10
    resize_size: int = 224
    replan_steps: int = 5
    video_out_path: str = "/home/ubuntu/robotics/runs/molmoact2-libero/videos"
    log_path: str = "/home/ubuntu/robotics/runs/molmoact2-libero/eval.log"
    dtype: str = "bfloat16"
    num_action_steps: int = 10
    enable_cuda_graph: bool = False
    seed: int = 7


def _patch_transformers_extra_special_tokens() -> None:
    original = SpecialTokensMixin._set_model_specific_special_tokens
    if getattr(original, "_molmoact2_patched", False):
        return

    def patched(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {
                f"extra_token_{idx}": token for idx, token in enumerate(special_tokens)
            }
        return original(self, special_tokens)

    patched._molmoact2_patched = True
    SpecialTokensMixin._set_model_specific_special_tokens = patched


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _to_pil(image: np.ndarray, resize_size: int) -> Image.Image:
    # Match the standard LIBERO visual orientation used by public pi/openvla eval scripts.
    image = np.ascontiguousarray(image[::-1, ::-1])
    pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    return pil.resize((resize_size, resize_size), Image.Resampling.BILINEAR)


def _state_from_obs(obs: dict) -> np.ndarray:
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    gripper_scalar = np.array([float(gripper.mean())], dtype=np.float32)
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
            gripper_scalar,
        ]
    ).astype(np.float32)


def _max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def eval_libero(args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_path, mode="w")],
        force=True,
    )
    np.random.seed(args.seed)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    dtype = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }[args.dtype]

    _patch_transformers_extra_special_tokens()
    processor = AutoProcessor.from_pretrained(args.repo_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.repo_id,
        trust_remote_code=True,
        dtype=dtype,
    ).to("cuda").eval()

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task_ids = list(range(task_suite.n_tasks))
    if args.task_ids is not None:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",") if x.strip()]
    max_steps = _max_steps(args.task_suite_name)

    total_episodes = 0
    total_successes = 0
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        for episode_idx in range(args.num_trials_per_task):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            action_plan = collections.deque()
            replay_images = []
            done = False
            t = 0

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                agent_img = _to_pil(obs["agentview_image"], args.resize_size)
                wrist_img = _to_pil(obs["robot0_eye_in_hand_image"], args.resize_size)
                replay_images.append(np.asarray(agent_img))

                if not action_plan:
                    state = _state_from_obs(obs)
                    with torch.inference_mode():
                        ctx = (
                            torch.autocast("cuda", dtype=dtype)
                            if dtype in {torch.bfloat16, torch.float16}
                            else torch.inference_mode()
                        )
                        with ctx:
                            out = model.predict_action(
                                processor=processor,
                                images=[agent_img, wrist_img],
                                task=str(task_description),
                                state=state,
                                norm_tag="libero",
                                inference_action_mode="continuous",
                                enable_depth_reasoning=False,
                                num_steps=args.num_action_steps,
                                normalize_language=True,
                                enable_cuda_graph=args.enable_cuda_graph,
                            )
                    actions_obj = out.actions
                    if hasattr(actions_obj, "detach"):
                        actions_obj = actions_obj.detach().cpu().numpy()
                    actions = np.asarray(actions_obj, dtype=np.float32)
                    if actions.ndim == 3 and actions.shape[0] == 1:
                        actions = actions[0]
                    if actions.ndim == 1:
                        actions = actions[None, :]
                    if actions.shape[-1] != 7:
                        raise ValueError(f"Expected LIBERO 7D actions, got {actions.shape}")
                    action_plan.extend(actions[: args.replan_steps])

                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    total_successes += 1
                    break
                t += 1

            total_episodes += 1
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            out_path = pathlib.Path(args.video_out_path) / (
                f"molmoact2_libero_task{task_id}_episode{episode_idx}_{task_segment}_{suffix}.mp4"
            )
            imageio.mimwrite(out_path, replay_images, fps=10)
            logging.info("Saved rollout MP4 at %s", out_path)
            logging.info("Success: %s", done)
            logging.info("Total episodes: %s successes: %s", total_episodes, total_successes)


if __name__ == "__main__":
    tyro.cli(eval_libero)
