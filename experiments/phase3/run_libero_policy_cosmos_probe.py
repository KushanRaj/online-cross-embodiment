from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Protocol

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _add_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def _jpeg_dataset(handle: h5py.File, name: str, value: np.ndarray, jpeg_encode_image, jpeg: bool) -> None:
    is_image = value.ndim == 4 and value.shape[-1] == 3 and value.dtype == np.uint8
    if is_image and jpeg:
        jpeg_list = [jpeg_encode_image(frame, quality=95) for frame in value]
        dt = h5py.vlen_dtype(np.dtype("uint8"))
        handle.create_dataset(name + "_jpeg", data=jpeg_list, dtype=dt)
    else:
        handle.create_dataset(name, data=value)


def _write_episode_h5(
    out_dir: Path,
    suite: str,
    task_id: int,
    episode_idx: int,
    success: bool,
    run_id_note: str,
    task_description: str,
    policy_name: str,
    collected: dict[str, Any],
    jpeg_encode_image,
    jpeg: bool,
) -> Path:
    data_dir = out_dir / "rollout_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    path = data_dir / (
        f"episode_data--suite={suite}--{timestamp}--task={task_id}--ep={episode_idx}"
        f"--success={success}--{run_id_note}.hdf5"
    )
    with h5py.File(path, "w") as h5:
        for key, value in collected.items():
            if isinstance(value, np.ndarray):
                _jpeg_dataset(h5, key, value, jpeg_encode_image, jpeg)
            else:
                h5.attrs[key] = value
        h5.attrs["task_description"] = task_description
        h5.attrs["policy_name"] = policy_name
        h5.attrs["policy_mode"] = "external_policy_with_cosmos_future_probe"
    return path


class PolicyAdapter(Protocol):
    name: str

    def predict_chunk(self, obs: dict[str, Any], task_description: str, horizon: int) -> np.ndarray:
        ...


class OpenPIWebsocketPolicy:
    name = "pi05_libero"

    def __init__(self, openpi_repo: str, host: str, port: int, resize_size: int) -> None:
        _add_path(str(Path(openpi_repo) / "packages/openpi-client/src"))
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy

        self._image_tools = image_tools
        self._client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self._resize_size = resize_size

    def _image(self, image: np.ndarray) -> np.ndarray:
        image = np.ascontiguousarray(image[::-1, ::-1])
        return self._image_tools.convert_to_uint8(
            self._image_tools.resize_with_pad(image, self._resize_size, self._resize_size)
        )

    def _state(self, obs: dict[str, Any]) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
            )
        ).astype(np.float32)

    def predict_chunk(self, obs: dict[str, Any], task_description: str, horizon: int) -> np.ndarray:
        element = {
            "observation/image": self._image(obs["agentview_image"]),
            "observation/wrist_image": self._image(obs["robot0_eye_in_hand_image"]),
            "observation/state": self._state(obs),
            "prompt": str(task_description),
        }
        actions = np.asarray(self._client.infer(element)["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape[-1] != 7:
            raise ValueError(f"Expected π0.5 LIBERO 7D actions, got {actions.shape}")
        if len(actions) < horizon:
            raise ValueError(f"π0.5 returned {len(actions)} actions, need horizon={horizon}")
        return actions[:horizon].astype(np.float32)


class MolmoAct2Policy:
    name = "molmoact2_libero"

    def __init__(self, repo_id: str, resize_size: int, dtype: str, enable_cuda_graph: bool) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from transformers.tokenization_utils_base import SpecialTokensMixin

        original = SpecialTokensMixin._set_model_specific_special_tokens
        if not getattr(original, "_molmoact2_patched", False):

            def patched(self, special_tokens):
                if isinstance(special_tokens, list):
                    special_tokens = {f"extra_token_{idx}": token for idx, token in enumerate(special_tokens)}
                return original(self, special_tokens)

            patched._molmoact2_patched = True
            SpecialTokensMixin._set_model_specific_special_tokens = patched

        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        self._dtype = dtype_map[dtype]
        self._processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            repo_id,
            trust_remote_code=True,
            dtype=self._dtype,
        ).to("cuda").eval()
        self._resize_size = resize_size
        self._enable_cuda_graph = enable_cuda_graph

    def _pil(self, image: np.ndarray) -> Image.Image:
        image = np.ascontiguousarray(image[::-1, ::-1])
        pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        return pil.resize((self._resize_size, self._resize_size), Image.Resampling.BILINEAR)

    def _state(self, obs: dict[str, Any]) -> np.ndarray:
        gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
        return np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
                np.array([float(gripper.mean())], dtype=np.float32),
            ]
        ).astype(np.float32)

    def predict_chunk(self, obs: dict[str, Any], task_description: str, horizon: int) -> np.ndarray:
        agent_img = self._pil(obs["agentview_image"])
        wrist_img = self._pil(obs["robot0_eye_in_hand_image"])
        ctx = (
            torch.autocast("cuda", dtype=self._dtype)
            if self._dtype in {torch.bfloat16, torch.float16}
            else torch.inference_mode()
        )
        with torch.inference_mode(), ctx:
            out = self._model.predict_action(
                processor=self._processor,
                images=[agent_img, wrist_img],
                task=str(task_description),
                state=self._state(obs),
                norm_tag="libero",
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=horizon,
                normalize_language=True,
                enable_cuda_graph=self._enable_cuda_graph,
            )
        actions = out.actions
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape[-1] != 7:
            raise ValueError(f"Expected MolmoAct2 LIBERO 7D actions, got {actions.shape}")
        if len(actions) < horizon:
            raise ValueError(f"MolmoAct2 returned {len(actions)} actions, need horizon={horizon}")
        return actions[:horizon].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosmos-policy-repo", default="/home/ubuntu/robotics/repos/cosmos-policy")
    parser.add_argument("--openpi-repo", default="/home/ubuntu/robotics/repos/PolaRiS/third_party/openpi")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--policy", choices=["pi05", "molmo"], required=True)
    parser.add_argument("--pi-host", default="127.0.0.1")
    parser.add_argument("--pi-port", type=int, default=8010)
    parser.add_argument("--molmo-repo-id", default="allenai/MolmoAct2-LIBERO")
    parser.add_argument("--molmo-dtype", default="bfloat16")
    parser.add_argument("--molmo-enable-cuda-graph", action="store_true")
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-ids", default="0")
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--run-id-note", default="policy-cosmos-probe")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--env-img-res", type=int, default=256)
    parser.add_argument("--num-denoising-steps-action", type=int, default=5)
    args = parser.parse_args()

    _add_path(args.cosmos_policy_repo)

    from libero.libero import benchmark

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )
    from cosmos_policy.experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        save_rollout_video,
        save_rollout_video_with_future_image_predictions,
    )
    from cosmos_policy.experiments.robot.libero.run_libero_eval import (
        TASK_MAX_STEPS,
        PolicyEvalConfig,
        prepare_observation,
        validate_config,
    )
    from cosmos_policy.utils.utils import jpeg_encode_image, set_seed_everywhere

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(out_dir)

    cfg = PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        config_file="cosmos_policy/config/config.py",
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        dataset_stats_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json",
        t5_text_embeddings_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
        trained_with_image_aug=True,
        chunk_size=args.horizon,
        num_open_loop_steps=args.horizon,
        task_suite_name=args.task_suite_name,
        task_ids=args.task_ids,
        num_trials_per_task=args.num_trials_per_task,
        local_log_dir=str(out_dir),
        randomize_seed=False,
        data_collection=True,
        available_gpus="0",
        seed=args.seed,
        use_variance_scale=False,
        deterministic=True,
        run_id_note=args.run_id_note,
        ar_future_prediction=False,
        ar_value_prediction=False,
        use_jpeg_compression=True,
        flip_images=True,
        num_denoising_steps_action=args.num_denoising_steps_action,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
    )
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    cosmos_model, cosmos_config = get_model(cfg)
    assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size

    if args.policy == "pi05":
        policy: PolicyAdapter = OpenPIWebsocketPolicy(args.openpi_repo, args.pi_host, args.pi_port, args.resize_size)
    else:
        policy = MolmoAct2Policy(
            args.molmo_repo_id,
            args.resize_size,
            args.molmo_dtype,
            args.molmo_enable_cuda_graph,
        )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task_ids = [int(x.strip()) for x in args.task_ids.split(",") if x.strip()]
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    rows: list[dict[str, Any]] = []
    total_episodes = 0
    total_successes = 0

    for task_id in tqdm(task_ids, desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=args.env_img_res)

        for trial_idx in tqdm(range(args.num_trials_per_task), desc=f"task {task_id}", leave=False):
            env.reset()
            obs = env.set_init_state(initial_states[trial_idx])
            action_queue: deque[np.ndarray] = deque(maxlen=args.horizon)
            cosmos_action_queue: deque[np.ndarray] = deque(maxlen=args.horizon)
            replay_images: list[np.ndarray] = []
            replay_wrist_images: list[np.ndarray] = []
            future_predictions: list[dict[str, np.ndarray]] = []
            primary_images: list[np.ndarray] = []
            wrist_images: list[np.ndarray] = []
            proprio_list: list[np.ndarray] = []
            actions_list: list[np.ndarray] = []
            cosmos_actions_list: list[np.ndarray] = []
            query_t: list[int] = []
            query_primary: list[np.ndarray] = []
            query_wrist: list[np.ndarray] = []
            query_proprio: list[np.ndarray] = []
            query_policy_chunks: list[np.ndarray] = []
            query_cosmos_chunks: list[np.ndarray] = []
            query_future_primary: list[np.ndarray] = []
            query_future_wrist: list[np.ndarray] = []
            query_values: list[float] = []
            success = False
            t = 0
            wait_steps = 10

            while t < max_steps + wait_steps:
                if t < wait_steps:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                observation = prepare_observation(obs, None, cfg.flip_images)
                replay_images.append(observation["primary_image"])
                replay_wrist_images.append(observation["wrist_image"])
                primary_images.append(observation["primary_image"])
                wrist_images.append(observation["wrist_image"])
                proprio_list.append(observation["proprio"])

                if not action_queue:
                    query_start = time.time()
                    cosmos_result = get_action(
                        cfg,
                        cosmos_model,
                        dataset_stats,
                        observation,
                        task_description,
                        seed=cfg.seed,
                        randomize_seed=False,
                        num_denoising_steps_action=cfg.num_denoising_steps_action,
                        generate_future_state_and_value_in_parallel=True,
                    )
                    future = cosmos_result["future_image_predictions"]
                    future_image = future.get("future_image")
                    if future_image is None:
                        raise RuntimeError("Cosmos did not return future_image")

                    policy_chunk = policy.predict_chunk(obs, task_description, args.horizon)
                    cosmos_chunk = np.asarray(cosmos_result["actions"], dtype=np.float32).reshape(args.horizon, 7)
                    action_queue.extend(policy_chunk)
                    cosmos_action_queue.extend(cosmos_chunk)
                    future_predictions.append(future)
                    query_t.append(t - wait_steps)
                    query_primary.append(observation["primary_image"])
                    query_wrist.append(observation["wrist_image"])
                    query_proprio.append(observation["proprio"])
                    query_policy_chunks.append(policy_chunk.astype(np.float32))
                    query_cosmos_chunks.append(cosmos_chunk.astype(np.float32))
                    query_values.append(float(cosmos_result.get("value_prediction", 0.0)))
                    query_future_primary.append(future_image)
                    if future.get("future_wrist_image") is not None:
                        query_future_wrist.append(future["future_wrist_image"])
                    print(
                        {
                            "policy": policy.name,
                            "task": task_id,
                            "trial": trial_idx,
                            "t": t,
                            "query_sec": round(time.time() - query_start, 3),
                            "policy_l2": float(np.linalg.norm(policy_chunk.reshape(-1))),
                            "cosmos_l2": float(np.linalg.norm(cosmos_chunk.reshape(-1))),
                            "policy_cosmos_l2": float(
                                np.linalg.norm(policy_chunk.reshape(-1) - cosmos_chunk.reshape(-1))
                            ),
                        },
                        flush=True,
                    )

                action = action_queue.popleft()
                cosmos_action = cosmos_action_queue.popleft()
                actions_list.append(action.astype(np.float32))
                cosmos_actions_list.append(cosmos_action.astype(np.float32))
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    success = True
                    break
                t += 1

            total_episodes += 1
            total_successes += int(success)
            save_rollout_video(replay_images, total_episodes, success, task_description)
            save_rollout_video_with_future_image_predictions(
                replay_images,
                total_episodes,
                success=success,
                task_description=task_description,
                chunk_size=cfg.chunk_size,
                num_open_loop_steps=cfg.num_open_loop_steps,
                rollout_wrist_images=replay_wrist_images,
                future_primary_image_predictions=[x["future_image"] for x in future_predictions],
                future_wrist_image_predictions=[
                    x["future_wrist_image"] for x in future_predictions if x.get("future_wrist_image") is not None
                ],
                show_diff=False,
            )
            collected = {
                "primary_images": np.stack(primary_images, axis=0),
                "wrist_images": np.stack(wrist_images, axis=0),
                "proprio": np.stack(proprio_list, axis=0),
                "actions": np.stack(actions_list, axis=0),
                "cosmos_policy_actions": np.stack(cosmos_actions_list, axis=0),
                "query_t": np.asarray(query_t, dtype=np.int32),
                "query_primary_images": np.stack(query_primary, axis=0),
                "query_wrist_images": np.stack(query_wrist, axis=0),
                "query_proprio": np.stack(query_proprio, axis=0),
                "query_action_chunks": np.stack(query_policy_chunks, axis=0),
                "query_policy_action_chunks": np.stack(query_policy_chunks, axis=0),
                "query_cosmos_action_chunks": np.stack(query_cosmos_chunks, axis=0),
                "query_values": np.asarray(query_values, dtype=np.float32),
                "query_future_primary_images": np.stack(query_future_primary, axis=0),
                "success": success,
            }
            if query_future_wrist:
                collected["query_future_wrist_images"] = np.stack(query_future_wrist, axis=0)
            h5_path = _write_episode_h5(
                out_dir,
                cfg.task_suite_name,
                task_id,
                total_episodes,
                success,
                f"{policy.name}-{args.run_id_note}",
                task_description,
                policy.name,
                collected,
                jpeg_encode_image,
                cfg.jpeg_compress,
            )
            row = {
                "policy": policy.name,
                "task_id": task_id,
                "trial_idx": trial_idx,
                "episode": total_episodes,
                "success": success,
                "steps": len(actions_list),
                "queries": len(query_t),
                "task_description": task_description,
                "h5": str(h5_path),
            }
            rows.append(row)
            print(row, flush=True)

    summary = {
        "mode": "external_policy_with_cosmos_future_probe",
        "policy": policy.name,
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": total_successes / max(1, total_episodes),
        "task_ids": task_ids,
        "num_trials_per_task": args.num_trials_per_task,
        "horizon": args.horizon,
        "cosmos_ckpt": "nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        "note": "query_action_chunks is the executed policy chunk; query_future_primary_images is Cosmos P.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out_dir / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["policy"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
