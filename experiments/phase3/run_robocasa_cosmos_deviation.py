#!/usr/bin/env python3
"""Run Cosmos Policy on RoboCasa and save Phase 3 IDM-deviation HDF5s.

This mirrors the LIBERO rollout data contract used by
`evaluate_online_cosmos_deviation.py`:

  primary_images
  query_t
  query_primary_images
  query_future_primary_images
  query_proprio
  query_action_chunks

RoboCasa-specific action convention:
- Cosmos policy predicts 7D manipulation actions.
- RoboCasa env expects 12D, so rollout pads with [0, 0, 0, 0, -1].
- The saved IDM reference chunk remains 16 x 7.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm


DEFAULT_TASKS = [
    "CloseDrawer",
    "CoffeeServeMug",
    "CoffeeSetupMug",
    "OpenDrawer",
    "TurnOffMicrowave",
    "TurnOffSinkFaucet",
    "TurnOffStove",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
    "TurnOnStove",
    "TurnSinkSpout",
]
MOBILE_BASE_PAD = np.asarray([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def _add_repo_to_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _jpeg_dataset(handle: h5py.File, name: str, value: np.ndarray, jpeg_encode_image, jpeg: bool) -> None:
    is_image = value.ndim == 4 and value.shape[-1] == 3 and value.dtype == np.uint8
    if is_image and jpeg:
        jpeg_list = [np.frombuffer(jpeg_encode_image(frame, quality=95), dtype=np.uint8) for frame in value]
        dt = h5py.vlen_dtype(np.dtype("uint8"))
        ds = handle.create_dataset(name + "_jpeg", shape=(len(jpeg_list),), dtype=dt)
        for idx, encoded in enumerate(jpeg_list):
            ds[idx] = encoded
    else:
        handle.create_dataset(name, data=value)


def _write_episode_h5(
    out_dir: Path,
    task_name: str,
    episode_idx: int,
    success: bool,
    task_description: str,
    collected: dict[str, Any],
    jpeg_encode_image,
    jpeg: bool,
) -> Path:
    data_dir = out_dir / "rollout_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    safe_desc = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:35]
    path = data_dir / (
        f"{timestamp}--episode_data--task={task_name}--desc={safe_desc}"
        f"--ep={episode_idx}--success={success}.hdf5"
    )
    with h5py.File(path, "w") as h5:
        for key, value in collected.items():
            if isinstance(value, np.ndarray):
                _jpeg_dataset(h5, key, value, jpeg_encode_image, jpeg)
            else:
                h5.attrs[key] = value
        h5.attrs["task_name"] = task_name
        h5.attrs["task_description"] = task_description
        h5.attrs["success"] = success
        h5.attrs["policy_mode"] = "cosmos_policy_robocasa"
        h5.attrs["idm_reference_horizon"] = 16
        h5.attrs["policy_native_chunk_size"] = 32
        h5.attrs["env_action_pad"] = json.dumps(MOBILE_BASE_PAD.tolist())
    return path


def _stack_optional(values: list[np.ndarray]) -> np.ndarray | None:
    if not values:
        return None
    return np.stack(values, axis=0)


def _pad_env_action(action_7d: np.ndarray, env_action_dim: int) -> np.ndarray:
    action_7d = np.asarray(action_7d, dtype=np.float32)
    if action_7d.shape[-1] == 7 and env_action_dim == 12:
        return np.concatenate([action_7d, MOBILE_BASE_PAD], axis=0)
    return action_7d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosmos-policy-repo", default="/home/ubuntu/robotics/repos/cosmos-policy")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--env-img-res", type=int, default=224)
    parser.add_argument("--obj-instance-split", default="target")
    parser.add_argument("--num-denoising-steps-action", type=int, default=5)
    parser.add_argument("--num-denoising-steps-future-state", type=int, default=1)
    parser.add_argument("--max-steps-override", type=int, default=None)
    parser.add_argument("--debug-timing", action="store_true")
    args = parser.parse_args()

    _add_repo_to_path(args.cosmos_policy_repo)
    os.chdir(args.cosmos_policy_repo)

    # Cosmos Policy's RoboCasa eval helper currently imports the older registry
    # names. Newer RoboCasa exposes the same split as atomic/composite.
    import robocasa.utils.dataset_registry as robocasa_registry

    if not hasattr(robocasa_registry, "SINGLE_STAGE_TASK_DATASETS"):
        robocasa_registry.SINGLE_STAGE_TASK_DATASETS = robocasa_registry.ATOMIC_TASK_DATASETS
    if not hasattr(robocasa_registry, "MULTI_STAGE_TASK_DATASETS"):
        robocasa_registry.MULTI_STAGE_TASK_DATASETS = robocasa_registry.COMPOSITE_TASK_DATASETS

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )
    from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
        TASK_MAX_STEPS,
        PolicyEvalConfig,
        create_robocasa_env,
        prepare_observation,
        validate_config,
    )
    from cosmos_policy.utils.utils import jpeg_encode_image, set_seed_everywhere

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []

    cfg = PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference",
        ckpt_path="nvidia/Cosmos-Policy-RoboCasa-Predict2-2B",
        config_file="cosmos_policy/config/config.py",
        use_wrist_image=True,
        num_wrist_images=1,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        dataset_stats_path="nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json",
        t5_text_embeddings_path="nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl",
        trained_with_image_aug=True,
        chunk_size=32,
        num_open_loop_steps=16,
        task_name=tasks[0],
        num_trials_per_task=args.num_trials_per_task,
        obj_instance_split=args.obj_instance_split,
        local_log_dir=str(out_dir),
        randomize_seed=False,
        data_collection=True,
        available_gpus="0",
        seed=args.seed,
        use_variance_scale=False,
        deterministic=True,
        run_id_note="phase3-robocasa-cosmos-deviation",
        ar_future_prediction=False,
        ar_value_prediction=False,
        use_jpeg_compression=True,
        flip_images=True,
        num_denoising_steps_action=args.num_denoising_steps_action,
        num_denoising_steps_future_state=args.num_denoising_steps_future_state,
        num_denoising_steps_value=1,
    )
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    model, cosmos_config = get_model(cfg)
    assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size, (
        f"Mismatch between checkpoint chunk {cosmos_config.dataloader_train.dataset.chunk_size} and cfg {cfg.chunk_size}"
    )

    total_successes = 0
    total_episodes = 0
    global_ep = 0

    for task_name in tqdm(tasks, desc="RoboCasa tasks"):
        cfg.task_name = task_name
        validate_config(cfg)
        max_steps = int(args.max_steps_override or TASK_MAX_STEPS.get(task_name, 500))

        for trial_idx in tqdm(range(args.num_trials_per_task), desc=task_name, leave=False):
            seed = cfg.seed * trial_idx * 256 if cfg.deterministic else None
            if args.debug_timing:
                print(f"[debug] create_env task={task_name} trial={trial_idx}", flush=True)
            env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=trial_idx)
            if args.debug_timing:
                print(f"[debug] reset_start task={task_name} trial={trial_idx}", flush=True)
            env.reset()
            if args.debug_timing:
                print(f"[debug] reset_done task={task_name} trial={trial_idx}", flush=True)
            task_description = env.get_ep_meta()["lang"]
            model_task_description = task_description.strip().rstrip(".").lower()

            obs = None
            for _ in range(10):
                dummy = np.zeros(env.action_spec[0].shape)
                obs, _, _, _ = env.step(dummy)
            if args.debug_timing:
                print(f"[debug] warmup_done task={task_name} trial={trial_idx}", flush=True)
            assert obs is not None

            action_queue: deque[np.ndarray] = deque()
            primary_images: list[np.ndarray] = []
            secondary_images: list[np.ndarray] = []
            wrist_images: list[np.ndarray] = []
            proprio_list: list[np.ndarray] = []
            actions_padded: list[np.ndarray] = []
            actions_7d: list[np.ndarray] = []
            query_t: list[int] = []
            query_primary: list[np.ndarray] = []
            query_secondary: list[np.ndarray] = []
            query_wrist: list[np.ndarray] = []
            query_proprio: list[np.ndarray] = []
            query_chunks_16: list[np.ndarray] = []
            query_chunks_full: list[np.ndarray] = []
            query_future_primary: list[np.ndarray] = []
            query_future_secondary: list[np.ndarray] = []
            query_future_wrist: list[np.ndarray] = []
            query_values: list[float] = []
            success = False

            for t in range(max_steps):
                observation = prepare_observation(obs, cfg.flip_images)
                dense_t = len(primary_images)
                primary_images.append(observation["primary_image"])
                secondary_images.append(observation["secondary_image"])
                wrist_images.append(observation["wrist_image"])
                proprio_list.append(observation["proprio"])

                if not action_queue:
                    query_start = time.perf_counter()
                    if args.debug_timing:
                        print(f"[debug] get_action_start task={task_name} trial={trial_idx} t={t}", flush=True)
                    result = get_action(
                        cfg,
                        model,
                        dataset_stats,
                        observation,
                        model_task_description,
                        seed=cfg.seed,
                        randomize_seed=False,
                        num_denoising_steps_action=cfg.num_denoising_steps_action,
                        generate_future_state_and_value_in_parallel=True,
                    )
                    if args.debug_timing:
                        elapsed = time.perf_counter() - query_start
                        print(f"[debug] get_action_done task={task_name} trial={trial_idx} t={t} sec={elapsed:.2f}", flush=True)
                    action_chunk = np.asarray(result["actions"], dtype=np.float32).reshape(cfg.chunk_size, 7)
                    action_prefix = action_chunk[: cfg.num_open_loop_steps]
                    future = result["future_image_predictions"]
                    if future.get("future_image") is None:
                        raise RuntimeError("Cosmos did not return primary future_image")

                    action_queue.extend(action_prefix)
                    query_t.append(dense_t)
                    query_primary.append(observation["primary_image"])
                    query_secondary.append(observation["secondary_image"])
                    query_wrist.append(observation["wrist_image"])
                    query_proprio.append(observation["proprio"])
                    query_chunks_16.append(action_prefix.astype(np.float32))
                    query_chunks_full.append(action_chunk.astype(np.float32))
                    query_future_primary.append(future["future_image"])
                    if future.get("future_image2") is not None:
                        query_future_secondary.append(future["future_image2"])
                    if future.get("future_wrist_image") is not None:
                        query_future_wrist.append(future["future_wrist_image"])
                    query_values.append(float(result.get("value_prediction", 0.0)))

                action_7d = np.asarray(action_queue.popleft(), dtype=np.float32)
                env_action = _pad_env_action(action_7d, int(env.action_dim))
                actions_7d.append(action_7d)
                actions_padded.append(env_action.astype(np.float32))
                obs, _, _, _ = env.step(env_action)
                if env._check_success():
                    success = True
                    break

            global_ep += 1
            total_episodes += 1
            total_successes += int(success)
            collected: dict[str, Any] = {
                "primary_images": np.stack(primary_images, axis=0).astype(np.uint8),
                "secondary_images": np.stack(secondary_images, axis=0).astype(np.uint8),
                "wrist_images": np.stack(wrist_images, axis=0).astype(np.uint8),
                "proprio": np.stack(proprio_list, axis=0).astype(np.float32),
                "actions": np.stack(actions_padded, axis=0).astype(np.float32),
                "actions_7d": np.stack(actions_7d, axis=0).astype(np.float32),
                "query_t": np.asarray(query_t, dtype=np.int32),
                "query_primary_images": np.stack(query_primary, axis=0).astype(np.uint8),
                "query_secondary_images": np.stack(query_secondary, axis=0).astype(np.uint8),
                "query_wrist_images": np.stack(query_wrist, axis=0).astype(np.uint8),
                "query_proprio": np.stack(query_proprio, axis=0).astype(np.float32),
                "query_action_chunks": np.stack(query_chunks_16, axis=0).astype(np.float32),
                "query_policy_action_chunks_full": np.stack(query_chunks_full, axis=0).astype(np.float32),
                "query_values": np.asarray(query_values, dtype=np.float32),
                "query_future_primary_images": np.stack(query_future_primary, axis=0).astype(np.uint8),
                "success": success,
            }
            if query_future_secondary:
                collected["query_future_secondary_images"] = np.stack(query_future_secondary, axis=0).astype(np.uint8)
            if query_future_wrist:
                collected["query_future_wrist_images"] = np.stack(query_future_wrist, axis=0).astype(np.uint8)

            h5_path = _write_episode_h5(
                out_dir,
                task_name,
                global_ep,
                success,
                task_description,
                collected,
                jpeg_encode_image,
                cfg.jpeg_compress,
            )
            row = {
                "task_name": task_name,
                "trial_idx": trial_idx,
                "episode": global_ep,
                "success": success,
                "steps": len(actions_7d),
                "queries": len(query_t),
                "task_description": task_description,
                "model_task_description": model_task_description,
                "h5": str(h5_path),
            }
            rows.append(row)
            print(row, flush=True)
            env.close()

    summary = {
        "mode": "cosmos_policy_robocasa_deviation",
        "tasks": tasks,
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": total_successes / max(1, total_episodes),
        "num_trials_per_task": args.num_trials_per_task,
        "policy_ckpt": cfg.ckpt_path,
        "policy_native_chunk_size": cfg.chunk_size,
        "idm_reference_horizon": cfg.num_open_loop_steps,
        "note": "query_action_chunks stores the first 16 policy actions; query_policy_action_chunks_full stores the native 32-action Cosmos chunk.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out_dir / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["task_name"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
