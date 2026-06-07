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
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from experiments.idm.model.idm import FeatureIDM


def _add_cosmos_policy_to_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _encode_image(model, processor, image: np.ndarray, device: torch.device) -> torch.Tensor:
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    batch = processor(images=[pil], return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            output = model.get_image_features(**batch)
        else:
            output = model(**batch)
    if isinstance(output, torch.Tensor):
        features = output
    elif getattr(output, "pooler_output", None) is not None:
        features = output.pooler_output
    elif hasattr(output, "last_hidden_state"):
        features = output.last_hidden_state.mean(dim=1)
    else:
        raise RuntimeError("Encoder output has neither pooler_output nor last_hidden_state")
    return torch.nn.functional.normalize(features.float(), dim=-1)


def _load_idm(checkpoint_path: Path, device: torch.device) -> FeatureIDM:
    checkpoint = torch.load(checkpoint_path.expanduser(), map_location="cpu", weights_only=False)
    idm = FeatureIDM(
        feature_dim=int(checkpoint["feature_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        proprio_dim=int(checkpoint["proprio_dim"]),
        hidden_dim=int(checkpoint["args"]["hidden_dim"]),
        depth=int(checkpoint["args"]["depth"]),
    ).to(device)
    idm.load_state_dict(checkpoint["model_state"])
    idm.eval()
    return idm


def _idm_action_chunk(
    idm: FeatureIDM,
    encoder,
    processor,
    current_image: np.ndarray,
    future_image: np.ndarray,
    proprio: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    z_current = _encode_image(encoder, processor, current_image, device)
    z_future = _encode_image(encoder, processor, future_image, device)
    proprio_t = torch.as_tensor(proprio.reshape(1, -1), dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = idm(z_current, z_future, proprio_t).detach().cpu().numpy()[0].astype(np.float32)
    return pred.reshape(16, 7)


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
        h5.attrs["policy_mode"] = "idm_from_cosmos_future"
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosmos-policy-repo", default="/home/ubuntu/robotics/repos/cosmos-policy")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--idm-checkpoint", required=True)
    parser.add_argument("--encoder", default="google/siglip-base-patch16-224")
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-ids", default="4,8")
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--run-id-note", default="idm-policy-from-cosmos-future")
    parser.add_argument("--clip-actions", action="store_true")
    parser.add_argument("--clip-min", type=float, default=-1.0)
    parser.add_argument("--clip-max", type=float, default=1.0)
    parser.add_argument("--num-denoising-steps-action", type=int, default=5)
    parser.add_argument("--env-img-res", type=int, default=256)
    args = parser.parse_args()

    _add_cosmos_policy_to_path(args.cosmos_policy_repo)

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
        chunk_size=16,
        num_open_loop_steps=16,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idm = _load_idm(Path(args.idm_checkpoint), device)
    processor = AutoProcessor.from_pretrained(args.encoder)
    encoder = AutoModel.from_pretrained(args.encoder).to(device).eval()

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task_ids = [int(x.strip()) for x in args.task_ids.split(",") if x.strip()]
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    rows = []
    total_episodes = 0
    total_successes = 0

    for task_id in tqdm(task_ids, desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=args.env_img_res)
        task_successes = 0

        for trial_idx in tqdm(range(args.num_trials_per_task), desc=f"task {task_id}", leave=False):
            env.reset()
            obs = env.set_init_state(initial_states[trial_idx])
            action_queue: deque[np.ndarray] = deque(maxlen=cfg.num_open_loop_steps)
            cosmos_action_queue: deque[np.ndarray] = deque(maxlen=cfg.num_open_loop_steps)
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
            query_idm_chunks: list[np.ndarray] = []
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
                    result = get_action(
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
                    future = result["future_image_predictions"]
                    future_image = future.get("future_image")
                    if future_image is None:
                        raise RuntimeError("Cosmos did not return future_image; cannot query IDM policy")
                    idm_chunk = _idm_action_chunk(
                        idm,
                        encoder,
                        processor,
                        observation["primary_image"],
                        future_image,
                        observation["proprio"],
                        device,
                    )
                    if args.clip_actions:
                        idm_chunk = np.clip(idm_chunk, args.clip_min, args.clip_max)
                    cosmos_chunk = np.asarray(result["actions"], dtype=np.float32).reshape(16, 7)
                    action_queue.extend(idm_chunk)
                    cosmos_action_queue.extend(cosmos_chunk)
                    future_predictions.append(future)
                    query_t.append(t - wait_steps)
                    query_primary.append(observation["primary_image"])
                    query_wrist.append(observation["wrist_image"])
                    query_proprio.append(observation["proprio"])
                    query_idm_chunks.append(idm_chunk.astype(np.float32))
                    query_cosmos_chunks.append(cosmos_chunk.astype(np.float32))
                    query_values.append(float(result.get("value_prediction", 0.0)))
                    query_future_primary.append(future_image)
                    if future.get("future_wrist_image") is not None:
                        query_future_wrist.append(future["future_wrist_image"])
                    print(
                        {
                            "task": task_id,
                            "trial": trial_idx,
                            "t": t,
                            "query_sec": round(time.time() - query_start, 3),
                            "idm_l2": float(np.linalg.norm(idm_chunk.reshape(-1))),
                            "cosmos_l2": float(np.linalg.norm(cosmos_chunk.reshape(-1))),
                            "idm_cosmos_l2": float(np.linalg.norm(idm_chunk.reshape(-1) - cosmos_chunk.reshape(-1))),
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
            task_successes += int(success)
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
                "query_action_chunks": np.stack(query_idm_chunks, axis=0),
                "query_idm_action_chunks": np.stack(query_idm_chunks, axis=0),
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
                args.run_id_note,
                task_description,
                collected,
                jpeg_encode_image,
                cfg.jpeg_compress,
            )
            row = {
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

        print(
            {
                "task_id": task_id,
                "task_successes": task_successes,
                "task_episodes": args.num_trials_per_task,
                "task_success_rate": task_successes / max(1, args.num_trials_per_task),
            },
            flush=True,
        )

    summary = {
        "mode": "idm_from_cosmos_future",
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": total_successes / max(1, total_episodes),
        "task_ids": task_ids,
        "num_trials_per_task": args.num_trials_per_task,
        "idm_checkpoint": str(Path(args.idm_checkpoint).expanduser()),
        "encoder": args.encoder,
        "clip_actions": args.clip_actions,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out_dir / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["task_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
