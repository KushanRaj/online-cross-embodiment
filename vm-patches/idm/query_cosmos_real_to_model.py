#!/usr/bin/env python3
"""Generate real -> model IDM rows from raw LIBERO demonstrations.

For sampled real LIBERO demo states, query Cosmos Policy once and store the
decoded predicted future image plus the model action chunk. Real frames stay as
HDF5 references; generated futures are saved as PNGs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig, validate_config
from cosmos_policy.utils.utils import set_seed_everywhere


ACTION_SPACE = "libero_7d_eef_delta_gripper"
MODEL_ID = "cosmos_policy_libero_predict2_2b"


def _task_from_filename(path: Path) -> str:
    stem = re.sub(r"_demo$", "", path.stem)
    return stem.replace("_", " ")


def _iter_hdf5_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.hdf5")) + sorted(path.rglob("*.h5"))


def _sorted_demo_keys(handle: h5py.File) -> list[str]:
    keys = list(handle["data"].keys())
    return sorted(keys, key=lambda k: int(k.split("_")[-1]) if k.startswith("demo_") else k)


def _as_jsonable(array) -> list:
    return np.asarray(array).astype(float).tolist()


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array).astype(np.uint8)).save(path)


def _build_cfg(args: argparse.Namespace) -> PolicyEvalConfig:
    return PolicyEvalConfig(
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
        chunk_size=args.chunk_size,
        num_open_loop_steps=args.chunk_size,
        task_suite_name="libero_spatial",
        task_ids=None,
        num_trials_per_task=1,
        local_log_dir=str(args.out),
        randomize_seed=False,
        data_collection=False,
        available_gpus="0",
        seed=args.seed,
        use_variance_scale=False,
        deterministic=True,
        run_id_note="real-to-model",
        ar_future_prediction=False,
        ar_value_prediction=False,
        use_jpeg_compression=True,
        flip_images=True,
        num_denoising_steps_action=args.num_denoising_steps_action,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw LIBERO HDF5 file or directory")
    parser.add_argument("--out", required=True, help="Output dataset directory")
    parser.add_argument("--limit", type=int, default=100, help="Max model queries")
    parser.add_argument("--stride", type=int, default=16, help="Sample every N timesteps in each demo")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--num-denoising-steps-action", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser()
    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    cfg = _build_cfg(args)
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    model, cosmos_config = get_model(cfg)
    assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size

    source_files = _iter_hdf5_files(Path(args.input).expanduser())
    count = 0

    with manifest_path.open("w") as manifest:
        for h5_path in source_files:
            task_name = _task_from_filename(h5_path)
            with h5py.File(h5_path, "r") as h5:
                for demo_key in _sorted_demo_keys(h5):
                    demo = h5["data"][demo_key]
                    actions = demo["actions"]
                    robot_states = demo["robot_states"]
                    rewards = demo["rewards"] if "rewards" in demo else None
                    success = bool(rewards is not None and len(rewards) > 0 and rewards[-1] > 0)
                    max_start = len(actions) - cfg.chunk_size
                    if max_start <= 0:
                        continue

                    for t in range(0, max_start, args.stride):
                        if count >= args.limit:
                            break
                        primary = demo["obs/agentview_rgb"][t]
                        wrist = demo["obs/eye_in_hand_rgb"][t]
                        if cfg.flip_images:
                            primary = np.flipud(primary)
                            wrist = np.flipud(wrist)
                        observation = {
                            "primary_image": primary,
                            "wrist_image": wrist,
                            "proprio": robot_states[t],
                        }
                        result = get_action(
                            cfg,
                            model,
                            dataset_stats,
                            observation,
                            task_name,
                            seed=cfg.seed,
                            randomize_seed=False,
                            num_denoising_steps_action=cfg.num_denoising_steps_action,
                            generate_future_state_and_value_in_parallel=True,
                        )
                        futures = result["future_image_predictions"]
                        sample_id = f"{h5_path.stem}--{demo_key}--real_model--q{count:06d}--t{t:05d}"
                        future_primary_path = image_dir / f"{sample_id}_future.png"
                        _save_png(future_primary_path, futures["future_image"])

                        future_wrist_rel = None
                        if "future_wrist_image" in futures:
                            future_wrist_path = image_dir / f"{sample_id}_wrist_future.png"
                            _save_png(future_wrist_path, futures["future_wrist_image"])
                            future_wrist_rel = future_wrist_path.relative_to(out_dir).as_posix()

                        row = {
                            "sample_id": sample_id,
                            "trajectory_id": f"{h5_path.stem}--{demo_key}",
                            "source_file": str(h5_path),
                            "source_type": "real_model",
                            "model_id": MODEL_ID,
                            "environment": "libero",
                            "task_name": task_name,
                            "instruction": task_name,
                            "camera_name": "agentview_rgb",
                            "window_start_t": int(t),
                            "horizon_k": int(cfg.chunk_size),
                            "image_t_hdf5_path": f"data/{demo_key}/obs/agentview_rgb",
                            "image_t_index": int(t),
                            "image_t_transform": "flipud",
                            "wrist_image_t_hdf5_path": f"data/{demo_key}/obs/eye_in_hand_rgb",
                            "wrist_image_t_index": int(t),
                            "wrist_image_t_transform": "flipud",
                            "real_future_hdf5_path": f"data/{demo_key}/obs/agentview_rgb",
                            "real_future_index": int(t + cfg.chunk_size),
                            "real_future_transform": "flipud",
                            "wrist_real_future_hdf5_path": f"data/{demo_key}/obs/eye_in_hand_rgb",
                            "wrist_real_future_index": int(t + cfg.chunk_size),
                            "wrist_real_future_transform": "flipud",
                            "model_future_path": future_primary_path.relative_to(out_dir).as_posix(),
                            "wrist_model_future_path": future_wrist_rel,
                            "proprio_t": _as_jsonable(robot_states[t]),
                            "action_chunk": _as_jsonable(result["actions"]),
                            "action_space": ACTION_SPACE,
                            "success_label": success,
                            "model_value": float(result.get("value_prediction", 0.0)),
                        }
                        manifest.write(json.dumps(row) + "\n")
                        count += 1
                    if count >= args.limit:
                        break
            if count >= args.limit:
                break

    metadata = {
        "dataset_version": "idm-real-model-v0",
        "manifest": str(manifest_path),
        "source_files": [str(path) for path in source_files],
        "model_id": MODEL_ID,
        "action_space": ACTION_SPACE,
        "source_type": "real_model",
        "horizon_k": cfg.chunk_size,
        "counts": {"real_model": count},
        "frame_convention": "cosmos_libero_eval_flipud",
        "hdf5_frame_transform": "flipud",
        "generated_frame_transform": "none",
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
