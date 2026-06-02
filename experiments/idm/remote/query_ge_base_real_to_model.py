#!/usr/bin/env python3
"""Generate GE-Base real -> model future rows from raw LIBERO demonstrations.

For sampled LIBERO demo states:

  C = real current frame/history
  P = GE-Base decoded predicted future frame
  R = real future frame from the same LIBERO demo, optional reference

GE-Base does not produce an action chunk. This exporter is for prediction
honesty / future-quality probes, not supervised IDM action training by itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from PIL import Image


MODEL_ID = "ge_base_fast_v0.1"


def _import_ge(ge_repo: Path):
    sys.path.insert(0, str(ge_repo))
    sys.path.insert(0, str(ge_repo / "video_gen_examples"))
    from infer import prepare_model

    return prepare_model


def _load_config(config_file: Path, ckpt_path: Path, pretrained_path: Path) -> Namespace:
    with config_file.open("r") as f:
        cfg = yaml.safe_load(f)
    cfg["pretrained_model_name_or_path"] = str(pretrained_path)
    cfg["return_action"] = False
    cfg["return_video"] = True
    cfg["train_mode"] = "video_only"
    cfg["add_state"] = False
    cfg["load_weights"] = True
    cfg["diffusion_model"]["model_path"] = str(ckpt_path)
    cfg["diffusion_model"]["config"]["action_expert"] = False
    cfg["data"]["train"]["data_roots"] = ["/unused"]
    cfg["data"]["val"]["data_roots"] = ["/unused"]
    return Namespace(**cfg)


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


def _rot180(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame[::-1, ::-1])


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array).astype(np.uint8)).save(path)


def _resize(frame: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    image = Image.fromarray(frame.astype(np.uint8)).resize((width, height), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _to_uint8_from_model_video(frame: torch.Tensor) -> np.ndarray:
    frame = frame.detach().float().cpu().clamp(-1, 1)
    frame = ((frame + 1.0) * 127.5).round().byte()
    return np.ascontiguousarray(frame.permute(1, 2, 0).numpy())


def _build_memory(primary: np.ndarray, wrist: np.ndarray, n_prev: int, device: str) -> torch.Tensor:
    primary_t = torch.tensor(primary.copy()).to(device).permute(2, 0, 1)
    wrist_t = torch.tensor(wrist.copy()).to(device).permute(2, 0, 1)
    obs = torch.stack([primary_t, wrist_t], dim=0).unsqueeze(2)
    obs = obs.float() / 255.0 * 2.0 - 1.0
    return obs.repeat(1, 1, n_prev, 1, 1)


@torch.no_grad()
def _predict_video(pipe, vae, config: Namespace, obs: torch.Tensor, prompt: str, seed: int):
    device = obs.device
    v, _c, _t, h, w = obs.shape
    temporal_down = vae.temporal_compression_ratio
    pred_all = pipe.infer(
        image=obs,
        prompt=[prompt],
        negative_prompt="",
        num_inference_steps=config.num_inference_step,
        decode_timestep=0.03,
        decode_noise_scale=0.025,
        guidance_scale=1.0,
        height=h,
        width=w,
        n_view=v,
        return_action=False,
        return_video=True,
        chunk=(config.data["train"]["chunk"] - 1) // temporal_down + 1,
        noise_seed=seed,
        pixel_wise_timestep=config.pixel_wise_timestep,
        n_chunk=1,
        n_prev=config.data["train"]["n_previous"],
    )[0]
    video = pred_all["video"].detach().cpu()
    generated_primary = _to_uint8_from_model_video(video[0, :, -1])
    generated_wrist = _to_uint8_from_model_video(video[1, :, -1]) if video.shape[0] > 1 else None
    return generated_primary, generated_wrist, tuple(video.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ge-repo", default="/home/ubuntu/robotics/repos/Genie-Envisioner")
    parser.add_argument("--config-file", default="/home/ubuntu/robotics/repos/Genie-Envisioner/configs/ltx_model/libero/video_model_libero.yaml")
    parser.add_argument("--ckpt-path", default="/home/ubuntu/robotics/repos/Genie-Envisioner/checkpoints/agibot_world/Genie-Envisioner/GE_base_fast_v0.1.safetensors")
    parser.add_argument("--pretrained-path", default="/home/ubuntu/robotics/checkpoints/LTX-Video")
    parser.add_argument("--input", required=True, help="Raw LIBERO HDF5 file or directory")
    parser.add_argument("--out", required=True, help="Output dataset directory")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--stride", type=int, default=36)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resize-to-config", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser()
    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    ge_repo = Path(args.ge_repo).expanduser()
    prepare_model = _import_ge(ge_repo)
    os.chdir(ge_repo)
    config = _load_config(Path(args.config_file), Path(args.ckpt_path), Path(args.pretrained_path))
    _tokenizer, _text_encoder, vae, _diffusion_model, _scheduler, pipe = prepare_model(
        config,
        dtype=torch.bfloat16,
        device=args.device,
    )

    sample_size = tuple(config.data["train"]["sample_size"])
    n_prev = int(config.data["train"]["n_previous"])
    source_files = _iter_hdf5_files(Path(args.input).expanduser())
    count = 0
    video_shapes: list[tuple[int, ...]] = []

    with manifest_path.open("w") as manifest:
        for h5_path in source_files:
            task_name = _task_from_filename(h5_path)
            with h5py.File(h5_path, "r") as h5:
                for demo_key in _sorted_demo_keys(h5):
                    demo = h5["data"][demo_key]
                    actions = demo["actions"]
                    horizon = int(config.data["train"].get("action_chunk", 36))
                    max_start = len(actions) - horizon
                    if max_start <= 0:
                        continue
                    rewards = demo["rewards"] if "rewards" in demo else None
                    success = bool(rewards is not None and len(rewards) > 0 and rewards[-1] > 0)

                    for t in range(0, max_start, args.stride):
                        if count >= args.limit:
                            break

                        primary = _rot180(demo["obs/agentview_rgb"][t])
                        wrist = _rot180(demo["obs/eye_in_hand_rgb"][t])
                        if args.resize_to_config:
                            primary = _resize(primary, sample_size)
                            wrist = _resize(wrist, sample_size)
                        obs = _build_memory(primary, wrist, n_prev, args.device)
                        future_primary, future_wrist, video_shape = _predict_video(
                            pipe,
                            vae,
                            config,
                            obs,
                            task_name,
                            args.seed,
                        )
                        video_shapes.append(video_shape)

                        sample_id = f"{h5_path.stem}--{demo_key}--ge_base_real_model--q{count:06d}--t{t:05d}"
                        future_primary_path = image_dir / f"{sample_id}_future.png"
                        _save_png(future_primary_path, future_primary)

                        future_wrist_rel = None
                        if future_wrist is not None:
                            future_wrist_path = image_dir / f"{sample_id}_wrist_future.png"
                            _save_png(future_wrist_path, future_wrist)
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
                            "horizon_k": horizon,
                            "effective_future_frame": "last_decoded_model_frame",
                            "image_t_hdf5_path": f"data/{demo_key}/obs/agentview_rgb",
                            "image_t_index": int(t),
                            "image_t_transform": "rot180",
                            "wrist_image_t_hdf5_path": f"data/{demo_key}/obs/eye_in_hand_rgb",
                            "wrist_image_t_index": int(t),
                            "wrist_image_t_transform": "rot180",
                            "real_future_hdf5_path": f"data/{demo_key}/obs/agentview_rgb",
                            "real_future_index": int(t + horizon),
                            "real_future_transform": "rot180",
                            "wrist_real_future_hdf5_path": f"data/{demo_key}/obs/eye_in_hand_rgb",
                            "wrist_real_future_index": int(t + horizon),
                            "wrist_real_future_transform": "rot180",
                            "model_future_path": future_primary_path.relative_to(out_dir).as_posix(),
                            "wrist_model_future_path": future_wrist_rel,
                            "action_chunk": None,
                            "action_space": None,
                            "success_label": success,
                            "ge_video_shape": list(video_shape),
                        }
                        manifest.write(json.dumps(row) + "\n")
                        manifest.flush()
                        count += 1
                    if count >= args.limit:
                        break
            if count >= args.limit:
                break

    metadata = {
        "dataset_version": "idm-ge-base-real-model-v0",
        "manifest": str(manifest_path),
        "source_files": [str(path) for path in source_files],
        "model_id": MODEL_ID,
        "source_type": "real_model",
        "horizon_k": int(config.data["train"].get("action_chunk", 36)),
        "counts": {"real_model": count},
        "frame_convention": "ge_libero_train_rot180",
        "hdf5_frame_transform": "rot180",
        "generated_frame_transform": "none",
        "video_shapes": [list(shape) for shape in sorted(set(video_shapes))],
        "notes": "GE-Base emits future video only; no action chunk is stored.",
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
