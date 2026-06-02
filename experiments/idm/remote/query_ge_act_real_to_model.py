#!/usr/bin/env python3
"""Generate GE-Act real -> model IDM rows from raw LIBERO demonstrations.

For sampled LIBERO demo states:

  C = real current frame
  a = GE-Act predicted action chunk
  P = GE-Act decoded predicted future frame
  R = real future frame from the same LIBERO demo, optional reference

This mirrors the Cosmos `real -> model` export, but uses the GE-Act pipeline
with `return_action=True` and `return_video=True`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from einops import rearrange
from PIL import Image


ACTION_SPACE = "libero_7d_eef_delta_gripper"
MODEL_ID = "ge_act_libero_spatial"


def _import_ge(ge_repo: Path):
    sys.path.insert(0, str(ge_repo))
    sys.path.insert(0, str(ge_repo / "experiments"))
    from eval_libero import InferenceLibero

    return InferenceLibero


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


def _rot180(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame[::-1, ::-1])


def _to_uint8_from_model_video(frame: torch.Tensor) -> np.ndarray:
    # GE decoded frames are C,H,W in [-1, 1].
    frame = frame.detach().float().cpu().clamp(-1, 1)
    frame = ((frame + 1.0) * 127.5).round().byte()
    frame = frame.permute(1, 2, 0).numpy()
    return np.ascontiguousarray(frame)


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array).astype(np.uint8)).save(path)


def _resize(frame: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    image = Image.fromarray(frame.astype(np.uint8)).resize((width, height), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _build_input_images(primary: np.ndarray, wrist: np.ndarray, device: str) -> torch.Tensor:
    primary_t = torch.tensor(primary.copy()).to(device).permute(2, 0, 1).unsqueeze(0)
    wrist_t = torch.tensor(wrist.copy()).to(device).permute(2, 0, 1).unsqueeze(0)
    return torch.cat([primary_t, wrist_t], dim=0)


@torch.no_grad()
def _predict_action_and_video(
    libero_infer,
    img_obs: torch.Tensor,
    prompt: str,
    state_raw: np.ndarray | None,
    seed: int,
    return_action: bool = True,
):
    if img_obs.dtype == torch.uint8:
        obs = img_obs.float() / 255.0 * 2.0 - 1.0
    else:
        obs = img_obs
        if obs.max().detach().cpu() > 1.0:
            obs = obs / 255.0 * 2.0 - 1.0

    obs = obs.to(libero_infer.device, dtype=libero_infer.dtype)

    history_action_state = None
    if libero_infer.add_state and state_raw is not None:
        state = (torch.tensor(state_raw) - libero_infer.states_min) / (
            libero_infer.states_max - libero_infer.states_min + 1e-6
        )
        state = state * 2 - 1
        state = torch.cat((torch.zeros([1, libero_infer.basic_action_dim]), state), dim=1)
        history_action_state = state.to(libero_infer.device, dtype=libero_infer.dtype)
        while len(history_action_state.shape) < 3:
            history_action_state = history_action_state.unsqueeze(dim=0)

    # Match GE-Act eval memory behavior for a single queried state.
    obs_history = [obs] * libero_infer.n_prev
    obs_tensor = torch.stack(obs_history, dim=1)
    obs_tensor = rearrange(obs_tensor, "v t c h w -> c v t h w")
    obs_tensor = obs_tensor.unsqueeze(0)
    obs_tensor = rearrange(obs_tensor, "b c v t h w -> (b v) c t h w")

    v, c, h, w = obs.shape
    pred_all = libero_infer.pipeline.infer(
        image=obs_tensor,
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=libero_infer.num_inference_steps,
        decode_timestep=0.03,
        decode_noise_scale=0.025,
        guidance_scale=1.0,
        height=h,
        width=w,
        n_view=v,
        return_action=return_action,
        return_video=True,
        chunk=(libero_infer.chunk - 1) // libero_infer.TEMPORAL_DOWN_RATIO + 1,
        action_chunk=libero_infer.action_chunk,
        history_action_state=history_action_state if libero_infer.add_state else None,
        noise_seed=seed,
        pixel_wise_timestep=libero_infer.args.pixel_wise_timestep,
        n_chunk=1,
        n_prev=libero_infer.n_prev,
        action_dim=libero_infer.action_dim,
    )[0]

    actions_pred = None
    if return_action:
        actions_pred = pred_all["action"].detach().cpu()[0]
        actions_pred = actions_pred[:, : libero_infer.basic_action_dim]
        actions_pred = (actions_pred + 1) / 2
        actions_pred = actions_pred * (libero_infer.act_max - libero_infer.act_min + 1e-6) + libero_infer.act_min

    video = pred_all["video"].detach().cpu()
    # video shape: (b*v), c, t, h, w. With v=2: primary view first, wrist second.
    generated_primary = _to_uint8_from_model_video(video[0, :, -1])
    generated_wrist = _to_uint8_from_model_video(video[1, :, -1]) if video.shape[0] > 1 else None
    return None if actions_pred is None else actions_pred.numpy(), generated_primary, generated_wrist, tuple(video.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ge-repo", default="/home/ubuntu/robotics/repos/Genie-Envisioner")
    parser.add_argument("--config-file", default="/home/ubuntu/robotics/repos/Genie-Envisioner/configs/ltx_model/libero/action_model_libero.yaml")
    parser.add_argument("--ckpt-path", default="/home/ubuntu/robotics/repos/Genie-Envisioner/checkpoints/agibot_world/Genie-Envisioner/ge_act_libero_spatial.safetensors")
    parser.add_argument("--input", required=True, help="Raw LIBERO HDF5 file or directory")
    parser.add_argument("--out", required=True, help="Output dataset directory")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--stride", type=int, default=36)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--exec-step", type=int, default=8)
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--resize-to-config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-only-diagnostic", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser()
    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    ge_repo = Path(args.ge_repo).expanduser()
    InferenceLibero = _import_ge(ge_repo)
    os.chdir(ge_repo)
    libero_infer = InferenceLibero(
        config_file=args.config_file,
        output_dir=str(out_dir / "_ge_eval_logs"),
        task_suite_name=args.task_suite_name,
        model_path=args.ckpt_path,
        exec_step=args.exec_step,
        threshold=args.threshold,
        device=args.device,
    )
    libero_infer.prepare_models()

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
                    robot_states = demo["robot_states"]
                    rewards = demo["rewards"] if "rewards" in demo else None
                    success = bool(rewards is not None and len(rewards) > 0 and rewards[-1] > 0)
                    horizon = int(libero_infer.action_chunk)
                    max_start = len(actions) - horizon
                    if max_start <= 0:
                        continue

                    for t in range(0, max_start, args.stride):
                        if count >= args.limit:
                            break

                        primary = _rot180(demo["obs/agentview_rgb"][t])
                        wrist = _rot180(demo["obs/eye_in_hand_rgb"][t])
                        if args.resize_to_config:
                            primary = _resize(primary, tuple(libero_infer.args.data["train"]["sample_size"]))
                            wrist = _resize(wrist, tuple(libero_infer.args.data["train"]["sample_size"]))
                        img_obs = _build_input_images(primary, wrist, libero_infer.device)
                        ge_state = np.concatenate(
                            [
                                np.asarray(demo["obs/ee_states"][t], dtype=np.float32),
                                np.asarray(demo["obs/gripper_states"][t], dtype=np.float32),
                            ]
                        )

                        action_chunk, future_primary, future_wrist, video_shape = _predict_action_and_video(
                            libero_infer,
                            img_obs,
                            task_name,
                            ge_state,
                            args.seed,
                            return_action=not args.video_only_diagnostic,
                        )
                        video_shapes.append(video_shape)

                        mode = "ge_act_video_only" if args.video_only_diagnostic else "ge_act_real_model"
                        sample_id = f"{h5_path.stem}--{demo_key}--{mode}--q{count:06d}--t{t:05d}"
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
                            "proprio_t": _as_jsonable(robot_states[t]),
                            "action_chunk": None if action_chunk is None else _as_jsonable(action_chunk),
                            "action_space": ACTION_SPACE,
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
        "dataset_version": "idm-ge-act-video-only-diagnostic-v0" if args.video_only_diagnostic else "idm-ge-act-real-model-v0",
        "manifest": str(manifest_path),
        "source_files": [str(path) for path in source_files],
        "model_id": MODEL_ID,
        "action_space": ACTION_SPACE,
        "source_type": "real_model",
        "horizon_k": int(libero_infer.action_chunk),
        "counts": {"real_model": count},
        "video_only_diagnostic": args.video_only_diagnostic,
        "frame_convention": "ge_libero_train_rot180",
        "hdf5_frame_transform": "rot180",
        "generated_frame_transform": "none",
        "video_shapes": [list(shape) for shape in sorted(set(video_shapes))],
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
