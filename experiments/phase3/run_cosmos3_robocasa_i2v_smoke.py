#!/usr/bin/env python3
"""Small Cosmos3-Nano image-to-video smoke test on RoboCasa trajectory frames."""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CASE_GLOBS = [
    ("open_left_drawer", "OpenDrawer/**/*open_the_left_drawer*.hdf5"),
    ("close_left_drawer", "CloseDrawer/**/*close_the_left_drawer*.hdf5"),
    ("turn_on_sink_faucet", "TurnOnSinkFaucet/**/*.hdf5"),
]

NEGATIVE_PROMPT = (
    "blurry, low quality, flickering, distorted robot arm, malformed gripper, "
    "objects disappearing, object teleportation, changing camera viewpoint, text, watermark"
)


def _decode_jpeg(value: object) -> Image.Image:
    if isinstance(value, np.ndarray):
        value = value.tobytes()
    if isinstance(value, str):
        value = value.encode("latin1")
    return Image.open(io.BytesIO(value)).convert("RGB")


def _fit_image(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.BICUBIC)


def _sample_video_frames(video_path: Path, count: int = 5) -> list[Image.Image]:
    frames = list(iio.imiter(video_path))
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    indices = np.linspace(0, len(frames) - 1, min(count, len(frames))).round().astype(int)
    return [Image.fromarray(frames[int(i)].astype(np.uint8)).convert("RGB") for i in indices]


def _font(size: int = 16) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    font = _font(16)
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle((bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3), fill=(255, 255, 255))
    draw.text((x, y), text, fill=(20, 20, 20), font=font)


def _make_contact_sheet(
    *,
    case_name: str,
    instruction: str,
    input_image: Image.Image,
    real_future: Image.Image,
    generated_frames: list[Image.Image],
    output_path: Path,
    cell: int,
) -> None:
    columns = 2 + len(generated_frames)
    header_h = 70
    sheet = Image.new("RGB", (columns * cell, header_h + cell), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), case_name, fill=(0, 0, 0), font=_font(16))
    draw.text((8, 34), instruction[:180], fill=(40, 40, 40), font=_font(14))

    tiles = [input_image, real_future, *generated_frames]
    labels = ["input t0", "real future", *[f"gen f{i}" for i in range(len(generated_frames))]]
    for idx, (tile, label) in enumerate(zip(tiles, labels)):
        x = idx * cell
        sheet.paste(tile.resize((cell, cell), Image.Resampling.BICUBIC), (x, header_h))
        _label(draw, (x + 8, header_h + 8), label)
    sheet.save(output_path)


def _safe_case_name(name: str, path: Path) -> str:
    ep_match = re.search(r"--ep=(\d+)", path.name)
    ep = ep_match.group(1) if ep_match else "unknown"
    return f"{name}_ep{ep}"


def _load_robocasa_frames(
    path: Path,
    *,
    camera_key: str,
    start_t: int,
    future_t: int,
) -> tuple[Image.Image, Image.Image, int, str]:
    with h5py.File(path, "r") as h5:
        images = h5[camera_key]
        start_index = min(start_t, len(images) - 1)
        future_index = min(start_index + future_t, len(images) - 1)
        current = _decode_jpeg(images[start_index])
        future = _decode_jpeg(images[future_index])
        instruction = str(h5.attrs.get("task_description", path.stem))
    return current, future, future_index, instruction


def _resolve_cases(data_root: Path, case_limit: int) -> list[tuple[str, Path]]:
    cases: list[tuple[str, Path]] = []
    for name, pattern in DEFAULT_CASE_GLOBS:
        matches = sorted((data_root / "all_episodes").glob(pattern))
        if matches:
            cases.append((name, matches[0]))
        if len(cases) >= case_limit:
            break
    return cases


def run_case(
    *,
    pipe: Cosmos3OmniPipeline,
    checkpoint: Path,
    out_dir: Path,
    case_name: str,
    hdf5_path: Path,
    camera_key: str,
    prompt_suffix: str,
    seed: int,
    num_frames: int,
    num_steps: int,
    fps: int,
    size: int,
    start_t: int,
    future_t: int,
    guidance_scale: float,
    add_templates: bool,
) -> dict:
    current_raw, future_raw, future_index, instruction = _load_robocasa_frames(
        hdf5_path, camera_key=camera_key, start_t=start_t, future_t=future_t
    )
    name = _safe_case_name(case_name, hdf5_path)
    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)

    input_image = _fit_image(current_raw, size)
    real_future = _fit_image(future_raw, size)
    input_path = case_dir / f"{camera_key}_input_t{start_t}.png"
    future_path = case_dir / f"{camera_key}_real_future_t{future_index}.png"
    input_image.save(input_path)
    real_future.save(future_path)

    prompt = (
        "A realistic robot simulation video from a fixed RoboCasa kitchen benchmark camera. "
        f"The task is: {instruction}. "
        "Continue from the exact starting frame. Keep the same camera, lighting, kitchen, robot, "
        "objects, and object identities. Show the robot beginning to carry out the instruction. "
        f"{prompt_suffix}"
    ).strip()

    generator = torch.Generator(device="cuda").manual_seed(seed)
    output_path = case_dir / "cosmos3_nano_i2v.mp4"
    start = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=input_image,
        num_frames=num_frames,
        height=size,
        width=size,
        fps=fps,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        enable_sound=False,
        add_resolution_template=add_templates,
        add_duration_template=add_templates,
        enable_safety_check=False,
        generator=generator,
    )
    elapsed_s = time.time() - start
    export_to_video(result.video, str(output_path), fps=fps, macro_block_size=1)

    generated_frames = _sample_video_frames(output_path)
    contact_path = case_dir / "qualitative_contact_sheet.png"
    _make_contact_sheet(
        case_name=name,
        instruction=instruction,
        input_image=input_image,
        real_future=real_future,
        generated_frames=generated_frames,
        output_path=contact_path,
        cell=size,
    )

    return {
        "case": name,
        "checkpoint": str(checkpoint),
        "hdf5_path": str(hdf5_path),
        "camera_key": camera_key,
        "instruction": instruction,
        "seed": seed,
        "num_frames": num_frames,
        "num_steps": num_steps,
        "fps": fps,
        "height": size,
        "width": size,
        "start_t": start_t,
        "future_reference_t": future_index,
        "future_offset": future_t,
        "guidance_scale": guidance_scale,
        "add_templates": add_templates,
        "elapsed_s": elapsed_s,
        "input_path": str(input_path),
        "real_future_path": str(future_path),
        "video_path": str(output_path),
        "contact_sheet_path": str(contact_path),
        "prompt": prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/ubuntu/robotics/checkpoints/Cosmos3-Nano")
    parser.add_argument("--data-root", default="/home/ubuntu/robotics/data/robocasa-cosmos-policy")
    parser.add_argument("--out", default="/home/ubuntu/robotics/runs/cosmos3-nano-robocasa-i2v-smoke")
    parser.add_argument("--camera-key", default="primary_images_jpeg")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--start-t", type=int, default=0)
    parser.add_argument("--future-t", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--add-templates", action="store_true")
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--case-limit", type=int, default=3)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser()
    data_root = Path(args.data_root).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _resolve_cases(data_root, args.case_limit)
    if not cases:
        raise FileNotFoundError(f"No RoboCasa HDF5 cases found under {data_root}")

    print(f"Loading Cosmos3-Nano from {checkpoint}")
    pipe = Cosmos3OmniPipeline.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        safety_checker=None,
        enable_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.to("cuda")
    print("Loaded pipeline")

    rows = []
    for index, (case_name, path) in enumerate(cases):
        row = run_case(
            pipe=pipe,
            checkpoint=checkpoint,
            out_dir=out_dir,
            case_name=f"{index:02d}_{case_name}",
            hdf5_path=path,
            camera_key=args.camera_key,
            prompt_suffix=args.prompt_suffix,
            seed=args.seed + index,
            num_frames=args.num_frames,
            num_steps=args.num_steps,
            fps=args.fps,
            size=args.size,
            start_t=args.start_t,
            future_t=args.future_t,
            guidance_scale=args.guidance_scale,
            add_templates=args.add_templates,
        )
        rows.append(row)
        print(json.dumps(row, indent=2))

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({"runs": rows}, indent=2) + "\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
