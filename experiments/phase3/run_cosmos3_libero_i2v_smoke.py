#!/usr/bin/env python3
"""Tiny Cosmos3-Nano image-to-video smoke test on LIBERO demo frames."""

from __future__ import annotations

import argparse
import json
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


DEFAULT_CASES = [
    (
        "between_plate_ramekin",
        "/home/ubuntu/robotics/data/libero/libero_spatial/"
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
        "Pick up the black bowl between the plate and the ramekin and place it on the plate.",
    ),
    (
        "table_center",
        "/home/ubuntu/robotics/data/libero/libero_spatial/"
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5",
        "Pick up the black bowl from the table center and place it on the plate.",
    ),
]


NEGATIVE_PROMPT = (
    "blurry, low quality, flickering, distorted robot arm, malformed gripper, "
    "objects disappearing, object teleportation, changing camera viewpoint, text, watermark"
)


def _sorted_demo_keys(handle: h5py.File) -> list[str]:
    keys = list(handle["data"].keys())
    return sorted(keys, key=lambda k: int(k.split("_")[-1]) if k.startswith("demo_") else k)


def _load_demo_frames(path: Path, demo_index: int, future_t: int) -> tuple[np.ndarray, np.ndarray, int]:
    with h5py.File(path, "r") as h5:
        demo_key = _sorted_demo_keys(h5)[demo_index]
        frames = h5[f"data/{demo_key}/obs/agentview_rgb"][()]
    future_index = min(int(future_t), len(frames) - 1)
    return frames[0], frames[future_index], future_index


def _fit_image(image: np.ndarray, size: int) -> Image.Image:
    pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    return pil.resize((size, size), Image.Resampling.BICUBIC)


def _sample_video_frames(video_path: Path, count: int = 4) -> list[Image.Image]:
    frames = list(iio.imiter(video_path))
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    indices = np.linspace(0, len(frames) - 1, min(count, len(frames))).round().astype(int)
    return [Image.fromarray(frames[int(i)].astype(np.uint8)).convert("RGB") for i in indices]


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    x, y = xy
    draw.rectangle((x - 4, y - 3, x + 7 * len(text) + 8, y + 20), fill=(255, 255, 255))
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
    header_h = 58
    sheet = Image.new("RGB", (columns * cell, header_h + cell), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), case_name, fill=(0, 0, 0))
    draw.text((8, 30), instruction[:150], fill=(40, 40, 40))

    tiles = [input_image, real_future, *generated_frames]
    labels = ["input t0", "real future", *[f"gen f{i}" for i in range(len(generated_frames))]]
    for idx, (tile, label) in enumerate(zip(tiles, labels)):
        x = idx * cell
        sheet.paste(tile.resize((cell, cell), Image.Resampling.BICUBIC), (x, header_h))
        _label(draw, (x + 8, header_h + 8), label)
    sheet.save(output_path)


def run_case(
    *,
    pipe: Cosmos3OmniPipeline,
    checkpoint: Path,
    out_dir: Path,
    name: str,
    hdf5_path: Path,
    instruction: str,
    prompt_suffix: str,
    seed: int,
    num_frames: int,
    num_steps: int,
    fps: int,
    size: int,
    future_t: int,
    demo_index: int,
    guidance_scale: float,
) -> dict:
    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)

    current, future, future_index = _load_demo_frames(hdf5_path, demo_index, future_t)
    input_image = _fit_image(current, size)
    real_future = _fit_image(future, size)
    input_path = case_dir / "input_t0.png"
    future_path = case_dir / f"real_future_t{future_index}.png"
    input_image.save(input_path)
    real_future.save(future_path)

    prompt = (
        "A realistic robot simulation video from a fixed LIBERO benchmark camera. "
        f"The task is: {instruction} "
        "Continue from the exact starting frame. The robot arm should begin the manipulation, "
        "keep the same tabletop layout, keep the same objects and camera view, and avoid changing object identity. "
        f"{prompt_suffix}"
    ).strip()
    output_path = case_dir / "cosmos3_nano_i2v.mp4"

    generator = torch.Generator(device="cuda").manual_seed(seed)
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
        add_resolution_template=False,
        add_duration_template=False,
        enable_safety_check=False,
        generator=generator,
    )
    elapsed_s = time.time() - start
    export_to_video(result.video, str(output_path), fps=fps, macro_block_size=1)

    generated_frames = _sample_video_frames(output_path, count=4)
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
        "instruction": instruction,
        "seed": seed,
        "num_frames": num_frames,
        "num_steps": num_steps,
        "fps": fps,
        "height": size,
        "width": size,
        "future_reference_t": future_index,
        "guidance_scale": guidance_scale,
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
    parser.add_argument("--out", default="/home/ubuntu/robotics/runs/cosmos3-nano-libero-i2v-smoke")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--future-t", type=int, default=16)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--case-limit", type=int, default=2)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

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
    for index, (name, path, instruction) in enumerate(DEFAULT_CASES[: args.case_limit]):
        rows.append(
            run_case(
                pipe=pipe,
                checkpoint=checkpoint,
                out_dir=out_dir,
                name=f"{index:02d}_{name}",
                hdf5_path=Path(path),
                instruction=instruction,
                prompt_suffix=args.prompt_suffix,
                seed=args.seed + index,
                num_frames=args.num_frames,
                num_steps=args.num_steps,
                fps=args.fps,
                size=args.size,
                future_t=args.future_t,
                demo_index=args.demo_index,
                guidance_scale=args.guidance_scale,
            )
        )
        print(json.dumps(rows[-1], indent=2))

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({"runs": rows}, indent=2) + "\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
