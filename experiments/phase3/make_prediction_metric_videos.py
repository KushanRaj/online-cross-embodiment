from __future__ import annotations

import argparse
import csv
import io
import re
from collections import defaultdict
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

"""Render the canonical Phase 3 Cosmos diagnostic videos.

Layout is intentionally fixed:
- left: dense policy execution video from primary_images
- middle: Cosmos predicted future for the active query chunk
- right: corrected IDM(C,P)-a metric traces with red elapsed trace and green
  active chunk/dot

Do not drive playback from query_primary_images; those are sparse query frames
and make the execution panel look frame-by-frame instead of like the rollout.
"""


def _read_images(h5: h5py.File, name: str) -> np.ndarray | None:
    if name in h5:
        return np.asarray(h5[name][()])
    jpeg_name = f"{name}_jpeg"
    if jpeg_name not in h5:
        return None
    frames = []
    for encoded in h5[jpeg_name]:
        frames.append(np.asarray(Image.open(io.BytesIO(bytes(encoded))).convert("RGB"), dtype=np.uint8))
    return np.stack(frames, axis=0) if frames else None


def _load_rows(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    by_episode: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            by_episode[row["episode"]].append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda r: int(float(r["query_t"])))
    return dict(by_episode)


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def _find_rows_for_h5(h5_path: Path, rows_by_episode: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    stem = h5_path.stem
    if stem in rows_by_episode:
        return rows_by_episode[stem]
    for key, rows in rows_by_episode.items():
        if key in stem or stem in key:
            return rows
    return []


def _episode_num(path: Path) -> int:
    match = re.search(r"--ep=(\d+)--", path.name)
    return int(match.group(1)) if match else 0


def _task_id(path: Path) -> int:
    match = re.search(r"--task=(\d+)--", path.name)
    return int(match.group(1)) if match else 0


def _plot_frame(rows: list[dict[str, str]], active: int, width: int, height: int) -> Image.Image:
    metrics = [
        ("idm_model_vs_selected_step_l2", "|| IDM(C,P) - a ||"),
        ("idm_model_vs_selected_eef_step_cos_dist", "Mean per-step EEF cosine distance: IDM(C,P) vs a"),
        ("idm_model_vs_selected_gripper_mismatch_rate", "Gripper mismatch rate"),
    ]
    xs = np.asarray([_float(r, "query_t") for r in rows], dtype=np.float32)
    fig, axes = plt.subplots(3, 1, figsize=(width / 120, height / 120), sharex=True)
    for ax, (key, title) in zip(axes, metrics):
        ys = np.asarray([_float(r, key) for r in rows], dtype=np.float32)
        ax.plot(xs, ys, color="#d9d9d9", linewidth=1.4, marker="o", markersize=3, alpha=0.65)
        ax.axvspan(xs[active], xs[active] + 16, color="#2ca25f", alpha=0.10, zorder=0)
        if active > 0:
            ax.plot(xs[: active + 1], ys[: active + 1], color="#de2d26", linewidth=2.4, marker="o", markersize=4)
        ax.scatter([xs[active]], [ys[active]], color="#31a354", edgecolor="#0f3b1d", s=70, zorder=6)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_ylabel("L2" if key.endswith("step_l2") else ("1-cos" if "cos" in key else "rate"), fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("query timestep", fontsize=8)
    axes[-1].text(0.5, -0.36, "green dot / band = current 16-action chunk", ha="center", transform=axes[-1].transAxes, fontsize=9)
    fig.tight_layout(pad=0.9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


def _caption(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], fill: str = "#111827") -> None:
    draw.text(xy, text, fill=fill)


def _make_video(h5_path: Path, rows: list[dict[str, str]], out_path: Path, fps: int, repeats: int) -> None:
    with h5py.File(h5_path, "r") as h5:
        primary = _read_images(h5, "primary_images")
        query_future = _read_images(h5, "query_future_primary_images")
        if primary is None or query_future is None:
            return
        if "query_t" in h5:
            query_ts = np.asarray(h5["query_t"][()], dtype=np.int32)
        else:
            query_ts = np.asarray([int(float(r["query_t"])) for r in rows], dtype=np.int32)
        task = str(h5.attrs.get("task_description", rows[0].get("task_description", "")))
        success = bool(h5.attrs.get("success", rows[0].get("success") == "True"))

    n = min(len(rows), len(query_future))
    if n == 0:
        return
    query_ts = query_ts[:n]
    rows = rows[:n]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    status = "SUCCESS" if success else "FAILURE"
    status_color = "#15803d" if success else "#b91c1c"
    task_label = f"{status} | task {_task_id(h5_path)} ep {_episode_num(h5_path)} | {task}"

    image_w, image_h = 560, 420
    plot_w, plot_h = 800, 420
    header_h = 124
    canvas_w = image_w * 2 + plot_w
    canvas_h = header_h + image_h
    plot_cache: dict[int, Image.Image] = {}

    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for t in range(len(primary)):
            active = int(np.searchsorted(query_ts, t, side="right") - 1)
            active = max(0, min(active, n - 1))
            current = Image.fromarray(primary[t].astype(np.uint8)).convert("RGB").resize((image_w, image_h))
            pred = Image.fromarray(query_future[active].astype(np.uint8)).convert("RGB").resize((image_w, image_h))
            if active not in plot_cache:
                plot_cache[active] = _plot_frame(rows, active, plot_w, plot_h)
            plot = plot_cache[active]
            canvas = Image.new("RGB", (canvas_w, canvas_h), "#f8fafc")
            canvas.paste(current, (0, header_h))
            canvas.paste(pred, (image_w, header_h))
            canvas.paste(plot, (image_w * 2, header_h))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, canvas_w, header_h), fill="#ffffff")
            draw.rectangle((0, 0, 7, header_h), fill=status_color)
            _caption(draw, task_label[:150], (18, 14), status_color)
            _caption(
                draw,
                f"green dot/chunk active now | middle image is Cosmos P for that active chunk | frame t={t}",
                (18, 45),
                "#6b7280",
            )
            draw.rectangle((0, header_h, image_w, header_h + 28), fill="#050505")
            draw.rectangle((image_w, header_h, image_w * 2, header_h + 28), fill="#050505")
            _caption(draw, "current execution frame", (12, header_h + 6), "#f9fafb")
            _caption(draw, f"Cosmos predicted future for chunk q={int(query_ts[active])}", (image_w + 12, header_h + 6), "#f9fafb")
            for _ in range(repeats):
                writer.append_data(np.asarray(canvas))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    rows_by_episode = _load_rows(Path(args.csv))
    out_dir = Path(args.out_dir)
    files = sorted(run_root.rglob("rollout_data/*.hdf5"), key=lambda p: (_task_id(p), _episode_num(p)))
    for h5_path in files:
        rows = _find_rows_for_h5(h5_path, rows_by_episode)
        if not rows:
            continue
        status = "true" if rows[0]["success"] == "True" else "false"
        out_name = f"prediction_metrics_task{_task_id(h5_path)}_ep{_episode_num(h5_path)}_{status}.mp4"
        _make_video(h5_path, rows, out_dir / out_name, args.fps, args.repeats)
        print(out_dir / out_name)


if __name__ == "__main__":
    main()
