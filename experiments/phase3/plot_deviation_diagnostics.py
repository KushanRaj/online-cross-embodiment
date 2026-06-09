from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def _episode_task(row: dict[str, str]) -> str:
    episode = row["episode"]
    if "--task=" in episode:
        return "task " + episode.split("--task=")[1].split("--")[0]
    return row.get("task_description", "")[:24]


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _by_episode(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[row["episode"]].append(row)
    for ep_rows in out.values():
        ep_rows.sort(key=lambda r: int(float(r["query_t"])))
    return dict(out)


def _save_success_failure_boxplots(rows: list[dict[str, str]], out_dir: Path) -> None:
    metrics = [
        ("idm_model_vs_selected_l2", "chunk L2: IDM(C,P) vs a"),
        ("idm_observed_vs_selected_l2", "chunk L2: IDM(C,R) vs a"),
        ("idm_model_observed_l2", "chunk L2: IDM(C,P) vs IDM(C,R)"),
        ("idm_model_vs_selected_step_l2", "mean per-step L2"),
        ("idm_model_vs_selected_eef_step_l2", "EEF mean per-step L2"),
        ("idm_model_vs_selected_step_cos_dist", "mean per-step 1-cos"),
        ("idm_model_vs_selected_eef_step_cos_dist", "EEF mean per-step 1-cos"),
        ("idm_model_vs_selected_gripper_step_l1", "gripper mean per-step L1"),
        ("idm_model_vs_selected_gripper_mismatch_rate", "gripper mismatch rate"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(18, 11))
    for ax, (key, title) in zip(axes.flat, metrics):
        success_vals = [_float(r, key) for r in rows if r["success"] == "True"]
        failure_vals = [_float(r, key) for r in rows if r["success"] == "False"]
        ax.boxplot([success_vals, failure_vals], tick_labels=["success", "failure"], showfliers=True)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "success_failure_corrected_metric_boxplots.png", dpi=180)
    plt.close(fig)


def _save_episode_mean_bar(rows: list[dict[str, str]], out_dir: Path) -> None:
    eps = _by_episode(rows)
    metrics = [
        "idm_model_vs_selected_l2",
        "idm_observed_vs_selected_l2",
        "idm_model_vs_selected_step_l2",
        "idm_model_vs_selected_eef_step_l2",
        "idm_model_vs_selected_step_cos_dist",
        "idm_model_vs_selected_eef_step_cos_dist",
        "idm_model_vs_selected_gripper_step_l1",
        "idm_model_vs_selected_gripper_mismatch_rate",
    ]
    labels = []
    success = []
    means = {key: [] for key in metrics}
    for episode, ep_rows in eps.items():
        labels.append(f"{_episode_task(ep_rows[0])} ep {ep_rows[0]['episode'].split('--ep=')[-1].split('--')[0]}")
        success.append(ep_rows[0]["success"] == "True")
        for key in metrics:
            values = np.asarray([_float(r, key) for r in ep_rows], dtype=np.float64)
            means[key].append(float(np.nanmean(values)))

    x = np.arange(len(labels))
    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(12, len(labels) * 0.75), 18), sharex=True)
    for ax, key in zip(axes, metrics):
        colors = ["#2ca25f" if ok else "#de2d26" for ok in success]
        ax.bar(x, means[key], color=colors)
        ax.set_ylabel(key.replace("idm_model_vs_selected_", "").replace("_", " "))
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "per_episode_corrected_metric_means.png", dpi=180)
    plt.close(fig)


def _save_timeseries(rows: list[dict[str, str]], out_dir: Path) -> None:
    eps = _by_episode(rows)
    metrics = [
        ("idm_model_vs_selected_l2", "IDM(C,P) vs a: chunk L2"),
        ("idm_observed_vs_selected_l2", "IDM(C,R) vs a: chunk L2"),
        ("idm_model_vs_selected_step_l2", "IDM(C,P) vs a: mean per-step L2"),
        ("idm_model_vs_selected_eef_step_l2", "IDM(C,P) vs a: EEF mean per-step L2"),
        ("idm_model_vs_selected_step_cos_dist", "IDM(C,P) vs a: mean per-step 1-cos"),
        ("idm_model_vs_selected_eef_step_cos_dist", "IDM(C,P) vs a: EEF mean per-step 1-cos"),
        ("idm_model_vs_selected_gripper_step_l1", "IDM(C,P) vs a: gripper mean per-step L1"),
        ("idm_model_vs_selected_gripper_mismatch_rate", "IDM(C,P) vs a: gripper sign mismatch rate"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 20), sharex=True)
    for ax, (key, title) in zip(axes, metrics):
        for episode, ep_rows in eps.items():
            ok = ep_rows[0]["success"] == "True"
            color = "#74c476" if ok else "#de2d26"
            alpha = 0.28 if ok else 0.95
            linewidth = 1.1 if ok else 2.0
            ax.plot(
                [_float(r, "query_t") for r in ep_rows],
                [_float(r, key) for r in ep_rows],
                marker="o",
                markersize=3,
                linewidth=linewidth,
                color=color,
                alpha=alpha,
            )
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("query timestep")
    fig.tight_layout()
    fig.savefig(out_dir / "corrected_timeseries_success_failure.png", dpi=180)
    plt.close(fig)


def _save_eef_gripper_scatter(rows: list[dict[str, str]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for ok, label, color in [(True, "success", "#2ca25f"), (False, "failure", "#de2d26")]:
        subset = [r for r in rows if (r["success"] == "True") == ok]
        ax.scatter(
            [_float(r, "idm_model_vs_selected_eef_step_cos_dist") for r in subset],
            [_float(r, "idm_model_vs_selected_gripper_mismatch_rate") for r in subset],
            s=22,
            alpha=0.55,
            label=label,
            color=color,
        )
    ax.set_xlabel("EEF mean per-step 1-cos")
    ax.set_ylabel("gripper sign mismatch rate")
    ax.set_title("EEF vs gripper contribution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "eef_vs_gripper_scatter.png", dpi=180)
    plt.close(fig)


def _write_summary(rows: list[dict[str, str]], out_dir: Path) -> None:
    eps = _by_episode(rows)
    summary = {
        "rows": len(rows),
        "episodes": len(eps),
        "success_episodes": sum(1 for ep_rows in eps.values() if ep_rows[0]["success"] == "True"),
        "failure_episodes": sum(1 for ep_rows in eps.values() if ep_rows[0]["success"] == "False"),
    }
    for key in [
        "idm_model_vs_selected_l2",
        "idm_observed_vs_selected_l2",
        "idm_model_observed_l2",
        "idm_model_vs_selected_step_cos_dist",
        "idm_model_vs_selected_eef_step_cos_dist",
        "idm_model_vs_selected_step_l2",
        "idm_model_vs_selected_eef_step_l2",
        "idm_model_vs_selected_gripper_step_l1",
        "idm_model_vs_selected_gripper_mismatch_rate",
    ]:
        values = np.asarray([_float(r, key) for r in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(np.nanmean(values)),
            "median": float(np.nanmedian(values)),
            "p90": float(np.nanpercentile(values, 90)),
        }
    (out_dir / "corrected_diagnostics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(Path(args.csv))
    _save_success_failure_boxplots(rows, out_dir)
    _save_episode_mean_bar(rows, out_dir)
    _save_timeseries(rows, out_dir)
    _save_eef_gripper_scatter(rows, out_dir)
    _write_summary(rows, out_dir)


if __name__ == "__main__":
    main()
