#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path("run-artifacts/libero-language-sweep-20260529-1120")
SUMMARY = ROOT / "summary_success.tsv"

PROMPT_GROUPS = {
    "valid": ["P0", "P1", "P2", "P3", "C4"],
    "bad": ["C1", "C2", "C3", "O1", "O2"],
    "paraphrase": ["P0", "P1", "P2", "P3"],
    "counterfactual": ["C1", "C2", "C3"],
    "negative": ["O1", "O2"],
}

PROMPT_LABELS = {
    "P0": "P0 canonical",
    "P1": "P1 paraphrase",
    "P2": "P2 expanded",
    "P3": "P3 stepwise",
    "C1": "C1 ramekin",
    "C2": "C2 next-to",
    "C3": "C3 plate",
    "C4": "C4 bowl+avoid",
    "O1": "O1 approach only",
    "O2": "O2 do nothing",
}

MODEL_LABELS = {
    "pi05_libero": "Pi 0.5",
    "molmoact2_libero": "MolmoAct2",
    "cosmos_policy": "Cosmos",
    "ge_act": "GE-Act",
    "fastwam": "FastWAM",
}

COLORS = {
    "pi05_libero": "#2563eb",
    "molmoact2_libero": "#16a34a",
    "cosmos_policy": "#dc2626",
    "ge_act": "#9333ea",
    "fastwam": "#f59e0b",
}


def read_rows():
    with SUMMARY.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def pct(v):
    return f"{v * 100:.0f}%"


def svg_bar_chart(path: Path, title: str, labels: list[str], series: dict[str, list[float]], note: str = ""):
    width = 1180
    height = 620
    margin_l, margin_r, margin_t, margin_b = 90, 30, 78, 150
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b
    n = len(labels)
    models = list(series)
    group_w = chart_w / max(n, 1)
    bar_w = min(24, group_w / (len(models) + 1.4))

    def x_for(i, j):
        start = margin_l + i * group_w + (group_w - bar_w * len(models)) / 2
        return start + j * bar_w

    def y_for(v):
        return margin_t + chart_h * (1 - v)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_l}" y="34" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">{title}</text>',
    ]
    if note:
        lines.append(f'<text x="{margin_l}" y="60" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">{note}</text>')

    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = y_for(tick)
        lines.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width-margin_r}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{margin_l-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">{pct(tick)}</text>')

    for i, label in enumerate(labels):
        x_mid = margin_l + i * group_w + group_w / 2
        lines.append(f'<text x="{x_mid:.1f}" y="{height-112}" text-anchor="end" transform="rotate(-45 {x_mid:.1f},{height-112})" font-family="Arial, sans-serif" font-size="12" fill="#374151">{label}</text>')

    for j, model in enumerate(models):
        color = COLORS.get(model, "#64748b")
        for i, value in enumerate(series[model]):
            x = x_for(i, j)
            y = y_for(value)
            h = margin_t + chart_h - y
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-2:.1f}" height="{h:.1f}" fill="{color}" rx="2"/>')
            if value < 0.98:
                lines.append(f'<text x="{x+(bar_w-2)/2:.1f}" y="{y-5:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#111827">{pct(value)}</text>')

    legend_x = margin_l
    legend_y = height - 40
    for model in models:
        label = MODEL_LABELS.get(model, model)
        color = COLORS.get(model, "#64748b")
        lines.append(f'<rect x="{legend_x}" y="{legend_y-11}" width="12" height="12" fill="{color}" rx="2"/>')
        lines.append(f'<text x="{legend_x+18}" y="{legend_y}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{label}</text>')
        legend_x += 120

    lines.append("</svg>")
    path.write_text("\n".join(lines))


def svg_score_chart(path: Path, rows_by_model: dict[str, dict[str, float]]):
    models = list(rows_by_model)
    labels = [MODEL_LABELS.get(m, m) for m in models]
    valid = [mean([rows_by_model[m][p] for p in PROMPT_GROUPS["valid"]]) for m in models]
    bad_persist = [mean([rows_by_model[m][p] for p in PROMPT_GROUPS["bad"]]) for m in models]
    sensitivity = [1 - v for v in bad_persist]

    width, height = 960, 520
    margin_l, margin_r, margin_t, margin_b = 92, 40, 72, 92
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b
    group_w = chart_w / len(models)
    bar_w = 34
    metrics = [
        ("valid task success", valid, "#2563eb"),
        ("bad-prompt persistence", bad_persist, "#dc2626"),
        ("prompt sensitivity", sensitivity, "#16a34a"),
    ]

    def y_for(v):
        return margin_t + chart_h * (1 - v)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_l}" y="34" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">Prompt Sensitivity Summary</text>',
        f'<text x="{margin_l}" y="57" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Sensitivity = 1 - canonical LIBERO success on bad prompts. Higher means the prompt perturbed canonical behavior more.</text>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = y_for(tick)
        lines.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width-margin_r}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{margin_l-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">{pct(tick)}</text>')

    for i, model in enumerate(models):
        base_x = margin_l + i * group_w + group_w / 2 - bar_w * 1.5
        for j, (_, values, color) in enumerate(metrics):
            v = values[i]
            x = base_x + j * bar_w
            y = y_for(v)
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-4}" height="{margin_t+chart_h-y:.1f}" fill="{color}" rx="2"/>')
            lines.append(f'<text x="{x+(bar_w-4)/2:.1f}" y="{y-5:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#111827">{pct(v)}</text>')
        x_mid = margin_l + i * group_w + group_w / 2
        lines.append(f'<text x="{x_mid:.1f}" y="{height-58}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#111827">{labels[i]}</text>')

    legend_x = margin_l
    for name, _, color in metrics:
        lines.append(f'<rect x="{legend_x}" y="{height-28}" width="12" height="12" fill="{color}" rx="2"/>')
        lines.append(f'<text x="{legend_x+18}" y="{height-17}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{name}</text>')
        legend_x += 210

    lines.append("</svg>")
    path.write_text("\n".join(lines))


def svg_scatter_chart(path: Path, rows_by_model: dict[str, dict[str, float]]):
    prompt_order = ["P0", "P1", "P2", "P3", "C4", "C1", "C2", "C3", "O1", "O2"]
    bands = [
        ("valid compatible", 0, 5, "#eff6ff"),
        ("near-miss / changed goal", 5, 8, "#fff7ed"),
        ("negative / inhibitory", 8, 10, "#fef2f2"),
    ]
    models = list(rows_by_model)
    width, height = 1180, 620
    margin_l, margin_r, margin_t, margin_b = 90, 40, 92, 140
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b
    step = chart_w / (len(prompt_order) - 1)
    jitter = {
        "pi05_libero": -14,
        "molmoact2_libero": -7,
        "cosmos_policy": 0,
        "ge_act": 7,
        "fastwam": 14,
    }

    def x_for(i, model=None):
        return margin_l + i * step + (jitter.get(model, 0) if model else 0)

    def y_for(v):
        return margin_t + chart_h * (1 - v)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_l}" y="34" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">Prompt Sensitivity Scatter</text>',
        f'<text x="{margin_l}" y="58" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Y-axis is raw canonical LIBERO success. On bad prompts, high values mean canonical-task persistence, not necessarily adherence.</text>',
    ]

    for label, start, end, color in bands:
        x1 = x_for(start) - step / 2
        x2 = x_for(end - 1) + step / 2
        x1 = max(margin_l - step / 2, x1)
        x2 = min(width - margin_r + step / 2, x2)
        lines.append(f'<rect x="{x1:.1f}" y="{margin_t-10}" width="{x2-x1:.1f}" height="{chart_h+20}" fill="{color}"/>')
        lines.append(f'<text x="{(x1+x2)/2:.1f}" y="{margin_t-20}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#374151">{label}</text>')

    for tick in [0, 1 / 3, 2 / 3, 1]:
        y = y_for(tick)
        lines.append(f'<line x1="{margin_l-step/2:.1f}" y1="{y:.1f}" x2="{width-margin_r+step/2:.1f}" y2="{y:.1f}" stroke="#d1d5db" stroke-width="1"/>')
        lines.append(f'<text x="{margin_l-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">{pct(tick)}</text>')

    for i, prompt_id in enumerate(prompt_order):
        x = x_for(i)
        lines.append(f'<line x1="{x:.1f}" y1="{margin_t-10}" x2="{x:.1f}" y2="{margin_t+chart_h+10}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{x:.1f}" y="{height-108}" text-anchor="end" transform="rotate(-45 {x:.1f},{height-108})" font-family="Arial, sans-serif" font-size="12" fill="#111827">{PROMPT_LABELS[prompt_id]}</text>')

    for model in models:
        color = COLORS.get(model, "#64748b")
        points = []
        for i, prompt_id in enumerate(prompt_order):
            x = x_for(i, model)
            y = y_for(rows_by_model[model][prompt_id])
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.45"/>')
        for i, prompt_id in enumerate(prompt_order):
            v = rows_by_model[model][prompt_id]
            x = x_for(i, model)
            y = y_for(v)
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
            if v < 0.98:
                lines.append(f'<text x="{x:.1f}" y="{y-10:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#111827">{pct(v)}</text>')

    legend_x = margin_l
    legend_y = height - 38
    for model in models:
        label = MODEL_LABELS.get(model, model)
        color = COLORS.get(model, "#64748b")
        lines.append(f'<circle cx="{legend_x+6}" cy="{legend_y-5}" r="5.5" fill="{color}"/>')
        lines.append(f'<text x="{legend_x+18}" y="{legend_y}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{label}</text>')
        legend_x += 122

    lines.append("</svg>")
    path.write_text("\n".join(lines))


def write_review(path: Path, rows_by_model: dict[str, dict[str, float]]):
    prompt_order = ["P0", "P1", "P2", "P3", "C1", "C2", "C3", "C4", "O1", "O2"]
    lines = [
        "# LIBERO Language Sweep Review",
        "",
        "## Interpretation",
        "",
        "LIBERO still evaluates the canonical task: pick up the black bowl and place it on the plate. That makes this a language-sensitivity probe rather than a plain success benchmark.",
        "",
        "- Valid prompts: P0-P3 and C4 are compatible with the canonical task. Success is normal task competence.",
        "- Bad prompts: C1-C3 and O1-O2 contradict or avoid the canonical task. Success is suspicious because it usually means the model kept doing the canonical task despite the prompt.",
        "- Bad-prompt failure is the interesting manual-review bucket: it may mean the model followed the changed instruction, or it may mean the prompt destabilized the policy.",
        "",
        "## Aggregate Scores",
        "",
        "| Model | Valid prompt success | Bad-prompt canonical persistence | Prompt sensitivity |",
        "|---|---:|---:|---:|",
    ]
    for model, vals in rows_by_model.items():
        valid = mean([vals[p] for p in PROMPT_GROUPS["valid"]])
        bad = mean([vals[p] for p in PROMPT_GROUPS["bad"]])
        lines.append(f"| {MODEL_LABELS.get(model, model)} | {pct(valid)} | {pct(bad)} | {pct(1-bad)} |")

    lines += [
        "",
        "## Prompt-Level Results",
        "",
        "| Model | " + " | ".join(prompt_order) + " |",
        "|---|" + "|".join(["---:"] * len(prompt_order)) + "|",
    ]
    for model, vals in rows_by_model.items():
        lines.append(
            f"| {MODEL_LABELS.get(model, model)} | "
            + " | ".join(pct(vals[p]) for p in prompt_order)
            + " |"
        )

    lines += [
        "",
        "## Main Takeaways",
        "",
        "- All models are strong on canonical-compatible wording. The valid prompt average is 100% for every model in this 3-trial sweep.",
        "- Cosmos is the least sensitive by this probe: it succeeds on every bad prompt, including explicit do-nothing prompts. That is likely task-prior persistence, not good language adherence.",
        "- FastWAM is also mostly insensitive, with one dip on C2. It still succeeds on O1 and O2, so the policy is not strongly respecting negative instructions in this setup.",
        "- Pi 0.5 and MolmoAct2 show modest sensitivity only on O2, with MolmoAct2 also dipping hard on C3.",
        "- GE-Act is the most sensitive on explicit do-nothing O2, dropping to 1/3 while staying perfect on other bad prompts.",
        "",
        "## Manual Review Priority",
        "",
        "Review the bad-prompt failures first: MolmoAct2 C3, Pi 0.5 O2, MolmoAct2 O2, GE-Act O2, FastWAM C2. These are the only places where the prompt changed canonical success.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    rows = read_rows()
    rows_by_model: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        rows_by_model[row["model"]][row["prompt_id"]] = float(row["success_rate"])

    prompt_order = ["P0", "P1", "P2", "P3", "C1", "C2", "C3", "C4", "O1", "O2"]
    svg_bar_chart(
        ROOT / "plot_prompt_success.svg",
        "Raw LIBERO Success by Prompt",
        [PROMPT_LABELS[p] for p in prompt_order],
        {model: [vals[p] for p in prompt_order] for model, vals in rows_by_model.items()},
        "Raw score is canonical task success, not direct instruction-following.",
    )
    svg_bar_chart(
        ROOT / "plot_bad_prompt_persistence.svg",
        "Canonical-Task Persistence on Bad Prompts",
        [PROMPT_LABELS[p] for p in PROMPT_GROUPS["bad"]],
        {model: [vals[p] for p in PROMPT_GROUPS["bad"]] for model, vals in rows_by_model.items()},
        "Higher means more suspicious: the model still completed the canonical task under contradictory or negative prompts.",
    )
    svg_score_chart(ROOT / "plot_sensitivity_summary.svg", rows_by_model)
    svg_scatter_chart(ROOT / "plot_prompt_sensitivity_scatter.svg", rows_by_model)
    write_review(ROOT / "language_sensitivity_review.md", rows_by_model)

    print(ROOT / "plot_prompt_success.svg")
    print(ROOT / "plot_bad_prompt_persistence.svg")
    print(ROOT / "plot_sensitivity_summary.svg")
    print(ROOT / "plot_prompt_sensitivity_scatter.svg")
    print(ROOT / "language_sensitivity_review.md")


if __name__ == "__main__":
    main()
