"""Gendered Verb Frame Visualization"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Generate verb-frame figures.")
    parser.add_argument("--input-csv", default="data/processed/verb_frame_results.csv")
    parser.add_argument("--output-dir", default="figures")
    return parser.parse_args()


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    c_f = "#d6604d"
    c_m = "#4393c3"
    sns.set_theme(style="whitegrid")

    verbs = list(df["verb"])
    themes = list(df["theme"])
    f_ratios = list(df["f_ratio"])
    m_ratios = list(df["m_ratio"])
    gaps = list(df["gap"])

    pvals = []
    for p in df["pval"]:
        try:
            pvals.append(float(p))
        except (ValueError, TypeError):
            pvals.append(None)

    cramers_list = []
    for c in df["cramers"]:
        try:
            cramers_list.append(float(c))
        except (ValueError, TypeError):
            cramers_list.append(None)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    x = np.arange(len(verbs))
    w = 0.3
    axes[0].bar(x - w / 2, f_ratios, w, color=c_f, label="Female", alpha=0.85)
    axes[0].bar(x + w / 2, m_ratios, w, color=c_m, label="Male", alpha=0.85)
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"'{v}'" for v in verbs], fontsize=11)

    colors_gap = [c_m if g > 0 else c_f for g in gaps]
    axes[1].barh(verbs, gaps, color=colors_gap, alpha=0.85, height=0.5)
    axes[1].axvline(0, color="gray", linestyle="-", alpha=0.5)
    axes[1].invert_yaxis()

    axes[2].axis("off")
    lines = ["Verb Frame Summary", "=" * 40, ""]
    for i in range(len(verbs)):
        p = pvals[i]
        cv = cramers_list[i]
        sig_str = f"p = {p:.4f}" if p is not None else "n/a"
        lines.append(f"'{verbs[i]}' - {themes[i]}")
        lines.append(f"  Fisher's exact: {sig_str}")
        if cv is not None:
            lines.append(f"  Cramer's V = {cv:.3f}")
        lines.append("")
    axes[2].text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=axes[2].transAxes,
        fontsize=11,
        fontfamily="monospace",
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "verb_frame_comparison.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for idx in range(len(df)):
        row = df.iloc[idx]
        ax = axes[idx // 2, idx % 2]
        positions = np.arange(2)
        w = 0.3
        agent_vals = [int(row["agent_f"]), int(row["agent_m"])]
        patient_vals = [int(row["patient_f"]), int(row["patient_m"])]
        ax.bar(positions - w / 2, agent_vals, w, color=[c_f, c_m], alpha=0.85)
        ax.bar(positions + w / 2, patient_vals, w, color=[c_f, c_m], alpha=0.4)
        ax.set_xticks(positions)
        ax.set_xticklabels(["Female", "Male"], fontsize=11)
        ax.set_title(f"{row['verb']} - {row['theme']}", fontsize=12)

    a_patch = mpatches.Patch(color="gray", alpha=0.85, label="Agent")
    p_patch = mpatches.Patch(color="gray", alpha=0.4, label="Patient")
    fig.legend(handles=[a_patch, p_patch], loc="lower center", ncol=2, fontsize=11, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_dir / "verb_frame_detail.png", dpi=300)
    plt.close()

    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
