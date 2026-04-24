"""Agency Analysis Visualization"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Generate agency figures.")
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output-dir", default="figures")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_ratios = pd.read_csv(input_dir / "agency_ratios.csv")
    df_profile = pd.read_csv(input_dir / "agency_verb_profile.csv")

    c_female = "#d6604d"
    c_male = "#4393c3"
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    overall = df_ratios[df_ratios["tier"] == "all"]
    labels = ["Female" if g == "female" else "Male" for g in overall["gender"]]
    ratios = overall["agency_ratio"].tolist()
    colors = [c_female if g == "female" else c_male for g in overall["gender"]]
    bars = axes[0].bar(labels, ratios, color=colors)
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.6)
    axes[0].set_ylim(0, 1)
    for bar, ratio in zip(bars, ratios):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{ratio:.4f}", ha="center")

    for i, gender in enumerate(["female", "male"]):
        row = overall[overall["gender"] == gender].iloc[0]
        g_label = "Female" if gender == "female" else "Male"
        color = c_female if gender == "female" else c_male
        axes[1].bar(g_label, row["agent_count"], color=color, alpha=0.85)
        axes[1].bar(g_label, row["patient_count"], bottom=row["agent_count"], color=color, alpha=0.4)

    motifs = ["Agent", "Patient", "Description", "Possession"]
    x = np.arange(len(motifs))
    width = 0.3
    for i, gender in enumerate(["female", "male"]):
        row = overall[overall["gender"] == gender].iloc[0]
        values = [row["agent_count"], row["patient_count"], row["described_count"], row["possessor_count"]]
        color = c_female if gender == "female" else c_male
        label = "Female" if gender == "female" else "Male"
        axes[2].bar(x + (i - 0.5) * width, values, width, color=color, label=label, alpha=0.85)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(motifs)
    axes[2].legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "agency_ratios_main.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    categories_order = ["communication", "cognition", "perception", "motion", "action", "emotion", "other"]
    for idx, gender in enumerate(["female", "male"]):
        subset = df_profile[df_profile["gender"] == gender]
        cats = [c for c in categories_order if c in subset["category"].values]
        agent_vals = [subset[subset["category"] == c]["agent_pct"].values[0] if c in subset["category"].values else 0 for c in cats]
        patient_vals = [subset[subset["category"] == c]["patient_pct"].values[0] if c in subset["category"].values else 0 for c in cats]
        y = np.arange(len(cats))
        axes[idx].barh(y - 0.175, agent_vals, 0.35, alpha=0.85)
        axes[idx].barh(y + 0.175, patient_vals, 0.35, alpha=0.4)
        axes[idx].set_yticks(y)
        axes[idx].set_yticklabels(cats)
    plt.tight_layout()
    plt.savefig(output_dir / "agency_verb_profile.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    tier_data = df_ratios[df_ratios["tier"] != "all"]
    tiers = ["pronoun", "role_noun", "name"]
    x = np.arange(len(tiers))
    width = 0.3
    for i, gender in enumerate(["female", "male"]):
        color = c_female if gender == "female" else c_male
        vals = []
        for tier in tiers:
            row = tier_data[(tier_data["gender"] == gender) & (tier_data["tier"] == tier)]
            vals.append(row["agency_ratio"].values[0] if len(row) > 0 else 0)
        ax.bar(x + (i - 0.5) * width, vals, width, color=color, alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    plt.tight_layout()
    plt.savefig(output_dir / "agency_tier_detail.png", dpi=300)
    plt.close()

    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
