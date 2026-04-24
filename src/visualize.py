"""
Gendered Collocation Visualization
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Generate collocation figures from processed CSVs.")
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output-dir", default="figures")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_adj = pd.read_csv(input_dir / "gendered_adjectives.csv")
    df_verbs = pd.read_csv(input_dir / "gendered_subject_verbs.csv")
    df_objects = pd.read_csv(input_dir / "gendered_object_verbs.csv")
    df_poss = pd.read_csv(input_dir / "gendered_possessives.csv")
    norm = pd.read_csv(input_dir / "normalization_counts.csv")

    f_total = norm[norm["category"] == "female_total"]["token_count"].values[0]
    m_total = norm[norm["category"] == "male_total"]["token_count"].values[0]

    c_female = "#d6604d"
    c_male = "#4393c3"
    sns.set_theme(style="whitegrid")

    def norm_per_1000(count, gender):
        total = f_total if gender == "female" else m_total
        return round((count / total) * 1000, 2) if total > 0 else 0

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Pride and Prejudice: Expanded Gendered Collocation Analysis", fontsize=18, fontweight="bold")

    f_adj_top = df_adj[df_adj["gender"] == "female"]["adjective"].value_counts().head(12)
    sns.barplot(x=f_adj_top.values, y=f_adj_top.index, ax=axes[0, 0], color=c_female)

    m_adj_top = df_adj[df_adj["gender"] == "male"]["adjective"].value_counts().head(12)
    sns.barplot(x=m_adj_top.values, y=m_adj_top.index, ax=axes[0, 1], color=c_male)

    f_verb_top = df_verbs[df_verbs["gender"] == "female"]["verb"].value_counts().head(12)
    sns.barplot(x=f_verb_top.values, y=f_verb_top.index, ax=axes[1, 0], palette="flare", hue=f_verb_top.index, legend=False)

    m_verb_top = df_verbs[df_verbs["gender"] == "male"]["verb"].value_counts().head(12)
    sns.barplot(x=m_verb_top.values, y=m_verb_top.index, ax=axes[1, 1], palette="crest", hue=m_verb_top.index, legend=False)

    plt.tight_layout()
    plt.savefig(output_dir / "gender_collocation_main.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Attributive (amod) vs Predicative (acomp) Adjective Comparison", fontsize=16, fontweight="bold")
    for col, (gender, g_name, g_color) in enumerate([("female", "Female", c_female), ("male", "Male", c_male)]):
        amod_top = df_adj[(df_adj["gender"] == gender) & (df_adj["dep_type"] == "amod")]["adjective"].value_counts().head(10)
        sns.barplot(x=amod_top.values, y=amod_top.index, ax=axes[0, col], color=g_color, alpha=0.7)
        acomp_top = df_adj[(df_adj["gender"] == gender) & (df_adj["dep_type"].isin(["acomp", "attr"]))]["adjective"].value_counts().head(10)
        if len(acomp_top) > 0:
            sns.barplot(x=acomp_top.values, y=acomp_top.index, ax=axes[1, col], color=g_color, alpha=1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "adjective_amod_vs_acomp.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    tiers = [("pronoun", "Pronouns"), ("role_noun", "Role Nouns"), ("name", "Character Names")]
    for col, (tier_key, tier_label) in enumerate(tiers):
        for row, (gender, palette) in enumerate([("female", "Reds_r"), ("male", "Blues_r")]):
            subset = df_verbs[(df_verbs["gender"] == gender) & (df_verbs["tier"] == tier_key)]
            top = subset["verb"].value_counts().head(8)
            if len(top) > 0:
                sns.barplot(x=top.values, y=top.index, ax=axes[row, col], palette=palette, hue=top.index, legend=False)
            axes[row, col].set_title(f"{gender.title()} — {tier_label}")
    plt.tight_layout()
    plt.savefig(output_dir / "verb_tier_breakdown.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    f_obj_top = df_objects[df_objects["gender"] == "female"]["verb"].value_counts().head(10)
    if len(f_obj_top) > 0:
        sns.barplot(x=f_obj_top.values, y=f_obj_top.index, ax=axes[0, 0], color=c_female)
    m_obj_top = df_objects[df_objects["gender"] == "male"]["verb"].value_counts().head(10)
    if len(m_obj_top) > 0:
        sns.barplot(x=m_obj_top.values, y=m_obj_top.index, ax=axes[0, 1], color=c_male)
    f_poss_top = df_poss[df_poss["gender"] == "female"]["possessed"].value_counts().head(10)
    if len(f_poss_top) > 0:
        sns.barplot(x=f_poss_top.values, y=f_poss_top.index, ax=axes[1, 0], palette="flare", hue=f_poss_top.index, legend=False)
    m_poss_top = df_poss[df_poss["gender"] == "male"]["possessed"].value_counts().head(10)
    if len(m_poss_top) > 0:
        sns.barplot(x=m_poss_top.values, y=m_poss_top.index, ax=axes[1, 1], palette="crest", hue=m_poss_top.index, legend=False)
    plt.tight_layout()
    plt.savefig(output_dir / "patient_and_possessives.png", dpi=300)
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    top_adjs = df_adj["adjective"].value_counts().head(12).index.tolist()
    f_norms = [norm_per_1000(len(df_adj[(df_adj["adjective"] == a) & (df_adj["gender"] == "female")]), "female") for a in top_adjs]
    m_norms = [norm_per_1000(len(df_adj[(df_adj["adjective"] == a) & (df_adj["gender"] == "male")]), "male") for a in top_adjs]
    y_pos = np.arange(len(top_adjs))
    ax1.barh(y_pos - 0.2, f_norms, height=0.35, color=c_female)
    ax1.barh(y_pos + 0.2, m_norms, height=0.35, color=c_male)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_adjs)

    top_verbs = df_verbs["verb"].value_counts().head(12).index.tolist()
    f_v_norms = [norm_per_1000(len(df_verbs[(df_verbs["verb"] == v) & (df_verbs["gender"] == "female")]), "female") for v in top_verbs]
    m_v_norms = [norm_per_1000(len(df_verbs[(df_verbs["verb"] == v) & (df_verbs["gender"] == "male")]), "male") for v in top_verbs]
    y_pos2 = np.arange(len(top_verbs))
    ax2.barh(y_pos2 - 0.2, f_v_norms, height=0.35, color=c_female)
    ax2.barh(y_pos2 + 0.2, m_v_norms, height=0.35, color=c_male)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(top_verbs)

    plt.tight_layout()
    plt.savefig(output_dir / "normalized_comparison.png", dpi=300)
    plt.close()

    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
