"""
Gendered Collocation Visualization
====================================
Reads CSV files from analysis.py and produces publication-quality figures.

Outputs:
  - gender_collocation_main.png       (main 4-panel: adjectives + verbs)
  - adjective_amod_vs_acomp.png       (attributive vs predicative comparison)
  - verb_tier_breakdown.png           (pronoun vs role noun vs character name)
  - patient_and_possessives.png       (object verbs + possessives)
  - normalized_comparison.png         (per-1000 normalized bars)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

# ============================================================
# 1. LOAD DATA
# ============================================================

required_files = [
    "gendered_adjectives.csv", "gendered_subject_verbs.csv",
    "gendered_object_verbs.csv", "normalization_counts.csv",
]
for f in required_files:
    if not os.path.exists(f):
        print(f"ERROR: {f} not found! Run analysis.py first.")
        sys.exit(1)

df_adj = pd.read_csv("gendered_adjectives.csv")
df_verbs = pd.read_csv("gendered_subject_verbs.csv")
df_objects = pd.read_csv("gendered_object_verbs.csv")
df_poss = pd.read_csv("gendered_possessives.csv")
norm = pd.read_csv("normalization_counts.csv")

f_total = norm[norm["category"] == "female_total"]["token_count"].values[0]
m_total = norm[norm["category"] == "male_total"]["token_count"].values[0]

print(f"Data loaded: {len(df_adj)} adjectives, {len(df_verbs)} subject-verbs,")
print(f"  {len(df_objects)} objects, {len(df_poss)} possessives")
print(f"  Female tokens: {f_total}, Male tokens: {m_total}\n")

C_FEMALE = "#d6604d"
C_MALE = "#4393c3"
sns.set_theme(style="whitegrid")

def norm_per_1000(count, gender):
    total = f_total if gender == "female" else m_total
    return round((count / total) * 1000, 2) if total > 0 else 0


# ============================================================
# 2. FIGURE 1 — MAIN 4-PANEL
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Pride and Prejudice: Expanded Gendered Collocation Analysis',
             fontsize=18, fontweight='bold')

f_adj_top = df_adj[df_adj["gender"] == "female"]["adjective"].value_counts().head(12)
sns.barplot(x=f_adj_top.values, y=f_adj_top.index, ax=axes[0, 0], color=C_FEMALE)
axes[0, 0].set_title('Adjectives Describing Female Characters\n(amod + acomp combined)', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Frequency')

m_adj_top = df_adj[df_adj["gender"] == "male"]["adjective"].value_counts().head(12)
sns.barplot(x=m_adj_top.values, y=m_adj_top.index, ax=axes[0, 1], color=C_MALE)
axes[0, 1].set_title('Adjectives Describing Male Characters\n(amod + acomp combined)', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Frequency')

f_verb_top = df_verbs[df_verbs["gender"] == "female"]["verb"].value_counts().head(12)
sns.barplot(x=f_verb_top.values, y=f_verb_top.index, ax=axes[1, 0], palette="flare",
            hue=f_verb_top.index, legend=False)
axes[1, 0].set_title('Top Verbs with Female Subjects\n(nsubj, excluding be)', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Frequency')

m_verb_top = df_verbs[df_verbs["gender"] == "male"]["verb"].value_counts().head(12)
sns.barplot(x=m_verb_top.values, y=m_verb_top.index, ax=axes[1, 1], palette="crest",
            hue=m_verb_top.index, legend=False)
axes[1, 1].set_title('Top Verbs with Male Subjects\n(nsubj, excluding be)', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Frequency')

plt.tight_layout()
plt.savefig("gender_collocation_main.png", dpi=300)
print("Figure 1 saved: gender_collocation_main.png")
plt.close()


# ============================================================
# 3. FIGURE 2 — ATTRIBUTIVE vs PREDICATIVE ADJECTIVES
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Attributive (amod) vs Predicative (acomp) Adjective Comparison',
             fontsize=16, fontweight='bold')

for col, (gender, g_name, g_color) in enumerate([("female", "Female", C_FEMALE), ("male", "Male", C_MALE)]):
    amod_top = df_adj[(df_adj["gender"] == gender) & (df_adj["dep_type"] == "amod")]["adjective"].value_counts().head(10)
    sns.barplot(x=amod_top.values, y=amod_top.index, ax=axes[0, col], color=g_color, alpha=0.7)
    axes[0, col].set_title(f'{g_name} — Attributive Adjectives (amod)\n"young lady", "old gentleman"',
                           fontsize=12, fontweight='bold')
    axes[0, col].set_xlabel('Frequency')

    acomp_top = df_adj[(df_adj["gender"] == gender) & (df_adj["dep_type"].isin(["acomp", "attr"]))]["adjective"].value_counts().head(10)
    if len(acomp_top) > 0:
        sns.barplot(x=acomp_top.values, y=acomp_top.index, ax=axes[1, col], color=g_color, alpha=1.0)
    axes[1, col].set_title(f'{g_name} — Predicative Adjectives (acomp)\n"she was proud", "he seemed amiable"',
                           fontsize=12, fontweight='bold')
    axes[1, col].set_xlabel('Frequency')

plt.tight_layout()
plt.savefig("adjective_amod_vs_acomp.png", dpi=300)
print("Figure 2 saved: adjective_amod_vs_acomp.png")
plt.close()


# ============================================================
# 4. FIGURE 3 — VERB TIER BREAKDOWN
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Subject-Verb Relations by Tier (Pronoun / Role Noun / Character Name)',
             fontsize=16, fontweight='bold')

tiers = [("pronoun", "Pronouns\n(she/he)"), ("role_noun", "Role Nouns\n(lady/gentleman)"), ("name", "Character Names\n(Elizabeth/Darcy)")]

for col, (tier_key, tier_label) in enumerate(tiers):
    for row, (gender, g_name, palette) in enumerate([("female", "Female", "Reds_r"), ("male", "Male", "Blues_r")]):
        subset = df_verbs[(df_verbs["gender"] == gender) & (df_verbs["tier"] == tier_key)]
        top = subset["verb"].value_counts().head(8)
        if len(top) > 0:
            sns.barplot(x=top.values, y=top.index, ax=axes[row, col], palette=palette,
                        hue=top.index, legend=False)
        axes[row, col].set_title(f'{g_name} — {tier_label}', fontsize=11, fontweight='bold')
        axes[row, col].set_xlabel('Frequency')

plt.tight_layout()
plt.savefig("verb_tier_breakdown.png", dpi=300)
print("Figure 3 saved: verb_tier_breakdown.png")
plt.close()


# ============================================================
# 5. FIGURE 4 — PATIENT VERBS AND POSSESSIVES
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Object (Patient) Verbs and Possessive Relations',
             fontsize=16, fontweight='bold')

f_obj_top = df_objects[df_objects["gender"] == "female"]["verb"].value_counts().head(10)
if len(f_obj_top) > 0:
    sns.barplot(x=f_obj_top.values, y=f_obj_top.index, ax=axes[0, 0], color=C_FEMALE)
axes[0, 0].set_title('Female — Patient Verbs\n(Whose actions is the character subject to?)',
                      fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Frequency')

m_obj_top = df_objects[df_objects["gender"] == "male"]["verb"].value_counts().head(10)
if len(m_obj_top) > 0:
    sns.barplot(x=m_obj_top.values, y=m_obj_top.index, ax=axes[0, 1], color=C_MALE)
axes[0, 1].set_title('Male — Patient Verbs\n(Whose actions is the character subject to?)',
                      fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Frequency')

f_poss_top = df_poss[df_poss["gender"] == "female"]["possessed"].value_counts().head(10)
if len(f_poss_top) > 0:
    sns.barplot(x=f_poss_top.values, y=f_poss_top.index, ax=axes[1, 0], palette="flare",
                hue=f_poss_top.index, legend=False)
axes[1, 0].set_title('Female — Possessives\n("her ___" — what do they possess?)',
                      fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Frequency')

m_poss_top = df_poss[df_poss["gender"] == "male"]["possessed"].value_counts().head(10)
if len(m_poss_top) > 0:
    sns.barplot(x=m_poss_top.values, y=m_poss_top.index, ax=axes[1, 1], palette="crest",
                hue=m_poss_top.index, legend=False)
axes[1, 1].set_title('Male — Possessives\n("his ___" — what do they possess?)',
                      fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Frequency')

plt.tight_layout()
plt.savefig("patient_and_possessives.png", dpi=300)
print("Figure 4 saved: patient_and_possessives.png")
plt.close()


# ============================================================
# 6. FIGURE 5 — NORMALIZED COMPARISON
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Normalized Comparison (per 1000 gendered tokens)',
             fontsize=16, fontweight='bold')

top_adjs = df_adj["adjective"].value_counts().head(12).index.tolist()
f_norms = [norm_per_1000(len(df_adj[(df_adj["adjective"] == a) & (df_adj["gender"] == "female")]), "female") for a in top_adjs]
m_norms = [norm_per_1000(len(df_adj[(df_adj["adjective"] == a) & (df_adj["gender"] == "male")]), "male") for a in top_adjs]

y_pos = np.arange(len(top_adjs))
ax1.barh(y_pos - 0.2, f_norms, height=0.35, color=C_FEMALE, label="Female", alpha=0.85)
ax1.barh(y_pos + 0.2, m_norms, height=0.35, color=C_MALE, label="Male", alpha=0.85)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top_adjs)
ax1.set_xlabel('Frequency (per 1000 tokens)')
ax1.set_title('Top 12 Adjectives\n(Normalized)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.invert_yaxis()

top_verbs = df_verbs["verb"].value_counts().head(12).index.tolist()
f_v_norms = [norm_per_1000(len(df_verbs[(df_verbs["verb"] == v) & (df_verbs["gender"] == "female")]), "female") for v in top_verbs]
m_v_norms = [norm_per_1000(len(df_verbs[(df_verbs["verb"] == v) & (df_verbs["gender"] == "male")]), "male") for v in top_verbs]

y_pos2 = np.arange(len(top_verbs))
ax2.barh(y_pos2 - 0.2, f_v_norms, height=0.35, color=C_FEMALE, label="Female", alpha=0.85)
ax2.barh(y_pos2 + 0.2, m_v_norms, height=0.35, color=C_MALE, label="Male", alpha=0.85)
ax2.set_yticks(y_pos2)
ax2.set_yticklabels(top_verbs)
ax2.set_xlabel('Frequency (per 1000 tokens)')
ax2.set_title('Top 12 Subject Verbs\n(Normalized)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig("normalized_comparison.png", dpi=300)
print("Figure 5 saved: normalized_comparison.png")
plt.close()

print("\nAll figures saved successfully!")
