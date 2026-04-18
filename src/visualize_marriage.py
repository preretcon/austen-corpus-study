"""
Marriage Semantic Frame Visualization
=======================================
Reads CSV files from marriage_analysis.py and produces figures.

Outputs:
  - marriage_analysis_main.png     (subjects, objects, adjectives)
  - marriage_gender_asymmetry.png  (contingency table + stats)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import sys
import os

try:
    from scipy.stats import fisher_exact
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("pride_prejudice_parsed.csv")

# Reproduce the data (keeps this script self-contained)
FEMALE_WORDS = {
    'she', 'her', 'herself', 'hers',
    'woman', 'lady', 'girl', 'wife', 'mother', 'daughter', 'sister', 'mrs',
    'elizabeth', 'jane', 'lydia', 'kitty', 'mary', 'charlotte',
    'caroline', 'georgiana', 'catherine', 'anne', 'lizzy',
}
MALE_WORDS = {
    'he', 'him', 'himself', 'his',
    'man', 'gentleman', 'boy', 'husband', 'father', 'son', 'brother', 'mr',
    'darcy', 'bingley', 'wickham', 'collins', 'fitzwilliam', 'bennet',
}

def classify_gender(lemma):
    lemma = str(lemma).lower().strip()
    if lemma in FEMALE_WORDS: return "female"
    if lemma in MALE_WORDS: return "male"
    return "other"

C_FEMALE = "#d6604d"
C_MALE = "#4393c3"
C_OTHER = "#999999"
sns.set_theme(style="whitegrid")


# ============================================================
# 2. EXTRACT DATA
# ============================================================

marry_subj = df[(df['Dep'] == 'nsubj') & (df['Head_Lemma'] == 'marry')].copy()
marry_subj = marry_subj[marry_subj['Lemma'] != '_']
marry_obj = df[(df['Dep'].isin(['dobj', 'obj'])) & (df['Head_Lemma'] == 'marry')].copy()
marry_obj = marry_obj[marry_obj['Lemma'] != '_']
marry_pass = df[(df['Dep'] == 'nsubjpass') & (df['Head_Lemma'] == 'marry')].copy()

marriage_adj = df[(df['POS'] == 'ADJ') & (df['Dep'] == 'amod') &
                  (df['Head_Lemma'].isin(['marriage', 'match', 'engagement', 'proposal']))
                  ]['Lemma'].value_counts()

# Gender counts for contingency
subj_f = sum(1 for _, r in marry_subj.iterrows() if classify_gender(r['Lemma']) == 'female')
subj_m = sum(1 for _, r in marry_subj.iterrows() if classify_gender(r['Lemma']) == 'male')
obj_f = sum(1 for _, r in marry_obj.iterrows() if classify_gender(r['Lemma']) == 'female')
obj_m = sum(1 for _, r in marry_obj.iterrows() if classify_gender(r['Lemma']) == 'male')
pass_f = sum(1 for _, r in marry_pass.iterrows() if classify_gender(r['Lemma']) == 'female')
pass_m = sum(1 for _, r in marry_pass.iterrows() if classify_gender(r['Lemma']) == 'male')

patient_f = obj_f + pass_f
patient_m = obj_m + pass_m


# ============================================================
# 3. FIGURE 1 — MAIN 3-PANEL (updated from original)
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Pride and Prejudice: Marriage Semantic Frame Analysis",
             fontsize=17, fontweight='bold')

# Panel 1: Marriage/match adjectives
if len(marriage_adj) > 0:
    colors_adj = []
    positive = {"happy", "good", "eligible", "desirable", "convenient", "positive"}
    negative = {"imprudent", "unsuitable", "unequal", "improper", "tacit"}
    for adj in marriage_adj.index:
        if adj in positive: colors_adj.append("#2ca02c")
        elif adj in negative: colors_adj.append("#d62728")
        else: colors_adj.append("#7f7f7f")
    sns.barplot(x=marriage_adj.values, y=marriage_adj.index, ax=axes[0],
                palette=colors_adj, hue=marriage_adj.index, legend=False)
    pos_patch = mpatches.Patch(color="#2ca02c", label="Positive")
    neg_patch = mpatches.Patch(color="#d62728", label="Negative")
    neu_patch = mpatches.Patch(color="#7f7f7f", label="Neutral")
    axes[0].legend(handles=[pos_patch, neg_patch, neu_patch], fontsize=9)
axes[0].set_title("Adjectives Modifying\n'marriage', 'match', 'engagement', 'proposal'",
                   fontsize=12, fontweight='bold')
axes[0].set_xlabel("Frequency")
axes[0].set_xticks(range(0, max(marriage_adj.values) + 2 if len(marriage_adj) > 0 else 3))

# Panel 2: Subjects of "marry" (agents)
subj_counts = marry_subj['Lemma'].value_counts().head(10)
colors_subj = [C_FEMALE if classify_gender(l) == 'female' else
               C_MALE if classify_gender(l) == 'male' else C_OTHER
               for l in subj_counts.index]
sns.barplot(x=subj_counts.values, y=subj_counts.index, ax=axes[1],
            palette=colors_subj, hue=subj_counts.index, legend=False)
axes[1].set_title("Subjects of 'marry'\n(Agent — Who decides to marry?)",
                   fontsize=12, fontweight='bold')
axes[1].set_xlabel("Frequency")

# Panel 3: Objects of "marry" (patients)
obj_counts = marry_obj['Lemma'].value_counts().head(10)
colors_obj = [C_FEMALE if classify_gender(l) == 'female' else
              C_MALE if classify_gender(l) == 'male' else C_OTHER
              for l in obj_counts.index]
sns.barplot(x=obj_counts.values, y=obj_counts.index, ax=axes[2],
            palette=colors_obj, hue=obj_counts.index, legend=False)
axes[2].set_title("Objects of 'marry'\n(Patient — Who is married off?)",
                   fontsize=12, fontweight='bold')
axes[2].set_xlabel("Frequency")

# Add gender legend
f_patch = mpatches.Patch(color=C_FEMALE, label="Female")
m_patch = mpatches.Patch(color=C_MALE, label="Male")
o_patch = mpatches.Patch(color=C_OTHER, label="Other/Ambiguous")
axes[1].legend(handles=[f_patch, m_patch, o_patch], fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig("marriage_analysis_main.png", dpi=300)
print("Figure 1 saved: marriage_analysis_main.png")
plt.close()


# ============================================================
# 4. FIGURE 2 — GENDER ASYMMETRY + STATISTICAL TEST
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Gender Asymmetry in the Marriage Frame: Who Acts vs Who Is Acted Upon",
             fontsize=16, fontweight='bold')

# Panel 1: Contingency table as grouped bar
x = np.arange(2)
width = 0.3
axes[0].bar(x - width/2, [subj_f, subj_m], width, color=[C_FEMALE, C_MALE], alpha=0.85)
axes[0].bar(x + width/2, [patient_f, patient_m], width, color=[C_FEMALE, C_MALE], alpha=0.4)
axes[0].set_xticks(x)
axes[0].set_xticklabels(["Female", "Male"])
axes[0].set_ylabel("Count")
axes[0].set_title("Gender × Role in 'marry'\n(Subject vs Patient)", fontsize=13, fontweight='bold')

agent_patch = mpatches.Patch(color='gray', alpha=0.85, label='Subject (Agent)')
patient_patch = mpatches.Patch(color='gray', alpha=0.4, label='Patient (Object+Pass)')
axes[0].legend(handles=[agent_patch, patient_patch], fontsize=10)

# Add count labels
for i, (s, p) in enumerate([(subj_f, patient_f), (subj_m, patient_m)]):
    axes[0].text(i - width/2, s + 0.3, str(s), ha='center', fontweight='bold', fontsize=11)
    axes[0].text(i + width/2, p + 0.3, str(p), ha='center', fontweight='bold', fontsize=11)

# Panel 2: Marriage agency ratios
f_ratio = subj_f / (subj_f + patient_f) if (subj_f + patient_f) > 0 else 0
m_ratio = subj_m / (subj_m + patient_m) if (subj_m + patient_m) > 0 else 0

bars = axes[1].bar(["Female", "Male"], [f_ratio, m_ratio],
                    color=[C_FEMALE, C_MALE], width=0.5, edgecolor="white", linewidth=1.5)
axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Parity (0.5)")
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("Agency Ratio")
axes[1].set_title("Marriage Agency Ratio\nagent / (agent + patient)", fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)

for bar, ratio in zip(bars, [f_ratio, m_ratio]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{ratio:.3f}", ha='center', fontweight='bold', fontsize=13)

# Panel 3: Statistical test annotation
axes[2].axis('off')
contingency = np.array([[subj_f, patient_f], [subj_m, patient_m]])

text_lines = [
    "Contingency Table",
    "─" * 36,
    f"{'':12s} {'Agent':>8s}  {'Patient':>8s}",
    f"{'Female':12s} {subj_f:>8d}  {patient_f:>8d}",
    f"{'Male':12s} {subj_m:>8d}  {patient_m:>8d}",
    "",
]

if HAS_SCIPY:
    odds, p = fisher_exact(contingency)
    text_lines += [
        "Fisher's Exact Test",
        "─" * 36,
        f"Odds ratio:  {odds:.4f}",
        f"p-value:     {p:.6f}",
        "",
    ]
    if p < 0.05:
        text_lines.append("Result: STATISTICALLY SIGNIFICANT")
        text_lines.append(f"(p < {'0.001' if p < 0.001 else '0.01' if p < 0.01 else '0.05'})")
    else:
        text_lines.append("Result: NOT significant (p >= 0.05)")

    text_lines += [
        "",
        "Interpretation:",
        f"Women are {1/odds:.1f}x more likely to be" if odds < 1 else f"Men are {odds:.1f}x more likely to be",
        "in PATIENT position than AGENT" if odds < 1 else "in AGENT position than PATIENT",
        "of 'marry', compared to men." if odds < 1 else "of 'marry', compared to women.",
    ]
else:
    text_lines.append("[scipy not installed]")

axes[2].text(0.05, 0.95, "\n".join(text_lines),
             transform=axes[2].transAxes,
             fontsize=12, fontfamily='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
axes[2].set_title("Statistical Test Results", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig("marriage_gender_asymmetry.png", dpi=300)
print("Figure 2 saved: marriage_gender_asymmetry.png")
plt.close()

print("\nAll marriage figures saved successfully!")
