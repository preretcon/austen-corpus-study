"""
Agency Analysis Visualization
===============================
Reads CSV files from agency_analysis.py and produces figures.

Outputs:
  - agency_ratios_main.png     (core ratios + Stuhler motifs)
  - agency_verb_profile.png    (semantic category agent/patient breakdown)
  - agency_tier_detail.png     (pronoun/noun/name agency comparison)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import sys
import os

for f in ["agency_ratios.csv", "agency_verb_profile.csv"]:
    if not os.path.exists(f):
        print(f"ERROR: {f} not found! Run agency_analysis.py first.")
        sys.exit(1)

df_ratios = pd.read_csv("agency_ratios.csv")
df_profile = pd.read_csv("agency_verb_profile.csv")
df_agent = pd.read_csv("gendered_subject_verbs.csv")
df_patient = pd.read_csv("gendered_object_verbs.csv")

C_FEMALE = "#d6604d"
C_MALE = "#4393c3"
sns.set_theme(style="whitegrid")


# ============================================================
# FIGURE 1 — CORE AGENCY RATIOS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Pride and Prejudice: Agency Analysis (Stuhler 2024 Framework)',
             fontsize=16, fontweight='bold')

# Panel 1: Agency ratio
overall = df_ratios[df_ratios["tier"] == "all"]
labels = ["Female" if g == "female" else "Male" for g in overall["gender"]]
ratios = overall["agency_ratio"].tolist()
colors = [C_FEMALE if g == "female" else C_MALE for g in overall["gender"]]

bars = axes[0].bar(labels, ratios, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Parity line (0.5)")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Agency Ratio")
axes[0].set_title("Overall Agency Ratio\nagent / (agent + patient)", fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
for bar, ratio in zip(bars, ratios):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{ratio:.4f}", ha='center', va='bottom', fontsize=13, fontweight='bold')

# Panel 2: Agent vs Patient stacked
for i, gender in enumerate(["female", "male"]):
    row = overall[overall["gender"] == gender].iloc[0]
    g_label = "Female" if gender == "female" else "Male"
    color = C_FEMALE if gender == "female" else C_MALE

    axes[1].bar(g_label, row["agent_count"], color=color, alpha=0.85)
    axes[1].bar(g_label, row["patient_count"], bottom=row["agent_count"], color=color, alpha=0.4)
    axes[1].text(i, row["agent_count"]/2, f'Agent\n{int(row["agent_count"])}',
                 ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    axes[1].text(i, row["agent_count"] + row["patient_count"]/2,
                 f'Patient\n{int(row["patient_count"])}',
                 ha='center', va='center', fontsize=10, fontweight='bold')

axes[1].set_ylabel("Total Count")
axes[1].set_title("Agent vs Patient Counts\n(Stacked)", fontsize=13, fontweight='bold')
agent_patch = mpatches.Patch(color='gray', alpha=0.85, label='Agent (subject)')
patient_patch = mpatches.Patch(color='gray', alpha=0.4, label='Patient (object)')
axes[1].legend(handles=[agent_patch, patient_patch], fontsize=10)

# Panel 3: 4-motif distribution
motifs = ["Agent\n(nsubj)", "Patient\n(dobj)", "Description\n(adj)", "Possession\n(poss)"]
x = np.arange(len(motifs))
width = 0.3

for i, gender in enumerate(["female", "male"]):
    row = overall[overall["gender"] == gender].iloc[0]
    values = [row["agent_count"], row["patient_count"], row["described_count"], row["possessor_count"]]
    color = C_FEMALE if gender == "female" else C_MALE
    label = "Female" if gender == "female" else "Male"
    axes[2].bar(x + (i - 0.5) * width, values, width, color=color, label=label, alpha=0.85)

axes[2].set_xticks(x)
axes[2].set_xticklabels(motifs)
axes[2].set_ylabel("Count")
axes[2].set_title("Stuhler (2022) 4-Motif Distribution\n(Per Gender)", fontsize=13, fontweight='bold')
axes[2].legend(fontsize=10)

plt.tight_layout()
plt.savefig("agency_ratios_main.png", dpi=300)
print("Figure 1 saved: agency_ratios_main.png")
plt.close()


# ============================================================
# FIGURE 2 — SEMANTIC VERB CATEGORY PROFILE
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Agent vs Patient Distribution by Semantic Verb Category',
             fontsize=16, fontweight='bold')

categories_order = ["communication", "cognition", "perception", "motion", "action", "emotion", "other"]
cat_labels = {
    "communication": "Communication\n(say, tell, ask)",
    "cognition": "Cognition\n(think, know, believe)",
    "perception": "Perception\n(see, feel, hear)",
    "motion": "Motion\n(come, go, walk)",
    "action": "Action\n(do, make, take)",
    "emotion": "Emotion\n(love, like, hate)",
    "other": "Other",
}

for idx, gender in enumerate(["female", "male"]):
    gender_name = "Female" if gender == "female" else "Male"
    color = C_FEMALE if gender == "female" else C_MALE
    subset = df_profile[df_profile["gender"] == gender]

    cats = [c for c in categories_order if c in subset["category"].values]
    agent_vals = [subset[subset["category"] == c]["agent_pct"].values[0] if c in subset["category"].values else 0 for c in cats]
    patient_vals = [subset[subset["category"] == c]["patient_pct"].values[0] if c in subset["category"].values else 0 for c in cats]
    cat_display = [cat_labels.get(c, c) for c in cats]

    y = np.arange(len(cats))
    height = 0.35

    axes[idx].barh(y - height/2, agent_vals, height, color=color, alpha=0.85, label="Agent %")
    axes[idx].barh(y + height/2, patient_vals, height, color=color, alpha=0.4, label="Patient %")
    axes[idx].set_yticks(y)
    axes[idx].set_yticklabels(cat_display)
    axes[idx].set_xlabel("Percentage (%)")
    axes[idx].set_title(f"{gender_name}", fontsize=14, fontweight='bold')
    axes[idx].legend(fontsize=10)
    axes[idx].invert_yaxis()

plt.tight_layout()
plt.savefig("agency_verb_profile.png", dpi=300)
print("Figure 2 saved: agency_verb_profile.png")
plt.close()


# ============================================================
# FIGURE 3 — TIER-LEVEL AGENCY DETAIL
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Agency Ratios by Tier (Pronoun / Role Noun / Character Name)',
             fontsize=16, fontweight='bold')

tier_data = df_ratios[df_ratios["tier"] != "all"]
tier_labels_map = {"pronoun": "Pronoun\n(she/he)", "role_noun": "Role Noun\n(lady/gentleman)", "name": "Character Name\n(Elizabeth/Darcy)"}
tiers = ["pronoun", "role_noun", "name"]
x = np.arange(len(tiers))
width = 0.3

for i, gender in enumerate(["female", "male"]):
    color = C_FEMALE if gender == "female" else C_MALE
    label = "Female" if gender == "female" else "Male"
    vals = []
    for tier in tiers:
        row = tier_data[(tier_data["gender"] == gender) & (tier_data["tier"] == tier)]
        vals.append(row["agency_ratio"].values[0] if len(row) > 0 else 0)

    bars = ax.bar(x + (i - 0.5) * width, vals, width, color=color, label=label, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Parity (0.5)")
ax.set_xticks(x)
ax.set_xticklabels([tier_labels_map[t] for t in tiers])
ax.set_ylabel("Agency Ratio")
ax.set_ylim(0, 1)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("agency_tier_detail.png", dpi=300)
print("Figure 3 saved: agency_tier_detail.png")
plt.close()

print("\nAll agency figures saved successfully!")
