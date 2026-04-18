"""
Gendered Verb Frame Visualization
Reads verb_frame_results.csv and produces two figures.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import sys
import os

if not os.path.exists("verb_frame_results.csv"):
    print("ERROR: verb_frame_results.csv not found!")
    print("Run verb_frame_analysis.py first.")
    sys.exit(1)

df = pd.read_csv("verb_frame_results.csv")
print("Loaded verb_frame_results.csv")
print(df.to_string(index=False))

C_F = "#d6604d"
C_M = "#4393c3"
sns.set_theme(style="whitegrid")

verbs = list(df["verb"])
themes = list(df["theme"])
f_ratios = list(df["f_ratio"])
m_ratios = list(df["m_ratio"])
gaps = list(df["gap"])

# Handle p-values that might be empty strings
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

# ============================================================
# FIGURE 1 — COMPARISON OVERVIEW
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle("Gendered Power Across Four Verb Frames in Pride and Prejudice",
             fontsize=17, fontweight='bold')

# --- Panel 1: Agency ratios side by side ---
x = np.arange(len(verbs))
w = 0.3

bf = axes[0].bar(x - w/2, f_ratios, w, color=C_F, label="Female", alpha=0.85)
bm = axes[0].bar(x + w/2, m_ratios, w, color=C_M, label="Male", alpha=0.85)
axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Parity (0.5)")
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"'{v}'" for v in verbs], fontsize=11)
axes[0].set_ylabel("Agency Ratio")
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Agency Ratio by Verb\nagent / (agent + patient)", fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)

for bar, val in zip(bf, f_ratios):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bm, m_ratios):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

for i, p in enumerate(pvals):
    if p is not None and p < 0.05:
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
        top = max(f_ratios[i], m_ratios[i])
        axes[0].text(i, top + 0.05, stars, ha='center', fontsize=14, fontweight='bold')

# --- Panel 2: Gap chart ---
colors_gap = [C_M if g > 0 else C_F for g in gaps]
bars_gap = axes[1].barh(verbs, gaps, color=colors_gap, alpha=0.85, height=0.5)
axes[1].axvline(0, color="gray", linestyle="-", alpha=0.5)
axes[1].set_xlabel("Agency Gap (Male ratio - Female ratio)")
axes[1].set_title("Gender Agency Gap\n(+ = men more agentive, - = women more agentive)",
                   fontsize=13, fontweight='bold')
axes[1].invert_yaxis()

for bar, gap, p in zip(bars_gap, gaps, pvals):
    offset = 0.01 if gap > 0 else -0.01
    ha = 'left' if gap > 0 else 'right'
    sig = ""
    if p is not None and p < 0.05:
        sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *"
    axes[1].text(gap + offset, bar.get_y() + bar.get_height()/2,
                 f"{gap:+.3f}{sig}", ha=ha, va='center', fontsize=11, fontweight='bold')

# --- Panel 3: Stats summary ---
axes[2].axis('off')
lines = ["Verb Frame Summary", "=" * 40, ""]

for i in range(len(verbs)):
    p = pvals[i]
    cv = cramers_list[i]

    if p is not None:
        if p < 0.001: sig_str = "p < 0.001 ***"
        elif p < 0.01: sig_str = f"p = {p:.4f} **"
        elif p < 0.05: sig_str = f"p = {p:.4f} *"
        else: sig_str = f"p = {p:.4f} (ns)"
    else:
        sig_str = "n/a"

    direction = "Women more PATIENT" if gaps[i] > 0 else "Women more AGENT"

    lines.append(f"'{verbs[i]}' - {themes[i]}")
    lines.append(f"  {direction}")
    lines.append(f"  Fisher's exact: {sig_str}")
    if cv is not None:
        eff = "large" if cv >= 0.5 else "medium" if cv >= 0.3 else "small" if cv >= 0.1 else "negligible"
        lines.append(f"  Cramer's V = {cv:.3f} ({eff})")
    lines.append("")

axes[2].text(0.05, 0.95, "\n".join(lines),
             transform=axes[2].transAxes, fontsize=11, fontfamily='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
axes[2].set_title("Statistical Summary", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig("verb_frame_comparison.png", dpi=300)
print("Figure 1 saved: verb_frame_comparison.png")
plt.close()


# ============================================================
# FIGURE 2 — INDIVIDUAL VERB DETAIL
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("Gender x Syntactic Role for Four Key Verb Frames",
             fontsize=17, fontweight='bold')

for idx in range(len(df)):
    row = df.iloc[idx]
    ax = axes[idx // 2, idx % 2]

    positions = np.arange(2)
    w = 0.3

    agent_vals = [int(row["agent_f"]), int(row["agent_m"])]
    patient_vals = [int(row["patient_f"]), int(row["patient_m"])]

    b1 = ax.bar(positions - w/2, agent_vals, w, color=[C_F, C_M], alpha=0.85)
    b2 = ax.bar(positions + w/2, patient_vals, w, color=[C_F, C_M], alpha=0.4)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Female", "Male"], fontsize=11)
    ax.set_ylabel("Count")

    # Title with significance
    p = pvals[idx]
    sig_str = ""
    if p is not None:
        if p < 0.001: sig_str = " (p < 0.001 ***)"
        elif p < 0.01: sig_str = f" (p = {p:.3f} **)"
        elif p < 0.05: sig_str = f" (p = {p:.3f} *)"
        else: sig_str = f" (p = {p:.3f}, ns)"

    ax.set_title(f"'{row['verb']}' - {row['theme']}{sig_str}",
                 fontsize=13, fontweight='bold')

    # Count labels on bars
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    str(int(h)), ha='center', fontsize=10, fontweight='bold')

    # Ratio annotation
    ax.text(0.98, 0.95,
            f"F ratio: {row['f_ratio']:.3f}\nM ratio: {row['m_ratio']:.3f}",
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Shared legend
a_patch = mpatches.Patch(color='gray', alpha=0.85, label='Agent (subject)')
p_patch = mpatches.Patch(color='gray', alpha=0.4, label='Patient (object + passive)')
fig.legend(handles=[a_patch, p_patch], loc='lower center',
           ncol=2, fontsize=11, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig("verb_frame_detail.png", dpi=300)
print("Figure 2 saved: verb_frame_detail.png")
plt.close()

print("\nAll figures saved!")
