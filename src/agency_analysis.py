"""
Agency Analysis — Stuhler (2024) Framework
============================================
Computes gender-based agency ratios from agent (subject) and patient (object)
roles extracted via dependency parsing.

Methodological basis:
  - Stuhler (2024, PNAS): agency_ratio = agent / (agent + patient)
  - Stuhler (2022, Sociological Methods & Research): 6-motif semantic grammar
  - Piper (2024): verb semantic categories for agency breakdown

Reads CSVs from analysis.py. Run analysis.py first.

Outputs:
  - agency_ratios.csv
  - agency_verb_profile.csv
  - Terminal summary report

Usage:
  python agency_analysis.py
"""

import pandas as pd
import numpy as np
import sys
import os
from collections import defaultdict

# ============================================================
# 1. LOAD DATA
# ============================================================

required = [
    "gendered_subject_verbs.csv", "gendered_object_verbs.csv",
    "gendered_possessives.csv", "gendered_adjectives.csv",
    "normalization_counts.csv",
]
for f in required:
    if not os.path.exists(f):
        print(f"ERROR: {f} not found! Run analysis.py first.")
        sys.exit(1)

df_agent = pd.read_csv("gendered_subject_verbs.csv")
df_patient = pd.read_csv("gendered_object_verbs.csv")
df_poss = pd.read_csv("gendered_possessives.csv")
df_adj = pd.read_csv("gendered_adjectives.csv")
norm = pd.read_csv("normalization_counts.csv")

print("=" * 60)
print(" AGENCY ANALYSIS — STUHLER (2024) FRAMEWORK")
print(" Pride and Prejudice: Gendered Agency Ratios")
print("=" * 60)

print(f"\n  Agent (subject) records:  {len(df_agent):,}")
print(f"  Patient (object) records: {len(df_patient):,}")
print(f"  Possessive records:       {len(df_poss):,}")
print(f"  Description records:      {len(df_adj):,}")


# ============================================================
# 2. CORE AGENCY RATIOS
# ============================================================

print("\n" + "-" * 60)
print(" 1. CORE AGENCY RATIOS")
print("-" * 60)

results = []

for gender in ["female", "male"]:
    agent_n = len(df_agent[df_agent["gender"] == gender])
    patient_n = len(df_patient[df_patient["gender"] == gender])
    described_n = len(df_adj[df_adj["gender"] == gender])
    possessor_n = len(df_poss[df_poss["gender"] == gender])

    total_agency = agent_n + patient_n
    agency_ratio = agent_n / total_agency if total_agency > 0 else 0

    results.append({
        "gender": gender, "tier": "all",
        "agent_count": agent_n, "patient_count": patient_n,
        "described_count": described_n, "possessor_count": possessor_n,
        "total_agency_events": total_agency,
        "agency_ratio": round(agency_ratio, 4),
    })

    print(f"\n  {gender.upper()}:")
    print(f"    Agent (subject) count:  {agent_n:,}")
    print(f"    Patient (object) count: {patient_n:,}")
    print(f"    Agency ratio:           {agency_ratio:.4f}  ", end="")
    if agency_ratio > 0.5:
        print(f"(> 0.5 = more ACTIVE)")
    elif agency_ratio < 0.5:
        print(f"(< 0.5 = more PASSIVE)")
    else:
        print(f"(= 0.5 = BALANCED)")
    print(f"    Description count:      {described_n:,}")
    print(f"    Possessive count:       {possessor_n:,}")


# ============================================================
# 3. TIER-LEVEL AGENCY RATIOS
# ============================================================

print("\n" + "-" * 60)
print(" 2. TIER-LEVEL AGENCY RATIOS")
print("-" * 60)

for gender in ["female", "male"]:
    print(f"\n  {gender.upper()}:")
    for tier in ["pronoun", "role_noun", "name"]:
        tier_name = {"pronoun": "Pronouns", "role_noun": "Role Nouns", "name": "Character Names"}[tier]
        agent_n = len(df_agent[(df_agent["gender"] == gender) & (df_agent["tier"] == tier)])
        patient_n = len(df_patient[(df_patient["gender"] == gender) & (df_patient["tier"] == tier)])
        total = agent_n + patient_n
        ratio = agent_n / total if total > 0 else 0

        results.append({
            "gender": gender, "tier": tier,
            "agent_count": agent_n, "patient_count": patient_n,
            "described_count": len(df_adj[(df_adj["gender"] == gender) & (df_adj["tier"] == tier)]),
            "possessor_count": len(df_poss[(df_poss["gender"] == gender) & (df_poss["tier"] == tier)]),
            "total_agency_events": total,
            "agency_ratio": round(ratio, 4),
        })

        arrow = "+" if ratio > 0.5 else ("-" if ratio < 0.5 else "=")
        print(f"    {tier_name:20s}  agent={agent_n:4d}  patient={patient_n:4d}  "
              f"ratio={ratio:.4f} {arrow}")


# ============================================================
# 4. VERB PROFILE: AGENT vs PATIENT BY SEMANTIC CATEGORY
# ============================================================

print("\n" + "-" * 60)
print(" 3. VERB PROFILE: AGENT vs PATIENT BY CATEGORY")
print("-" * 60)

VERB_CATEGORIES = {
    "cognition": {"think", "know", "believe", "suppose", "understand", "hope",
                  "wish", "imagine", "expect", "remember", "forget", "consider",
                  "mean", "wonder", "doubt", "fear", "mind"},
    "perception": {"see", "hear", "feel", "look", "watch", "notice", "observe",
                   "find", "perceive"},
    "communication": {"say", "tell", "speak", "talk", "reply", "answer", "ask",
                      "cry", "add", "declare", "write", "read", "call",
                      "mention", "assure", "convince", "persuade", "invite"},
    "motion": {"come", "go", "walk", "return", "leave", "enter", "run",
               "arrive", "move", "turn", "follow", "approach"},
    "action": {"do", "make", "take", "give", "send", "bring", "get",
               "set", "put", "open", "close", "keep", "pay", "marry"},
    "emotion": {"love", "like", "hate", "enjoy", "please", "surprise",
                "satisfy", "delight", "admire", "respect", "despise"},
}

def categorize_verb(verb: str) -> str:
    for cat, verbs in VERB_CATEGORIES.items():
        if verb in verbs: return cat
    return "other"

verb_profile_rows = []

for gender in ["female", "male"]:
    agent_subset = df_agent[df_agent["gender"] == gender].copy()
    agent_subset["category"] = agent_subset["verb"].apply(categorize_verb)
    agent_cats = agent_subset["category"].value_counts()

    patient_subset = df_patient[df_patient["gender"] == gender].copy()
    patient_subset["category"] = patient_subset["verb"].apply(categorize_verb)
    patient_cats = patient_subset["category"].value_counts()

    print(f"\n  {gender.upper()}:")
    print(f"    {'Category':18s} {'Agent':>8s} {'Patient':>8s} {'Agent%':>8s} {'Patient%':>8s} {'Diff':>8s}")
    print(f"    {'-'*52}")

    all_cats = sorted(set(list(agent_cats.index) + list(patient_cats.index)))
    for cat in all_cats:
        a = agent_cats.get(cat, 0)
        p = patient_cats.get(cat, 0)
        a_pct = (a / len(agent_subset) * 100) if len(agent_subset) > 0 else 0
        p_pct = (p / len(patient_subset) * 100) if len(patient_subset) > 0 else 0
        diff = a_pct - p_pct
        diff_note = "-> agent" if diff > 2 else ("-> patient" if diff < -2 else "")

        print(f"    {cat:18s} {a:8d} {p:8d} {a_pct:7.1f}% {p_pct:7.1f}% {diff:+7.1f}% {diff_note}")

        verb_profile_rows.append({
            "gender": gender, "category": cat,
            "agent_count": a, "patient_count": p,
            "agent_pct": round(a_pct, 2), "patient_pct": round(p_pct, 2),
            "diff_pct": round(diff, 2),
        })


# ============================================================
# 5. GENDER AGENCY GAP
# ============================================================

print("\n" + "-" * 60)
print(" 4. GENDER AGENCY GAP")
print("-" * 60)

f_ratio = [r for r in results if r["gender"] == "female" and r["tier"] == "all"][0]["agency_ratio"]
m_ratio = [r for r in results if r["gender"] == "male" and r["tier"] == "all"][0]["agency_ratio"]
gap = m_ratio - f_ratio

print(f"\n  Female agency ratio:  {f_ratio:.4f}")
print(f"  Male agency ratio:    {m_ratio:.4f}")
print(f"  Agency gap:           {gap:+.4f}")

if gap > 0:
    print(f"\n  -> Male characters are {gap:.4f} points more active (agentive) than female.")
    print(f"     Consistent with Stuhler (2024)'s findings for 19th-century fiction.")
elif gap < 0:
    print(f"\n  -> Female characters are {abs(gap):.4f} points more active than male.")
    print(f"     May reflect Austen's female-centered narrative perspective.")
else:
    print(f"\n  -> Perfect balance: male and female characters are equally active.")


# ============================================================
# 6. STUHLER 6-MOTIF SUMMARY
# ============================================================

print("\n" + "-" * 60)
print(" 5. STUHLER (2022) MOTIF SUMMARY")
print("-" * 60)

f_total = norm[norm["category"] == "female_total"]["token_count"].values[0]
m_total = norm[norm["category"] == "male_total"]["token_count"].values[0]

for gender in ["female", "male"]:
    total = f_total if gender == "female" else m_total
    r = [x for x in results if x["gender"] == gender and x["tier"] == "all"][0]

    print(f"\n  {gender.upper()} (total tokens: {total:,}):")
    print(f"    1. Actions (agent nsubj):         {r['agent_count']:4d}  ({r['agent_count']/total*1000:.1f}/1000)")
    print(f"    2. Treatments (patient dobj):      {r['patient_count']:4d}  ({r['patient_count']/total*1000:.1f}/1000)")
    print(f"    3. Descriptions (amod+acomp):      {r['described_count']:4d}  ({r['described_count']/total*1000:.1f}/1000)")
    print(f"    4. Possessions (poss):             {r['possessor_count']:4d}  ({r['possessor_count']/total*1000:.1f}/1000)")
    print(f"    -> Agency ratio:                   {r['agency_ratio']:.4f}")


# ============================================================
# 7. CSV OUTPUTS
# ============================================================

print("\n" + "-" * 60)
print(" CSV FILES")
print("-" * 60)

pd.DataFrame(results).to_csv("agency_ratios.csv", index=False)
print("  agency_ratios.csv saved.")

pd.DataFrame(verb_profile_rows).to_csv("agency_verb_profile.csv", index=False)
print("  agency_verb_profile.csv saved.")

print("\n" + "=" * 60)
print(" AGENCY ANALYSIS COMPLETE")
print("=" * 60)
