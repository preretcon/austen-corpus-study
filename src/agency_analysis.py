"""
Agency Analysis — Stuhler (2024) Framework
============================================
Computes gender-based agency ratios from agent (subject) and patient (object)
roles extracted via dependency parsing.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compute agency ratios and verb-profile outputs.")
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/processed")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "gendered_subject_verbs.csv",
        "gendered_object_verbs.csv",
        "gendered_possessives.csv",
        "gendered_adjectives.csv",
        "normalization_counts.csv",
    ]
    for f in required:
        if not os.path.exists(input_dir / f):
            print(f"ERROR: {input_dir / f} not found! Run analysis.py first.")
            sys.exit(1)

    df_agent = pd.read_csv(input_dir / "gendered_subject_verbs.csv")
    df_patient = pd.read_csv(input_dir / "gendered_object_verbs.csv")
    df_poss = pd.read_csv(input_dir / "gendered_possessives.csv")
    df_adj = pd.read_csv(input_dir / "gendered_adjectives.csv")
    norm = pd.read_csv(input_dir / "normalization_counts.csv")

    results = []

    for gender in ["female", "male"]:
        agent_n = len(df_agent[df_agent["gender"] == gender])
        patient_n = len(df_patient[df_patient["gender"] == gender])
        described_n = len(df_adj[df_adj["gender"] == gender])
        possessor_n = len(df_poss[df_poss["gender"] == gender])
        total_agency = agent_n + patient_n
        agency_ratio = agent_n / total_agency if total_agency > 0 else 0

        results.append(
            {
                "gender": gender,
                "tier": "all",
                "agent_count": agent_n,
                "patient_count": patient_n,
                "described_count": described_n,
                "possessor_count": possessor_n,
                "total_agency_events": total_agency,
                "agency_ratio": round(agency_ratio, 4),
            }
        )

    for gender in ["female", "male"]:
        for tier in ["pronoun", "role_noun", "name"]:
            agent_n = len(df_agent[(df_agent["gender"] == gender) & (df_agent["tier"] == tier)])
            patient_n = len(df_patient[(df_patient["gender"] == gender) & (df_patient["tier"] == tier)])
            total = agent_n + patient_n
            ratio = agent_n / total if total > 0 else 0

            results.append(
                {
                    "gender": gender,
                    "tier": tier,
                    "agent_count": agent_n,
                    "patient_count": patient_n,
                    "described_count": len(df_adj[(df_adj["gender"] == gender) & (df_adj["tier"] == tier)]),
                    "possessor_count": len(df_poss[(df_poss["gender"] == gender) & (df_poss["tier"] == tier)]),
                    "total_agency_events": total,
                    "agency_ratio": round(ratio, 4),
                }
            )

    verb_categories = {
        "cognition": {"think", "know", "believe", "suppose", "understand", "hope", "wish", "imagine", "expect", "remember", "forget", "consider", "mean", "wonder", "doubt", "fear", "mind"},
        "perception": {"see", "hear", "feel", "look", "watch", "notice", "observe", "find", "perceive"},
        "communication": {"say", "tell", "speak", "talk", "reply", "answer", "ask", "cry", "add", "declare", "write", "read", "call", "mention", "assure", "convince", "persuade", "invite"},
        "motion": {"come", "go", "walk", "return", "leave", "enter", "run", "arrive", "move", "turn", "follow", "approach"},
        "action": {"do", "make", "take", "give", "send", "bring", "get", "set", "put", "open", "close", "keep", "pay", "marry"},
        "emotion": {"love", "like", "hate", "enjoy", "please", "surprise", "satisfy", "delight", "admire", "respect", "despise"},
    }

    def categorize_verb(verb: str) -> str:
        for cat, verbs in verb_categories.items():
            if verb in verbs:
                return cat
        return "other"

    verb_profile_rows = []
    for gender in ["female", "male"]:
        agent_subset = df_agent[df_agent["gender"] == gender].copy()
        agent_subset["category"] = agent_subset["verb"].apply(categorize_verb)
        agent_cats = agent_subset["category"].value_counts()

        patient_subset = df_patient[df_patient["gender"] == gender].copy()
        patient_subset["category"] = patient_subset["verb"].apply(categorize_verb)
        patient_cats = patient_subset["category"].value_counts()

        all_cats = sorted(set(list(agent_cats.index) + list(patient_cats.index)))
        for cat in all_cats:
            a = agent_cats.get(cat, 0)
            p = patient_cats.get(cat, 0)
            a_pct = (a / len(agent_subset) * 100) if len(agent_subset) > 0 else 0
            p_pct = (p / len(patient_subset) * 100) if len(patient_subset) > 0 else 0
            diff = a_pct - p_pct
            verb_profile_rows.append(
                {
                    "gender": gender,
                    "category": cat,
                    "agent_count": a,
                    "patient_count": p,
                    "agent_pct": round(a_pct, 2),
                    "patient_pct": round(p_pct, 2),
                    "diff_pct": round(diff, 2),
                }
            )

    pd.DataFrame(results).to_csv(output_dir / "agency_ratios.csv", index=False)
    pd.DataFrame(verb_profile_rows).to_csv(output_dir / "agency_verb_profile.csv", index=False)

    metadata = {
        "script": "agency_analysis.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_rows": {
            "agent": int(len(df_agent)),
            "patient": int(len(df_patient)),
            "possessive": int(len(df_poss)),
            "adjective": int(len(df_adj)),
            "norm": int(len(norm)),
        },
    }
    (output_dir / "run_metadata_agency_analysis.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved: agency_ratios.csv, agency_verb_profile.csv")
    print(f"Saved metadata: {output_dir / 'run_metadata_agency_analysis.json'}")


if __name__ == "__main__":
    main()
