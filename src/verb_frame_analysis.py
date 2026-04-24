"""
Gendered Verb Frame Analysis — Pride and Prejudice
Analyzes gender asymmetry in agent/patient roles for four key verbs:
marry, tell, admire, persuade.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import fisher_exact
    HAS_SCIPY = True
except ImportError:
    print("WARNING: scipy not installed. Install with: pip install scipy")
    HAS_SCIPY = False


def parse_args():
    parser = argparse.ArgumentParser(description="Run verb-frame asymmetry analysis.")
    parser.add_argument("--input-csv", default="data/processed/pride_prejudice_parsed.csv")
    parser.add_argument("--output-csv", default="data/processed/verb_frame_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    df = pd.read_csv(input_csv)

    female = {
        "she", "her", "herself", "hers",
        "woman", "lady", "girl", "wife", "mother", "daughter", "sister", "mrs",
        "elizabeth", "jane", "lydia", "kitty", "mary", "charlotte", "caroline", "georgiana", "catherine", "anne", "lizzy",
    }
    male = {
        "he", "him", "himself", "his",
        "man", "gentleman", "boy", "husband", "father", "son", "brother", "mr",
        "darcy", "bingley", "wickham", "collins", "fitzwilliam", "bennet",
    }

    def get_gender(word):
        w = str(word).lower().strip()
        if w in female:
            return "female"
        if w in male:
            return "male"
        return "other"

    verbs = {
        "marry": "Institutional power",
        "tell": "Informational power",
        "admire": "Evaluative gaze",
        "persuade": "Social influence",
    }

    rows = []
    for verb, theme in verbs.items():
        agents = df[(df["Dep"] == "nsubj") & (df["Head_Lemma"] == verb)]
        objects = df[(df["Dep"] == "dobj") & (df["Head_Lemma"] == verb)]
        passives = df[(df["Dep"] == "nsubjpass") & (df["Head_Lemma"] == verb)]

        agent_f = sum(1 for lemma in agents["Lemma"] if get_gender(lemma) == "female")
        agent_m = sum(1 for lemma in agents["Lemma"] if get_gender(lemma) == "male")
        patient_f = sum(1 for lemma in list(objects["Lemma"]) + list(passives["Lemma"]) if get_gender(lemma) == "female")
        patient_m = sum(1 for lemma in list(objects["Lemma"]) + list(passives["Lemma"]) if get_gender(lemma) == "male")

        f_total = agent_f + patient_f
        m_total = agent_m + patient_m
        f_ratio = agent_f / f_total if f_total > 0 else 0.0
        m_ratio = agent_m / m_total if m_total > 0 else 0.0
        gap = m_ratio - f_ratio

        odds, pval, cramers = None, None, None
        if HAS_SCIPY and f_total > 0 and m_total > 0:
            table = [[agent_f, patient_f], [agent_m, patient_m]]
            odds, pval = fisher_exact(table)
            from scipy.stats import chi2_contingency

            chi2, _, _, _ = chi2_contingency(table, correction=False)
            n = agent_f + patient_f + agent_m + patient_m
            cramers = np.sqrt(chi2 / n) if n > 0 else 0.0

        rows.append(
            {
                "verb": verb,
                "theme": theme,
                "agent_f": agent_f,
                "agent_m": agent_m,
                "patient_f": patient_f,
                "patient_m": patient_m,
                "f_ratio": round(f_ratio, 4),
                "m_ratio": round(m_ratio, 4),
                "gap": round(gap, 4),
                "odds": round(odds, 4) if odds is not None else "",
                "pval": round(pval, 6) if pval is not None else "",
                "cramers": round(cramers, 4) if cramers is not None else "",
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(output_csv, index=False)

    metadata = {
        "script": "verb_frame_analysis.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows": int(len(result)),
        "scipy_available": HAS_SCIPY,
    }
    (output_csv.parent / "run_metadata_verb_frame_analysis.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
