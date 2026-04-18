"""
Gendered Verb Frame Analysis — Pride and Prejudice
Analyzes gender asymmetry in agent/patient roles for four key verbs:
marry, tell, admire, persuade.
"""

import pandas as pd
import numpy as np

try:
    from scipy.stats import fisher_exact
    HAS_SCIPY = True
except ImportError:
    print("WARNING: scipy not installed. Install with: pip install scipy")
    HAS_SCIPY = False

# ============================================================
# 1. LOAD AND SETUP
# ============================================================

df = pd.read_csv("pride_prejudice_parsed.csv")

FEMALE = {
    'she', 'her', 'herself', 'hers',
    'woman', 'lady', 'girl', 'wife', 'mother', 'daughter', 'sister', 'mrs',
    'elizabeth', 'jane', 'lydia', 'kitty', 'mary', 'charlotte',
    'caroline', 'georgiana', 'catherine', 'anne', 'lizzy',
}
MALE = {
    'he', 'him', 'himself', 'his',
    'man', 'gentleman', 'boy', 'husband', 'father', 'son', 'brother', 'mr',
    'darcy', 'bingley', 'wickham', 'collins', 'fitzwilliam', 'bennet',
}

def get_gender(word):
    w = str(word).lower().strip()
    if w in FEMALE:
        return "female"
    elif w in MALE:
        return "male"
    else:
        return "other"

VERBS = {
    "marry": "Institutional power",
    "tell": "Informational power",
    "admire": "Evaluative gaze",
    "persuade": "Social influence",
}

print("=" * 65)
print(" GENDERED VERB FRAME ANALYSIS")
print(" Pride and Prejudice")
print("=" * 65)

# ============================================================
# 2. COUNT AGENTS AND PATIENTS FOR EACH VERB
# ============================================================

rows = []

for verb, theme in VERBS.items():
    print(f"\n--- '{verb}' ({theme}) ---")

    # Agents = active subjects (nsubj)
    agents = df[(df["Dep"] == "nsubj") & (df["Head_Lemma"] == verb)]

    # Patients = direct objects (dobj) + passive subjects (nsubjpass)
    objects = df[(df["Dep"] == "dobj") & (df["Head_Lemma"] == verb)]
    passives = df[(df["Dep"] == "nsubjpass") & (df["Head_Lemma"] == verb)]

    # Classify gender
    agent_f = 0
    agent_m = 0
    for lemma in agents["Lemma"]:
        g = get_gender(lemma)
        if g == "female":
            agent_f += 1
        elif g == "male":
            agent_m += 1

    patient_f = 0
    patient_m = 0
    for lemma in list(objects["Lemma"]) + list(passives["Lemma"]):
        g = get_gender(lemma)
        if g == "female":
            patient_f += 1
        elif g == "male":
            patient_m += 1

    # Agency ratios
    f_total = agent_f + patient_f
    m_total = agent_m + patient_m
    f_ratio = agent_f / f_total if f_total > 0 else 0.0
    m_ratio = agent_m / m_total if m_total > 0 else 0.0
    gap = m_ratio - f_ratio

    # Fisher's exact test
    odds = None
    pval = None
    cramers = None

    if HAS_SCIPY and f_total > 0 and m_total > 0:
        table = [[agent_f, patient_f], [agent_m, patient_m]]
        odds, pval = fisher_exact(table)
        n = agent_f + patient_f + agent_m + patient_m
        # Cramers V from chi2
        from scipy.stats import chi2_contingency
        chi2, _, _, _ = chi2_contingency(table, correction=False)
        cramers = np.sqrt(chi2 / n) if n > 0 else 0.0

    # Print
    print(f"  Female:  agent={agent_f}, patient={patient_f}, ratio={f_ratio:.3f}")
    print(f"  Male:    agent={agent_m}, patient={patient_m}, ratio={m_ratio:.3f}")
    print(f"  Gap (M-F): {gap:+.3f}")
    if pval is not None:
        stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"  Fisher p={pval:.6f} {stars}, odds={odds:.3f}, Cramers V={cramers:.3f}")

    rows.append({
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
    })

# ============================================================
# 3. SAVE
# ============================================================

result = pd.DataFrame(rows)
result.to_csv("verb_frame_results.csv", index=False)
print("\nSaved: verb_frame_results.csv")
print(result.to_string(index=False))
print("\nDone.")
