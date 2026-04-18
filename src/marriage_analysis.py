"""
Marriage Semantic Frame Analysis — Pride and Prejudice
=======================================================
Analyzes the syntactic encoding of gender roles around the marriage
semantic frame using dependency parsing and statistical testing.

Examines: who marries (agent/subject) vs who is married (patient/object),
how marriage is characterized (adjective modifiers), and whether the
gender asymmetry in agent/patient roles is statistically significant.

Methodological basis:
  - Stuhler (2024): agent/patient extraction via dependency parsing
  - Baker (2014): collocational analysis of gendered representation
  - Fisher's exact test for small-sample contingency tables

Reads: pride_prejudice_parsed.csv (from corpus_builder.py)

Outputs:
  - marriage_frame_data.csv       : All marriage-related dependency extractions
  - marriage_contingency.csv      : Gender × Role contingency table
  - marriage_adjectives.csv       : Adjectives modifying marriage-related nouns
  - Terminal report with statistical test results

Usage:
  python marriage_analysis.py
  (Requires: pip install scipy)
"""

import pandas as pd
import numpy as np
import sys

try:
    from scipy.stats import fisher_exact, chi2_contingency
    HAS_SCIPY = True
except ImportError:
    print("WARNING: scipy not installed. Statistical tests will be skipped.")
    print("Install with: pip install scipy")
    HAS_SCIPY = False

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("pride_prejudice_parsed.csv")

print("=" * 65)
print(" MARRIAGE SEMANTIC FRAME ANALYSIS")
print(" Pride and Prejudice — Gender, Agency, and Marriage")
print("=" * 65)
print(f"\n  Corpus size: {len(df):,} tokens")

# ============================================================
# 2. GENDER CLASSIFICATION
# ============================================================

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

def classify_gender(lemma: str) -> str:
    lemma = lemma.lower().strip()
    if lemma in FEMALE_WORDS: return "female"
    if lemma in MALE_WORDS: return "male"
    return "other"


# ============================================================
# 3. CORE ANALYSIS: "MARRY" (verb)
# ============================================================

print("\n" + "-" * 65)
print(" 1. VERB 'MARRY' — Subject (Agent) vs Object (Patient)")
print("-" * 65)

# Subjects of "marry" — who decides / who acts
marry_subj = df[(df['Dep'] == 'nsubj') & (df['Head_Lemma'] == 'marry')].copy()
marry_subj['gender'] = marry_subj['Lemma'].apply(classify_gender)
marry_subj['role'] = 'subject'

# Objects of "marry" — who is acted upon
marry_obj = df[(df['Dep'].isin(['dobj', 'obj'])) & (df['Head_Lemma'] == 'marry')].copy()
marry_obj['gender'] = marry_obj['Lemma'].apply(classify_gender)
marry_obj['role'] = 'object'

# Passive subjects (nsubjpass) — "she was married" = patient despite subject position
marry_pass = df[(df['Dep'] == 'nsubjpass') & (df['Head_Lemma'] == 'marry')].copy()
marry_pass['gender'] = marry_pass['Lemma'].apply(classify_gender)
marry_pass['role'] = 'passive_subject'

print(f"\n  'Marry' subjects (active agent): {len(marry_subj)}")
print(f"  'Marry' objects (patient):       {len(marry_obj)}")
print(f"  'Marry' passive subjects:        {len(marry_pass)}")

# Print top subjects
print("\n  Top subjects of 'marry' (Who marries? Who decides?):")
subj_counts = marry_subj['Lemma'].value_counts()
for lemma, count in subj_counts.head(10).items():
    g = classify_gender(lemma)
    print(f"    {lemma:15s}  {count:2d}  ({g})")

# Print top objects
print("\n  Top objects of 'marry' (Who is married? Who is acted upon?):")
obj_counts = marry_obj['Lemma'].value_counts()
for lemma, count in obj_counts.head(10).items():
    g = classify_gender(lemma)
    print(f"    {lemma:15s}  {count:2d}  ({g})")


# ============================================================
# 4. CONTINGENCY TABLE: Gender × Syntactic Role
# ============================================================

print("\n" + "-" * 65)
print(" 2. CONTINGENCY TABLE: Gender × Role for 'marry'")
print("-" * 65)

# Count by gender (excluding "other" for the statistical test)
subj_f = len(marry_subj[marry_subj['gender'] == 'female'])
subj_m = len(marry_subj[marry_subj['gender'] == 'male'])
obj_f = len(marry_obj[marry_obj['gender'] == 'female'])
obj_m = len(marry_obj[marry_obj['gender'] == 'male'])

# Include passive subjects as patients
pass_f = len(marry_pass[marry_pass['gender'] == 'female'])
pass_m = len(marry_pass[marry_pass['gender'] == 'male'])

# Patient = direct object + passive subject
patient_f = obj_f + pass_f
patient_m = obj_m + pass_m

print(f"\n                    Subject(Agent)    Patient(Obj+Pass)")
print(f"  Female:           {subj_f:10d}       {patient_f:10d}")
print(f"  Male:             {subj_m:10d}       {patient_m:10d}")
print(f"  {'':20s} {'─'*35}")
print(f"  Total:            {subj_f+subj_m:10d}       {patient_f+patient_m:10d}")

# Agency ratios within marriage frame
f_total_marry = subj_f + patient_f
m_total_marry = subj_m + patient_m
f_marry_ratio = subj_f / f_total_marry if f_total_marry > 0 else 0
m_marry_ratio = subj_m / m_total_marry if m_total_marry > 0 else 0

print(f"\n  Female marriage agency ratio: {f_marry_ratio:.4f}  "
      f"({subj_f}/{f_total_marry})")
print(f"  Male marriage agency ratio:   {m_marry_ratio:.4f}  "
      f"({subj_m}/{m_total_marry})")

if f_marry_ratio < m_marry_ratio:
    print(f"\n  -> Women are more often PATIENT (acted upon) in marriage events.")
    print(f"     Men are more often AGENT (acting) in marriage events.")
    print(f"     Gap: {m_marry_ratio - f_marry_ratio:.4f}")


# ============================================================
# 5. STATISTICAL TESTING
# ============================================================

print("\n" + "-" * 65)
print(" 3. STATISTICAL SIGNIFICANCE TESTS")
print("-" * 65)

contingency_table = np.array([[subj_f, patient_f],
                               [subj_m, patient_m]])

# Save contingency table
ct_df = pd.DataFrame(
    contingency_table,
    index=["female", "male"],
    columns=["subject_agent", "patient_object"]
)
ct_df.to_csv("marriage_contingency.csv")

if HAS_SCIPY:
    # Fisher's exact test (preferred for small samples)
    odds_ratio, fisher_p = fisher_exact(contingency_table)
    print(f"\n  Fisher's Exact Test:")
    print(f"    Odds ratio:  {odds_ratio:.4f}")
    print(f"    p-value:     {fisher_p:.6f}")

    if fisher_p < 0.001:
        print(f"    Result:      HIGHLY SIGNIFICANT (p < 0.001)")
    elif fisher_p < 0.01:
        print(f"    Result:      SIGNIFICANT (p < 0.01)")
    elif fisher_p < 0.05:
        print(f"    Result:      SIGNIFICANT (p < 0.05)")
    else:
        print(f"    Result:      NOT SIGNIFICANT (p >= 0.05)")

    print(f"\n    Interpretation: Odds ratio {odds_ratio:.2f} means that")
    if odds_ratio < 1:
        print(f"    female words are {1/odds_ratio:.1f}x more likely to appear as")
        print(f"    PATIENT than as AGENT of 'marry', compared to male words.")
    else:
        print(f"    male words are {odds_ratio:.1f}x more likely to appear as")
        print(f"    PATIENT than as AGENT of 'marry', compared to female words.")

    # Chi-square test (for comparison, though Fisher's is better here)
    chi2, chi_p, dof, expected = chi2_contingency(contingency_table)
    print(f"\n  Chi-square Test (for comparison):")
    print(f"    Chi-square:  {chi2:.4f}")
    print(f"    p-value:     {chi_p:.6f}")
    print(f"    df:          {dof}")
    print(f"    Expected frequencies:")
    print(f"      Female subject: {expected[0][0]:.1f}  Female patient: {expected[0][1]:.1f}")
    print(f"      Male subject:   {expected[1][0]:.1f}  Male patient:   {expected[1][1]:.1f}")

    # Check expected frequency assumption
    if (expected < 5).any():
        print(f"\n    NOTE: Some expected frequencies < 5.")
        print(f"    Fisher's exact test is more reliable here than chi-square.")

    # Cramér's V (effect size)
    n = contingency_table.sum()
    k = min(contingency_table.shape)
    cramers_v = np.sqrt(chi2 / (n * (k - 1)))
    print(f"\n  Effect Size:")
    print(f"    Cramer's V:  {cramers_v:.4f}", end="")
    if cramers_v < 0.1:
        print(f"  (negligible)")
    elif cramers_v < 0.3:
        print(f"  (small)")
    elif cramers_v < 0.5:
        print(f"  (medium)")
    else:
        print(f"  (large)")

else:
    print("\n  [scipy not installed — skipping statistical tests]")
    print("  Install with: pip install scipy")


# ============================================================
# 6. ADJECTIVES MODIFYING MARRIAGE-RELATED NOUNS
# ============================================================

print("\n" + "-" * 65)
print(" 4. ADJECTIVES MODIFYING MARRIAGE-RELATED NOUNS")
print("-" * 65)

# Expanded semantic frame: marriage, match, engagement, proposal
marriage_lemmas = ['marriage', 'match', 'engagement', 'proposal', 'wedding']
adj_results = []

for target in marriage_lemmas:
    adj_amod = df[(df['POS'] == 'ADJ') & (df['Dep'] == 'amod') & (df['Head_Lemma'] == target)]
    for _, row in adj_amod.iterrows():
        adj_results.append({
            "adjective": row['Lemma'].lower(),
            "head_noun": target,
            "dep_type": "amod",
        })

df_marriage_adj = pd.DataFrame(adj_results)

if len(df_marriage_adj) > 0:
    print(f"\n  Total adjectives found: {len(df_marriage_adj)}")
    for noun in marriage_lemmas:
        subset = df_marriage_adj[df_marriage_adj["head_noun"] == noun]
        if len(subset) > 0:
            adjs = ", ".join(f"{a}({c})" for a, c in subset["adjective"].value_counts().items())
            print(f"\n  '{noun}' adjectives: {adjs}")

    # Sentiment classification of marriage adjectives
    positive_adj = {"happy", "good", "eligible", "desirable", "convenient", "positive"}
    negative_adj = {"imprudent", "unsuitable", "unequal", "improper", "unfortunate", "tacit"}

    pos_count = sum(1 for _, r in df_marriage_adj.iterrows() if r["adjective"] in positive_adj)
    neg_count = sum(1 for _, r in df_marriage_adj.iterrows() if r["adjective"] in negative_adj)
    neu_count = len(df_marriage_adj) - pos_count - neg_count

    print(f"\n  Adjective polarity breakdown:")
    print(f"    Positive (happy, eligible, good):       {pos_count}")
    print(f"    Negative (imprudent, unsuitable):       {neg_count}")
    print(f"    Neutral/other:                          {neu_count}")
else:
    print("\n  No adjectives found for marriage-related nouns.")


# ============================================================
# 7. EXPANDED FRAME: RELATED VERBS
# ============================================================

print("\n" + "-" * 65)
print(" 5. EXPANDED SEMANTIC FRAME: RELATED VERBS")
print("-" * 65)

related_verbs = ['marry', 'propose', 'engage', 'court', 'love', 'admire', 'elope']
frame_rows = []

for verb in related_verbs:
    subj = df[(df['Dep'] == 'nsubj') & (df['Head_Lemma'] == verb)]
    obj = df[(df['Dep'].isin(['dobj', 'obj'])) & (df['Head_Lemma'] == verb)]
    passv = df[(df['Dep'] == 'nsubjpass') & (df['Head_Lemma'] == verb)]

    if len(subj) + len(obj) + len(passv) == 0:
        continue

    s_f = sum(1 for _, r in subj.iterrows() if classify_gender(r['Lemma']) == 'female')
    s_m = sum(1 for _, r in subj.iterrows() if classify_gender(r['Lemma']) == 'male')
    o_f = sum(1 for _, r in obj.iterrows() if classify_gender(r['Lemma']) == 'female')
    o_m = sum(1 for _, r in obj.iterrows() if classify_gender(r['Lemma']) == 'male')
    p_f = sum(1 for _, r in passv.iterrows() if classify_gender(r['Lemma']) == 'female')
    p_m = sum(1 for _, r in passv.iterrows() if classify_gender(r['Lemma']) == 'male')

    frame_rows.append({
        "verb": verb,
        "subj_female": s_f, "subj_male": s_m, "subj_total": len(subj),
        "obj_female": o_f, "obj_male": o_m, "obj_total": len(obj),
        "pass_female": p_f, "pass_male": p_m, "pass_total": len(passv),
    })

    print(f"\n  '{verb}':")
    print(f"    Subjects:  {len(subj):3d} total  (F={s_f}, M={s_m})")
    print(f"    Objects:   {len(obj):3d} total  (F={o_f}, M={o_m})")
    if len(passv) > 0:
        print(f"    Passive:   {len(passv):3d} total  (F={p_f}, M={p_m})")

df_frame = pd.DataFrame(frame_rows)


# ============================================================
# 8. AGGREGATE: FULL MARRIAGE FRAME GENDER ASYMMETRY
# ============================================================

print("\n" + "-" * 65)
print(" 6. AGGREGATE MARRIAGE FRAME GENDER ASYMMETRY")
print("-" * 65)

if len(df_frame) > 0:
    total_subj_f = df_frame["subj_female"].sum()
    total_subj_m = df_frame["subj_male"].sum()
    total_patient_f = df_frame["obj_female"].sum() + df_frame["pass_female"].sum()
    total_patient_m = df_frame["obj_male"].sum() + df_frame["pass_male"].sum()

    print(f"\n  Across all marriage-frame verbs ({', '.join(df_frame['verb'].tolist())}):")
    print(f"                    Agent(subj)    Patient(obj+pass)")
    print(f"  Female:           {total_subj_f:7d}         {total_patient_f:7d}")
    print(f"  Male:             {total_subj_m:7d}         {total_patient_m:7d}")

    if HAS_SCIPY and (total_subj_f + total_patient_f > 0) and (total_subj_m + total_patient_m > 0):
        agg_table = np.array([[total_subj_f, total_patient_f],
                               [total_subj_m, total_patient_m]])
        odds_agg, p_agg = fisher_exact(agg_table)
        print(f"\n  Fisher's exact test (aggregate frame):")
        print(f"    Odds ratio: {odds_agg:.4f}, p = {p_agg:.6f}")
        if p_agg < 0.05:
            print(f"    SIGNIFICANT: Gender asymmetry holds across the full marriage frame.")
        else:
            print(f"    NOT significant at p < 0.05 for the aggregate frame.")


# ============================================================
# 9. CSV OUTPUTS
# ============================================================

print("\n" + "-" * 65)
print(" CSV FILES")
print("-" * 65)

# Combine all marry data
all_marry = pd.concat([
    marry_subj[['Lemma', 'gender', 'role']],
    marry_obj[['Lemma', 'gender', 'role']],
    marry_pass[['Lemma', 'gender', 'role']],
], ignore_index=True)
all_marry.to_csv("marriage_frame_data.csv", index=False)
print("  marriage_frame_data.csv saved.")

ct_df.to_csv("marriage_contingency.csv")
print("  marriage_contingency.csv saved.")

if len(df_marriage_adj) > 0:
    df_marriage_adj.to_csv("marriage_adjectives.csv", index=False)
    print("  marriage_adjectives.csv saved.")

if len(df_frame) > 0:
    df_frame.to_csv("marriage_expanded_frame.csv", index=False)
    print("  marriage_expanded_frame.csv saved.")

print("\n" + "=" * 65)
print(" MARRIAGE ANALYSIS COMPLETE")
print("=" * 65)
