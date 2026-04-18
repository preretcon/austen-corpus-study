"""
Gendered Collocation Analysis — Pride and Prejudice (Upgraded)
================================================================
Upgrades: en_core_web_trf, context-aware Bennet/Bingley disambiguation,
dialogue vs narration split, amod+acomp, 3-tier, normalization.
"""

import spacy
import pandas as pd
import re
import sys
from collections import defaultdict

# ============================================================
# 1. GENDERED WORD LISTS
# ============================================================

FEMALE_PRONOUNS = {"she", "her", "herself", "hers"}
MALE_PRONOUNS = {"he", "him", "himself", "his"}
FEMALE_ROLE_NOUNS = {"woman", "lady", "girl", "wife", "mother", "daughter", "sister", "mrs"}
MALE_ROLE_NOUNS = {"man", "gentleman", "boy", "husband", "father", "son", "brother", "mr"}
FEMALE_NAMES = {"elizabeth", "jane", "lydia", "kitty", "mary", "charlotte",
                "caroline", "georgiana", "catherine", "anne", "lizzy"}
MALE_NAMES = {"darcy", "wickham", "collins", "fitzwilliam"}


def classify_gender_tier(token, doc):
    lemma = token.lemma_.lower()
    if lemma in FEMALE_PRONOUNS: return ("female", "pronoun")
    if lemma in MALE_PRONOUNS: return ("male", "pronoun")
    if lemma in FEMALE_ROLE_NOUNS: return ("female", "role_noun")
    if lemma in MALE_ROLE_NOUNS: return ("male", "role_noun")
    if lemma in FEMALE_NAMES: return ("female", "name")
    if lemma in MALE_NAMES: return ("male", "name")

    if lemma in ("bennet", "bingley"):
        prev = ""
        if token.i > 0:
            prev = doc[token.i - 1].text.lower().rstrip(".")
        prev2 = ""
        if token.i > 1:
            prev2 = doc[token.i - 2].text.lower().rstrip(".")
        if prev in ("mrs", "miss") or prev2 in ("mrs", "miss"):
            return ("female", "name")
        if prev in ("elizabeth", "jane", "lydia", "caroline", "eliza", "lizzy"):
            return ("female", "name")
        if prev == "mr" or prev2 == "mr":
            return ("male", "name")
        return ("male", "name")

    return (None, None)


# ============================================================
# 2. DIALOGUE DETECTION
# ============================================================

def build_dialogue_mask(text):
    mask = bytearray(len(text))
    in_dialogue = False
    for i, ch in enumerate(text):
        if ch == '\u201c':
            in_dialogue = True
            mask[i] = 1
        elif ch == '\u201d':
            mask[i] = 1
            in_dialogue = False
        elif in_dialogue:
            mask[i] = 1
    return mask


def is_token_dialogue(token, dialogue_mask):
    if token.idx < len(dialogue_mask):
        return bool(dialogue_mask[token.idx])
    return False


# ============================================================
# 3. LOAD MODEL AND TEXT
# ============================================================

print("=" * 65)
print(" EXPANDED GENDERED COLLOCATION ANALYSIS (Upgraded)")
print("=" * 65)

print("\n1. Loading language model...")
try:
    nlp = spacy.load("en_core_web_trf", disable=["ner"])
    MODEL = "en_core_web_trf"
    print("   Loaded: en_core_web_trf (transformer)")
except OSError:
    print("   WARNING: en_core_web_trf not found. Using en_core_web_sm.")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    MODEL = "en_core_web_sm"
nlp.max_length = 1500000

print("\n2. Reading text...")
with open("pride_and_prejudice_clean.txt", "r", encoding="utf-8") as f:
    text = f.read()
text = re.sub(r'\s+', ' ', text)

print("   Building dialogue mask...")
dialogue_mask = build_dialogue_mask(text)
d_pct = sum(dialogue_mask) / len(text) * 100
print(f"   Dialogue: {d_pct:.1f}% of text")

print("   Running spaCy...")
if MODEL == "en_core_web_trf":
    print("   (Transformer model — may take 10-15 minutes)")
doc = nlp(text)
print(f"   Total tokens: {len(doc):,}\n")


# ============================================================
# 4. BASE FREQUENCY COUNTS
# ============================================================

gender_counts = {"female": 0, "male": 0}
tier_counts = defaultdict(int)
dial_counts = {"female_dialogue": 0, "female_narration": 0,
               "male_dialogue": 0, "male_narration": 0}

for token in doc:
    g, t = classify_gender_tier(token, doc)
    if g:
        gender_counts[g] += 1
        tier_counts[f"{g}_{t}"] += 1
        mode = "dialogue" if is_token_dialogue(token, dialogue_mask) else "narration"
        dial_counts[f"{g}_{mode}"] += 1

print("3. Base frequency counts:")
for g in ["female", "male"]:
    G = g.upper()
    print(f"   {G}: {gender_counts[g]:,} total")
    print(f"     Pronouns: {tier_counts[g+'_pronoun']:,}  Role nouns: {tier_counts[g+'_role_noun']:,}  Names: {tier_counts[g+'_name']:,}")
    print(f"     Dialogue: {dial_counts[g+'_dialogue']:,}  Narration: {dial_counts[g+'_narration']:,}")

def norm1k(count, gender):
    return round((count / gender_counts[gender]) * 1000, 2) if gender_counts[gender] > 0 else 0


# ============================================================
# 5. ADJECTIVE COLLOCATIONS
# ============================================================

print("\n4. Extracting adjectives...")

amod_results = []
for token in doc:
    if token.pos_ == "ADJ" and token.dep_ == "amod":
        g, t = classify_gender_tier(token.head, doc)
        if g:
            amod_results.append({
                "adjective": token.lemma_.lower(), "head_word": token.head.lemma_.lower(),
                "gender": g, "tier": t, "dep_type": "amod",
                "is_dialogue": is_token_dialogue(token, dialogue_mask),
                "example": token.sent.text.strip()[:150],
            })
print(f"   amod: {len(amod_results)}")

acomp_results = []
for token in doc:
    if token.pos_ == "ADJ" and token.dep_ in ("acomp", "attr"):
        for child in token.head.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                g, t = classify_gender_tier(child, doc)
                if g:
                    acomp_results.append({
                        "adjective": token.lemma_.lower(), "head_word": child.lemma_.lower(),
                        "gender": g, "tier": t, "dep_type": token.dep_,
                        "is_dialogue": is_token_dialogue(token, dialogue_mask),
                        "example": token.sent.text.strip()[:150],
                    })
                break
print(f"   acomp/attr: {len(acomp_results)}")

all_adj = amod_results + acomp_results
df_adj = pd.DataFrame(all_adj)

print("\n   --- ADJECTIVE SUMMARY ---")
for g, G in [("female", "FEMALE"), ("male", "MALE")]:
    s = df_adj[df_adj["gender"] == g]
    d = s[s["is_dialogue"] == True]
    n = s[s["is_dialogue"] == False]
    print(f"\n   {G}: {len(s)} total ({len(d)} dialogue, {len(n)} narration)")
    print(f"   Top 10 overall: {', '.join(f'{a}({c})' for a,c in s['adjective'].value_counts().head(10).items())}")
    print(f"   Top 5 dialogue: {', '.join(f'{a}({c})' for a,c in d['adjective'].value_counts().head(5).items())}")
    print(f"   Top 5 narration: {', '.join(f'{a}({c})' for a,c in n['adjective'].value_counts().head(5).items())}")


# ============================================================
# 6. SUBJECT VERBS
# ============================================================

print("\n5. Extracting subject-verb relations...")

verb_results = []
for token in doc:
    if token.dep_ in ("nsubj", "nsubjpass"):
        g, t = classify_gender_tier(token, doc)
        if g and token.head.pos_ == "VERB" and token.head.lemma_ != "be":
            verb_results.append({
                "subject": token.lemma_.lower(), "verb": token.head.lemma_.lower(),
                "gender": g, "tier": t, "dep_type": token.dep_,
                "is_passive": token.dep_ == "nsubjpass",
                "is_dialogue": is_token_dialogue(token, dialogue_mask),
                "example": token.sent.text.strip()[:150],
            })

df_verbs = pd.DataFrame(verb_results)
print(f"   Total: {len(df_verbs)}")

for g, G in [("female", "FEMALE"), ("male", "MALE")]:
    s = df_verbs[df_verbs["gender"] == g]
    d = s[s["is_dialogue"] == True]
    n = s[s["is_dialogue"] == False]
    print(f"\n   {G}: {len(s)} total ({len(d)} dialogue, {len(n)} narration)")
    print(f"   Top 10 dialogue: {', '.join(f'{v}({c})' for v,c in d['verb'].value_counts().head(10).items())}")
    print(f"   Top 10 narration: {', '.join(f'{v}({c})' for v,c in n['verb'].value_counts().head(10).items())}")


# ============================================================
# 7. OBJECT RELATIONS
# ============================================================

print("\n6. Extracting object relations...")

object_results = []
for token in doc:
    if token.dep_ in ("dobj", "obj"):
        g, t = classify_gender_tier(token, doc)
        if g and token.head.pos_ == "VERB":
            object_results.append({
                "object_word": token.lemma_.lower(), "verb": token.head.lemma_.lower(),
                "gender": g, "tier": t, "dep_type": token.dep_,
                "is_dialogue": is_token_dialogue(token, dialogue_mask),
                "example": token.sent.text.strip()[:150],
            })

df_objects = pd.DataFrame(object_results)
print(f"   Total: {len(df_objects)}")


# ============================================================
# 8. POSSESSIVES
# ============================================================

print("\n7. Extracting possessives...")

poss_results = []
for token in doc:
    if token.dep_ == "poss":
        g, t = classify_gender_tier(token, doc)
        if g:
            poss_results.append({
                "possessor": token.lemma_.lower(), "possessed": token.head.lemma_.lower(),
                "gender": g, "tier": t,
                "is_dialogue": is_token_dialogue(token, dialogue_mask),
                "example": token.sent.text.strip()[:150],
            })

df_poss = pd.DataFrame(poss_results)
print(f"   Total: {len(df_poss)}")


# ============================================================
# 9. DIALOGUE vs NARRATION AGENCY
# ============================================================

print("\n" + "=" * 65)
print(" DIALOGUE vs NARRATION: AGENCY RATIOS")
print("=" * 65)

for mode, mode_name in [(True, "DIALOGUE"), (False, "NARRATION")]:
    print(f"\n  --- {mode_name} ---")
    for g in ["female", "male"]:
        a = len(df_verbs[(df_verbs["gender"] == g) & (df_verbs["is_dialogue"] == mode)])
        p = len(df_objects[(df_objects["gender"] == g) & (df_objects["is_dialogue"] == mode)])
        total = a + p
        ratio = a / total if total > 0 else 0
        print(f"    {g.upper():8s}  agent={a:4d}  patient={p:4d}  ratio={ratio:.4f}")

print(f"\n  --- COMBINED ---")
for g in ["female", "male"]:
    a = len(df_verbs[df_verbs["gender"] == g])
    p = len(df_objects[df_objects["gender"] == g])
    total = a + p
    ratio = a / total if total > 0 else 0
    print(f"    {g.upper():8s}  agent={a:4d}  patient={p:4d}  ratio={ratio:.4f}")


# ============================================================
# 10. SAVE CSVs
# ============================================================

print("\n8. Saving CSV files...")
df_adj.to_csv("gendered_adjectives.csv", index=False)
df_verbs.to_csv("gendered_subject_verbs.csv", index=False)
df_objects.to_csv("gendered_object_verbs.csv", index=False)
df_poss.to_csv("gendered_possessives.csv", index=False)

adj_comp = []
for adj in df_adj["adjective"].unique():
    fc = len(df_adj[(df_adj["adjective"]==adj) & (df_adj["gender"]=="female")])
    mc = len(df_adj[(df_adj["adjective"]==adj) & (df_adj["gender"]=="male")])
    adj_comp.append({"adjective": adj, "f_raw": fc, "m_raw": mc, "total": fc+mc})
adj_df = pd.DataFrame(adj_comp).sort_values("total", ascending=False).head(15)
adj_df["f_norm"] = adj_df["f_raw"].apply(lambda x: norm1k(x, "female"))
adj_df["m_norm"] = adj_df["m_raw"].apply(lambda x: norm1k(x, "male"))
adj_df.to_csv("adjective_comparison.csv", index=False)

verb_comp = []
for v in df_verbs["verb"].unique():
    fc = len(df_verbs[(df_verbs["verb"]==v) & (df_verbs["gender"]=="female")])
    mc = len(df_verbs[(df_verbs["verb"]==v) & (df_verbs["gender"]=="male")])
    verb_comp.append({"verb": v, "f_raw": fc, "m_raw": mc, "total": fc+mc})
verb_df = pd.DataFrame(verb_comp).sort_values("total", ascending=False).head(15)
verb_df["f_norm"] = verb_df["f_raw"].apply(lambda x: norm1k(x, "female"))
verb_df["m_norm"] = verb_df["m_raw"].apply(lambda x: norm1k(x, "male"))
verb_df.to_csv("verb_comparison.csv", index=False)

norm_info = pd.DataFrame([
    {"category": "female_total", "token_count": gender_counts["female"]},
    {"category": "male_total", "token_count": gender_counts["male"]},
    {"category": "female_pronoun", "token_count": tier_counts["female_pronoun"]},
    {"category": "female_role_noun", "token_count": tier_counts["female_role_noun"]},
    {"category": "female_name", "token_count": tier_counts["female_name"]},
    {"category": "male_pronoun", "token_count": tier_counts["male_pronoun"]},
    {"category": "male_role_noun", "token_count": tier_counts["male_role_noun"]},
    {"category": "male_name", "token_count": tier_counts["male_name"]},
    {"category": "female_dialogue", "token_count": dial_counts["female_dialogue"]},
    {"category": "female_narration", "token_count": dial_counts["female_narration"]},
    {"category": "male_dialogue", "token_count": dial_counts["male_dialogue"]},
    {"category": "male_narration", "token_count": dial_counts["male_narration"]},
])
norm_info.to_csv("normalization_counts.csv", index=False)
print("   All 7 CSV files saved (with is_dialogue column).")

print("\n" + "=" * 65)
print(f" ANALYSIS COMPLETE (model: {MODEL})")
print(f" {len(all_adj)} adj + {len(verb_results)} verbs + {len(object_results)} obj + {len(poss_results)} poss")
print("=" * 65)
