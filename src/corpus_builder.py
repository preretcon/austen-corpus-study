"""
Corpus Builder — Pride and Prejudice (Upgraded)
=================================================
Tokenizes and parses the full text using spaCy, producing a CSV
with dependency relations, POS tags, and dialogue/narration labels.

Upgrades from original:
  - Uses en_core_web_trf (transformer model) for higher parsing accuracy
  - Adds Is_Dialogue column (dialogue vs narration detection)
  - Adds Sentence_ID for sentence-level grouping
  - Adds Token_Index for positional reference

Requirements:
  pip install spacy torch
  python -m spacy download en_core_web_trf

Usage:
  python corpus_builder.py
"""

import spacy
import pandas as pd
import re
import sys

# ============================================================
# 1. LOAD SPACY MODEL
# ============================================================
# Prefer en_core_web_trf (transformer, higher accuracy)
# Fall back to en_core_web_sm if trf is not installed

print("=" * 60)
print(" CORPUS BUILDER — Pride and Prejudice")
print("=" * 60)

print("\n1. Loading language model...")
MODEL_NAME = "en_core_web_trf"

try:
    nlp = spacy.load(MODEL_NAME, disable=["ner"])
    print(f"   Loaded: {MODEL_NAME} (transformer model)")
except OSError:
    print(f"   WARNING: {MODEL_NAME} not found.")
    print(f"   Install with: python -m spacy download en_core_web_trf")
    print(f"   Falling back to en_core_web_sm...")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
        MODEL_NAME = "en_core_web_sm"
        print(f"   Loaded: en_core_web_sm (statistical model)")
    except OSError:
        print("   ERROR: No spaCy model found. Install one first.")
        sys.exit(1)

nlp.max_length = 1500000

# ============================================================
# 2. READ AND CLEAN TEXT
# ============================================================

print("\n2. Reading text file...")
with open("pride_and_prejudice_clean.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Clean whitespace
text = re.sub(r'\s+', ' ', text)
print(f"   Text length: {len(text):,} characters")

# ============================================================
# 3. DIALOGUE DETECTION
# ============================================================
# Austen's text uses Unicode curly quotes: \u201c (\u201c) and \u201d (\u201d)
# Everything between opening and closing quotes is marked as dialogue.
# This captures ~90% of dialogue accurately for Austen's style.

print("\n3. Detecting dialogue spans...")

dialogue_mask = bytearray(len(text))  # 0 = narration, 1 = dialogue

in_dialogue = False
for i, ch in enumerate(text):
    if ch == '\u201c':  # opening curly quote
        in_dialogue = True
        dialogue_mask[i] = 1
    elif ch == '\u201d':  # closing curly quote
        dialogue_mask[i] = 1
        in_dialogue = False
    elif in_dialogue:
        dialogue_mask[i] = 1

dialogue_chars = sum(dialogue_mask)
narration_chars = len(text) - dialogue_chars
print(f"   Dialogue: {dialogue_chars:,} characters ({dialogue_chars/len(text)*100:.1f}%)")
print(f"   Narration: {narration_chars:,} characters ({narration_chars/len(text)*100:.1f}%)")

# ============================================================
# 4. spaCy PROCESSING
# ============================================================

print("\n4. Running spaCy pipeline...")
if MODEL_NAME == "en_core_web_trf":
    print("   (Transformer model is slower — this may take 10-15 minutes)")
else:
    print("   (This may take 1-2 minutes)")

doc = nlp(text)

# ============================================================
# 5. BUILD DATAFRAME
# ============================================================

print("\n5. Building DataFrame...")
data = []
sent_id = 0
prev_sent = None

for token in doc:
    # Track sentence boundaries
    if token.sent != prev_sent:
        sent_id += 1
        prev_sent = token.sent

    # Check if token is inside dialogue
    is_dialogue = bool(dialogue_mask[token.idx]) if token.idx < len(dialogue_mask) else False

    data.append({
        "Token": token.text,
        "Lemma": token.lemma_.lower(),
        "POS": token.pos_,
        "Dep": token.dep_,
        "Head_Lemma": token.head.lemma_.lower(),
        "Is_Stop": token.is_stop,
        "Is_Alpha": token.is_alpha,
        "Is_Dialogue": is_dialogue,
        "Sentence_ID": sent_id,
        "Token_Index": token.i,
    })

df = pd.DataFrame(data)

# ============================================================
# 6. SUMMARY AND SAVE
# ============================================================

print(f"\n--- COMPLETE ---")
print(f"   Total tokens:     {len(df):,}")
print(f"   Total sentences:  {sent_id:,}")
print(f"   Dialogue tokens:  {df['Is_Dialogue'].sum():,} ({df['Is_Dialogue'].mean()*100:.1f}%)")
print(f"   Narration tokens: {(~df['Is_Dialogue']).sum():,} ({(~df['Is_Dialogue']).mean()*100:.1f}%)")
print(f"   spaCy model used: {MODEL_NAME}")

print(f"\n   First 5 rows:")
print(df.head().to_string(index=False))

df.to_csv("pride_prejudice_parsed.csv", index=False)
print(f"\n   Saved: pride_prejudice_parsed.csv")
print("=" * 60)
