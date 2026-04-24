"""
Corpus Builder — Pride and Prejudice (Upgraded)
=================================================
Tokenizes and parses the full text using spaCy, producing a CSV
with dependency relations, POS tags, and dialogue/narration labels.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import spacy


def parse_args():
    parser = argparse.ArgumentParser(description="Build parsed corpus CSV from cleaned text.")
    parser.add_argument("--input-text", default="data/raw/pride_and_prejudice_clean.txt")
    parser.add_argument("--output-csv", default="data/processed/pride_prejudice_parsed.csv")
    parser.add_argument("--model", default="en_core_web_trf")
    return parser.parse_args()


def main():
    args = parse_args()
    input_text = Path(args.input_text)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" CORPUS BUILDER — Pride and Prejudice")
    print("=" * 60)

    print("\n1. Loading language model...")
    model_name = args.model

    try:
        nlp = spacy.load(model_name, disable=["ner"])
        print(f"   Loaded: {model_name} (transformer/statistical model)")
    except OSError:
        print(f"   WARNING: {model_name} not found.")
        print("   Falling back to en_core_web_sm...")
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner"])
            model_name = "en_core_web_sm"
            print("   Loaded: en_core_web_sm (statistical model)")
        except OSError:
            print("   ERROR: No spaCy model found. Install one first.")
            sys.exit(1)

    nlp.max_length = 1500000

    print("\n2. Reading text file...")
    if not input_text.exists():
        print(f"   ERROR: input text not found: {input_text}")
        sys.exit(1)

    text = input_text.read_text(encoding="utf-8")
    text = re.sub(r"\s+", " ", text)
    print(f"   Text length: {len(text):,} characters")

    print("\n3. Detecting dialogue spans...")
    dialogue_mask = bytearray(len(text))

    in_dialogue = False
    for i, ch in enumerate(text):
        if ch == "\u201c":
            in_dialogue = True
            dialogue_mask[i] = 1
        elif ch == "\u201d":
            dialogue_mask[i] = 1
            in_dialogue = False
        elif in_dialogue:
            dialogue_mask[i] = 1

    dialogue_chars = sum(dialogue_mask)
    narration_chars = len(text) - dialogue_chars
    print(f"   Dialogue: {dialogue_chars:,} characters ({dialogue_chars/len(text)*100:.1f}%)")
    print(f"   Narration: {narration_chars:,} characters ({narration_chars/len(text)*100:.1f}%)")

    print("\n4. Running spaCy pipeline...")
    if model_name == "en_core_web_trf":
        print("   (Transformer model is slower — this may take 10-15 minutes)")
    else:
        print("   (This may take 1-2 minutes)")

    doc = nlp(text)

    print("\n5. Building DataFrame...")
    data = []
    sent_id = 0
    prev_sent = None

    for token in doc:
        if token.sent != prev_sent:
            sent_id += 1
            prev_sent = token.sent

        is_dialogue = bool(dialogue_mask[token.idx]) if token.idx < len(dialogue_mask) else False

        data.append(
            {
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
            }
        )

    df = pd.DataFrame(data)

    print("\n--- COMPLETE ---")
    print(f"   Total tokens:     {len(df):,}")
    print(f"   Total sentences:  {sent_id:,}")
    print(f"   Dialogue tokens:  {df['Is_Dialogue'].sum():,} ({df['Is_Dialogue'].mean()*100:.1f}%)")
    print(f"   Narration tokens: {(~df['Is_Dialogue']).sum():,} ({(~df['Is_Dialogue']).mean()*100:.1f}%)")
    print(f"   spaCy model used: {model_name}")

    df.to_csv(output_csv, index=False)
    print(f"\n   Saved: {output_csv}")

    metadata = {
        "script": "corpus_builder.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "input_text": str(input_text),
        "output_csv": str(output_csv),
        "token_count": int(len(df)),
        "sentence_count": int(sent_id),
    }
    metadata_path = output_csv.parent / "run_metadata_corpus_builder.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"   Saved metadata: {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
