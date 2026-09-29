# A Computational Gender Analysis of *Pride and Prejudice*

**Dependency Parsing and the Encoding of Gendered Agency in Austen's Syntax**

A reproducible computational linguistics project examining how gendered agency and characterization are encoded in Jane Austen's *Pride and Prejudice* through dependency parsing, corpus analysis, and statistical testing.

**Python · spaCy Transformers · pandas · SciPy · NumPy · Matplotlib**

**Author:** Servan Gediz Bakış  
**Institution:** Hacettepe University, Department of English Linguistics  
**Course:** IDB 402 — Projects in Linguistics  
**Year:** 2026

> 📄 **[Read the full research paper](paper.pdf)**

---

## Overview

This project investigates whether grammatical patterns in *Pride and Prejudice* reveal systematic differences in how male and female characters are represented.

Using the full novel as a corpus of approximately **146,000 tokens**, I developed a Python-based NLP pipeline built around spaCy's transformer dependency parser. The analysis examines:

- grammatical agent and patient roles;
- subject–verb and object–verb relations;
- gendered verb and adjective distributions;
- attributive and predicative characterization;
- selected semantic verb frames;
- dialogue and narration separately;
- statistical differences in gendered syntactic patterns.

The project combines **computational linguistics, corpus analysis, statistical testing, and discourse-oriented interpretation**.

A central finding is what I describe as the **agency paradox**: aggregate grammatical measures initially suggest greater agency for female characters, but closer inspection shows that much of this effect is driven by narrative focalization through Elizabeth Bennet and by the concentration of female subjects in perception and cognition verbs.

---

## Key Findings

### 1. The agency paradox

Across the complete corpus, female referents have a higher aggregate agent ratio than male referents:

- **Female:** 0.8177
- **Male:** 0.7620

Taken alone, this could suggest greater grammatical agency for female characters.

However, verb-level analysis shows that female subjects are disproportionately associated with **perception, cognition, and interior-state verbs**, reflecting Elizabeth Bennet's role as the novel's focalizing consciousness.

This demonstrates why aggregate corpus metrics need to be interpreted together with their underlying linguistic distributions.

---

### 2. Gender asymmetry in the *marry* frame

The aggregate pattern changes substantially when the analysis is restricted to the verb *marry*.

- Female referents occur as patients in **26 of 39 cases (66.7%)**
- Male referents occur as patients in **10 of 37 cases (27.0%)**

Female referents are therefore approximately **2.5× as likely** to occupy the patient position within this frame.

The difference is statistically significant:

- **Fisher's exact test:** *p* < .001
- **Cramér's V:** 0.397

This suggests that broad agency measures can conceal asymmetries concentrated in particular semantic or institutional contexts.

---

### 3. Verb profiles

Female subjects are more strongly associated with verbs of:

- perception;
- cognition;
- interior state.

Male subjects show greater concentration in:

- motion;
- communication;
- overt action.

These patterns help explain why a single aggregate agency score can be misleading without semantic interpretation.

---

### 4. Characterization

The analysis distinguishes between different dependency relations used in characterization:

- `amod` — attributive adjectives integrated into noun phrases;
- `acomp` / related predicative structures — adjectives used to evaluate or describe referents.

Treating these constructions separately produces a more interpretable view of gendered characterization than raw adjective frequency alone.

---

## Research Questions

The project addresses four main questions:

1. How do male and female referents differ in their distribution across agent and patient roles?
2. Which verbs and adjectives are most distinctively associated with male and female referents?
3. Do specific semantic frames reveal gender asymmetries that aggregate agency measures obscure?
4. Does narrative focalization affect the interpretation of corpus-level agency metrics?

---

## Methodology

### Corpus

The full text of *Pride and Prejudice* was obtained from **Project Gutenberg (EBook #1342)** and cleaned of Gutenberg metadata before analysis.

The cleaned corpus is stored in:

```text
data/raw/pride_and_prejudice_clean.txt
```

---

### NLP Pipeline

The primary analysis uses:

- **Python**
- **spaCy**
- `en_core_web_trf`
- **pandas**
- **NumPy**
- **SciPy**
- **Matplotlib**

The transformer-based spaCy pipeline was selected after pilot testing because it performed more reliably on Austen's syntactically complex nineteenth-century prose than the smaller statistical model.

The pipeline produces structured token-level data including:

- token and lemma;
- part of speech;
- dependency relation;
- syntactic head;
- dialogue/narration status;
- sentence and token identifiers.

---

### Gender Attribution

Gendered references are identified through a three-level heuristic cascade:

1. gendered pronouns;
2. gendered role nouns and titles;
3. named characters.

Ambiguous surnames such as *Bennet* and *Bingley* are disambiguated using nearby titles and contextual cues where possible.

---

### Agency

Grammatical agency is operationalized primarily through dependency relations:

- `nsubj` → agent-like role;
- `dobj` / `obj` → patient-like role.

The principal measure is:

```text
agent / (agent + patient)
```

This is used as a syntactic proxy rather than as a complete semantic definition of agency.

---

### Statistical Analysis

Sparse contingency tables for selected verb frames are evaluated using:

- **Fisher's exact test**
- **Cramér's V** for effect size

The analysis therefore combines descriptive corpus statistics with inferential testing.

---

### Dialogue and Narration

The corpus pipeline identifies quoted passages and separates dialogue from narration.

This allows patterns associated with character speech to be distinguished from patterns associated with Austen's narrative voice and focalization.

---

## Methodological Revision: Why Sentiment Analysis Was Removed

An earlier version of the project included sentiment analysis using off-the-shelf models.

Pilot testing showed that these systems handled Austen's nineteenth-century prose, irony, and free indirect discourse poorly.

Rather than retain a weak measurement simply because the tool was available, the sentiment component was removed from the final study.

The final analysis instead focuses on:

- dependency relations;
- grammatical agency;
- verb frames;
- collocational patterns;
- characterization.

This revision reflects a broader methodological principle of the project: **computational methods should be selected according to the linguistic properties of the data and the validity of the measurement, rather than applied indiscriminately.**

---

## Reproducing the Analysis

### 1. Clone the repository

```bash
git clone https://github.com/preretcon/austen-corpus-study.git
cd austen-corpus-study
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

The transformer model is installed separately because it is not distributed through `requirements.txt`.

---

### 3. Build the parsed corpus

```bash
python src/corpus_builder.py
```

This reads the cleaned text from:

```text
data/raw/pride_and_prejudice_clean.txt
```

and produces the parsed corpus in:

```text
data/processed/pride_prejudice_parsed.csv
```

---

### 4. Run the gendered linguistic analysis

```bash
python src/analysis.py
```

This produces datasets for:

- subject–verb relations;
- object–verb relations;
- adjectival characterization;
- possessive constructions;
- normalization counts.

---

### 5. Compute agency measures

```bash
python src/agency_analysis.py
```

This generates:

```text
data/processed/agency_ratios.csv
data/processed/agency_verb_profile.csv
```

---

### 6. Run verb-frame analysis

```bash
python src/verb_frame_analysis.py
```

This generates statistical results for selected verbs including *marry*, *tell*, *admire*, and *persuade*.

Results are written to:

```text
data/processed/verb_frame_results.csv
```

---

## Reproducibility Notes

The reported results were produced using spaCy's:

```text
en_core_web_trf
```

Some scripts contain a fallback to `en_core_web_sm` if the transformer model is unavailable. However, the transformer model should be used when reproducing the results reported in the paper.

Processed datasets are included in the repository so that the outputs can also be inspected without rerunning the complete transformer pipeline.

The scripts additionally record run metadata for selected stages of the analysis to improve traceability.

---

## Repository Structure

```text
austen-corpus-study/
│
├── README.md
├── paper.pdf
├── requirements.txt
├── LICENSE
├── CITATION.cff
│
├── src/
│   ├── corpus_builder.py
│   ├── analysis.py
│   ├── agency_analysis.py
│   ├── verb_frame_analysis.py
│   ├── visualize.py
│   ├── visualize_agency.py
│   └── visualize_verb_frames.py
│
├── data/
│   ├── raw/
│   │   └── pride_and_prejudice_clean.txt
│   │
│   └── processed/
│       ├── pride_prejudice_parsed.csv
│       ├── gendered_subject_verbs.csv
│       ├── gendered_object_verbs.csv
│       ├── gendered_adjectives.csv
│       ├── gendered_possessives.csv
│       ├── agency_ratios.csv
│       ├── agency_verb_profile.csv
│       └── verb_frame_results.csv
│
└── figures/
```

---

## Scope and Limitations

The project treats dependency relations as interpretable proxies for grammatical agency. They should not be understood as complete representations of semantic, social, or narrative agency.

Gender attribution is heuristic and based on pronouns, role nouns, titles, and named characters rather than full coreference resolution.

Likewise, the study focuses on a single novel. Its findings describe patterns within *Pride and Prejudice* and should not automatically be generalized to Austen's complete work or nineteenth-century fiction as a whole.

These limitations are part of the reason the project emphasizes inspection of individual linguistic patterns alongside aggregate quantitative measures.

---

## Paper

The complete research paper, including theoretical background, methodology, statistical results, interpretation, and discussion, is available here:

📄 **[A Computational Gender Analysis of *Pride and Prejudice*](paper.pdf)**

---

## Citation

Citation metadata is available in [`CITATION.cff`](CITATION.cff).

---

## License

The code in this repository is released under the terms specified in [`LICENSE`](LICENSE).
