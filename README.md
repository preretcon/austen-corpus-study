# A Computational Gender Analysis of *Pride and Prejudice*

**Dependency Parsing and the Encoding of Gendered Agency in Austen's Syntax**

A reproducible computational linguistics study examining how gendered agency and characterization are encoded in Jane Austen's *Pride and Prejudice* through dependency parsing, corpus analysis, and statistical testing.

**Python · spaCy Transformers · pandas · SciPy · NumPy · Matplotlib**

**Author:** Servan Gediz Bakış  
**Institution:** Hacettepe University, Department of English Linguistics  
**Course:** IDB 402 — Projects in Linguistics  
**Year:** 2026

> 📄 **[Read the full research paper](paper.pdf)**

---

## Overview

This project investigates whether the grammatical structures of *Pride and Prejudice* reveal systematic differences in how male and female characters are represented.

Using the full novel as a corpus of approximately **146,000 tokens**, I developed a Python-based NLP pipeline using spaCy's transformer dependency parser to examine:

- grammatical agent and patient roles;
- subject–verb and object–verb relations;
- gendered verb and adjective distributions;
- attributive and predicative characterization;
- selected semantic verb frames;
- dialogue and narration separately;
- statistical differences in gendered syntactic patterns.

The project combines **computational linguistics, corpus analysis, statistical testing, and discourse-oriented interpretation**.

A central finding is what I describe as the **agency paradox**: aggregate syntactic measures initially suggest greater agency for female characters, but closer examination shows that much of this result is shaped by narrative focalization through Elizabeth Bennet and by the concentration of female subjects in perception and cognition verbs.

---

## Key Findings

### 1. The agency paradox

Across the complete corpus, female referents have a higher aggregate agent ratio than male referents:

- **Female:** 0.8177
- **Male:** 0.7620

Taken alone, this could appear to indicate greater grammatical agency for female characters.

However, verb-level analysis shows that female subjects are disproportionately associated with **perception, cognition, and interior-state verbs**, reflecting Elizabeth Bennet's role as the novel's focalizing consciousness.

The result illustrates an important methodological point: aggregate corpus metrics can be misleading when their underlying linguistic distributions are not inspected.

---

### 2. Gender asymmetry in the *marry* frame

The aggregate pattern changes substantially when the analysis is restricted to the verb *marry*.

- Female referents occur as patients in **26 of 39 cases (66.7%)**
- Male referents occur as patients in **10 of 37 cases (27.0%)**

Female referents are therefore approximately **2.5× as likely** to occupy the patient position within the marriage frame.

The difference is statistically significant:

- **Fisher's exact test:** *p* < .001
- **Cramér's V:** 0.397

This suggests that broad measures of grammatical agency can conceal asymmetries concentrated in particular semantic and institutional contexts.

---

### 3. Verb profiles

Female subjects show a stronger association with verbs involving:

- perception;
- cognition;
- interior state.

Male subjects show greater concentration in:

- motion;
- communication;
- overt action.

These distributions help explain why a single aggregate agency measure does not fully describe how agency is linguistically encoded.

---

### 4. Characterization

The analysis distinguishes between different dependency relations involved in characterization:

- `amod` — attributive adjectives integrated into noun phrases;
- `acomp` and related predicative structures — adjectives used to evaluate or describe referents.

Treating these constructions separately provides a more interpretable account of gendered characterization than aggregate adjective frequency alone.

---

## Research Questions

The project addresses four primary questions:

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

The transformer-based spaCy model was selected after pilot testing because it provided more reliable dependency analyses for Austen's syntactically complex nineteenth-century prose than the smaller statistical pipeline.

The corpus-building stage produces structured token-level information including:

- token and lemma;
- part of speech;
- dependency relation;
- syntactic head;
- dialogue/narration status;
- sentence identifier;
- token identifier.

---

### Gender Attribution

Gendered references are identified through a three-level heuristic cascade:

1. gendered pronouns;
2. gendered role nouns and titles;
3. named characters.

Ambiguous surnames such as *Bennet* and *Bingley* are disambiguated using nearby titles and contextual cues where possible.

This is a deliberately interpretable heuristic rather than a full coreference-resolution system.

---

### Grammatical Agency

Agency is operationalized using dependency relations as syntactic proxies:

- `nsubj` → agent-like grammatical role;
- `dobj` / `obj` → patient-like grammatical role.

The primary measure is:

```text
agent / (agent + patient)
```

This measure is treated as an operationalization of grammatical agency rather than a complete representation of semantic, social, or narrative agency.

---

### Statistical Analysis

Sparse contingency tables for selected semantic frames are evaluated using:

- **Fisher's exact test**
- **Cramér's V** for effect size

The project therefore combines descriptive corpus statistics with inferential testing.

---

### Dialogue and Narration

Quoted spans are identified programmatically so that character dialogue can be distinguished from narration.

This makes it possible to inspect whether observed patterns arise primarily from character speech or from Austen's narrative voice and focalization.

---

## Methodological Development

The final pipeline was not the first implementation of the project.

An earlier version used spaCy's smaller:

```text
en_core_web_sm
```

model.

During pilot analysis, the smaller statistical model proved less reliable for some of the syntactically complex structures found in Austen's prose. The project was therefore migrated to:

```text
en_core_web_trf
```

for the final analysis.

The earlier implementation is intentionally retained under:

```text
archive/sm-model/
```

to document the methodological evolution of the project.

The archived version is **not** the pipeline used to produce the results reported in the final paper. The current implementation in `src/` should be treated as the canonical version of the analysis.

---

## Why Sentiment Analysis Was Removed

An earlier version of the project also included sentiment analysis using off-the-shelf models.

Pilot testing showed that these systems handled Austen's nineteenth-century prose, irony, and free indirect discourse poorly.

Rather than retain a weak measurement simply because the tool was available, the sentiment component was removed from the final study.

The final analysis instead focuses on:

- dependency relations;
- grammatical agency;
- verb frames;
- collocational patterns;
- characterization.

This methodological revision reflects a broader principle of the project:

> **Computational methods should be selected according to the linguistic properties of the data and the validity of the resulting measurement, rather than applied simply because they are available.**

---

## Reproducing the Analysis

### 1. Clone the repository

```bash
git clone https://github.com/preretcon/austen-corpus-study.git
cd austen-corpus-study
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

The transformer model is installed separately and is therefore not bundled directly through `requirements.txt`.

---

### 3. Build the parsed corpus

```bash
python src/corpus_builder.py
```

By default, this reads:

```text
data/raw/pride_and_prejudice_clean.txt
```

and produces:

```text
data/processed/pride_prejudice_parsed.csv
```

The script also records metadata about the corpus-building run.

---

### 4. Run the gendered linguistic analysis

```bash
python src/analysis.py
```

This produces structured datasets covering:

- subject–verb relations;
- object–verb relations;
- adjectival characterization;
- possessive constructions;
- normalization counts.

Outputs are written to:

```text
data/processed/
```

---

### 5. Compute agency measures

```bash
python src/agency_analysis.py
```

Principal outputs include:

```text
data/processed/agency_ratios.csv
data/processed/agency_verb_profile.csv
```

---

### 6. Run the verb-frame analysis

```bash
python src/verb_frame_analysis.py
```

This evaluates selected verbs including:

- `marry`
- `tell`
- `admire`
- `persuade`

and writes the resulting statistics to:

```text
data/processed/verb_frame_results.csv
```

---

## Reproducibility Notes

The results reported in the final paper were produced using:

```text
en_core_web_trf
```

Some scripts include a fallback to `en_core_web_sm` when the transformer model is unavailable. This fallback is intended for convenience and experimentation; **the transformer pipeline should be used to reproduce the reported results**.

Processed datasets are included in the repository so that intermediate and final outputs can be inspected without rerunning the complete transformer pipeline.

Selected stages also generate run metadata to improve traceability across different executions of the analysis.

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
│       ├── normalization_counts.csv
│       ├── agency_ratios.csv
│       ├── agency_verb_profile.csv
│       ├── verb_comparison.csv
│       ├── adjective_comparison.csv
│       └── verb_frame_results.csv
│
├── figures/
│
└── archive/
    └── sm-model/    # earlier pilot implementation using en_core_web_sm
```

---

## Scope and Limitations

This project uses dependency relations as interpretable proxies for grammatical agency. They should not be understood as complete representations of semantic, social, or narrative agency.

Gender attribution is heuristic and based on pronouns, role nouns, titles, named characters, and limited contextual disambiguation rather than full coreference resolution.

Dialogue detection is based on quotation boundaries rather than a dedicated discourse-parsing system.

The study also examines a single novel. Its findings therefore describe patterns within *Pride and Prejudice* and should not automatically be generalized to Jane Austen's complete work or nineteenth-century fiction more broadly.

These limitations are one reason the project emphasizes the inspection of individual linguistic patterns alongside aggregate quantitative metrics.

---

## Full Paper

The complete paper contains the theoretical background, methodological discussion, statistical analysis, interpretation, and broader discussion of the findings.

📄 **[Read the full research paper](paper.pdf)**

---

## Citation

Citation metadata is provided in:

[`CITATION.cff`](CITATION.cff)

---

## License

The code in this repository is released under the terms specified in:

[`LICENSE`](LICENSE)
