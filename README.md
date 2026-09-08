# A Computational Gender Analysis of *Pride and Prejudice*

**Dependency Parsing and the Encoding of Gendered Agency in Austen's Syntax**

An undergraduate computational linguistics research project examining how gendered agency and characterization are encoded in Jane Austen's *Pride and Prejudice* through dependency parsing, corpus analysis, and statistical testing.

**Author:** Servan Gediz Bakış  
**Institution:** Hacettepe University, Department of English Linguistics  
**Course:** IDB 402 — Projects in Linguistics  
**Year:** 2026

> 📄 **[Read the full research paper](paper/austen final version.pdf)**

---

## Overview

This project investigates whether the grammatical structures of *Pride and Prejudice* reveal systematic differences in how male and female characters are represented.

Using the full novel as a corpus of approximately **146,000 tokens**, I developed a Python-based NLP pipeline using **spaCy's transformer dependency parser** to examine:

- agent and patient roles;
- subject–verb and object–verb relations;
- gendered verb and adjective distributions;
- attributive vs. predicative characterization;
- selected semantic verb frames, particularly *marry*;
- dialogue and narration separately.

The project combines **computational linguistics**, **corpus methods**, **statistical analysis**, and **Critical Discourse Analysis / feminist digital humanities**.

A central finding is what I term the **agency paradox**: aggregate syntactic measures initially appear to give female characters greater agency, but closer analysis shows that this result is strongly shaped by narrative focalization through Elizabeth Bennet.

---

## Key findings

### The agency paradox

Across the complete corpus, female characters have a higher aggregate agent ratio than male characters:

- **Female:** 0.8177
- **Male:** 0.7620

Taken alone, this appears to contradict large-scale studies reporting lower syntactic agency for female characters in nineteenth-century fiction.

However, verb-level analysis shows that female agency is concentrated particularly in **perception and cognition** verbs, reflecting Elizabeth Bennet's role as the novel's focalizing consciousness.

### Gender asymmetry in the *marry* frame

The aggregate pattern changes substantially when the analysis is restricted to the verb *marry*.

- Female referents occur as patients in **26 of 39 cases (66.7%)**
- Male referents occur as patients in **10 of 37 cases (27.0%)**

Female referents are therefore roughly **2.5× as likely** to occupy the patient position within the marriage frame.

The difference is statistically significant:

- **Fisher's exact test:** *p* < .001
- **Cramér's V:** 0.397

This suggests that aggregate agency statistics can conceal gendered asymmetries concentrated in particular semantic and institutional contexts.

### Verb profiles

Female subjects are more strongly associated with verbs of:

- perception;
- cognition;
- interior state.

Male subjects show greater concentration in:

- motion;
- communication;
- overt action.

### Characterization

The analysis distinguishes between:

- `amod` — attributive adjectives integrated into noun phrases;
- `acomp` — predicative adjectives used to evaluate characters.

Separating these dependency relations provides a more interpretable picture of gendered characterization than aggregate adjective frequencies.

---

## Research questions

The project addresses four main questions:

1. How do male and female characters differ in their distribution across agent and patient roles?
2. Which verbs and adjectives are most distinctively associated with male and female referents?
3. Do specific semantic frames reveal gender asymmetries that aggregate agency measures obscure?
4. Does narrative focalization affect the interpretation of corpus-level agency metrics?

---

## Methodology

### Corpus

The full text of *Pride and Prejudice* was obtained from **Project Gutenberg (EBook #1342)** and cleaned of Gutenberg metadata before analysis.

### NLP pipeline

The corpus is processed using:

- **Python**
- **spaCy**
- `en_core_web_trf`
- **pandas**
- **SciPy**
- **NumPy**
- **Matplotlib**

The transformer-based spaCy model was selected after pilot testing because of its stronger performance on Austen's syntactically complex prose.

### Gender attribution

Gendered references are identified through a three-level cascade:

1. gendered pronouns;
2. gendered role nouns and titles;
3. named characters.

Ambiguous surnames such as *Bennet* and *Bingley* are resolved using preceding titles such as *Mr.*, *Mrs.*, and *Miss*.

### Agency

Agency is operationalized through dependency relations:

- `nsubj` → agent;
- `dobj` → patient.

The primary agency measure is:

`agent / (agent + patient)`

### Statistical analysis

Sparse semantic-frame contingency tables are evaluated using:

- **Fisher's exact test**
- **Cramér's V** for effect size

### Dialogue and narration

Quotation marks are used to separate character speech from narration, allowing linguistic patterns associated with Austen's narrative voice to be distinguished from dialogue.

---

## Why sentiment analysis was removed

An earlier version of the project included sentiment analysis using off-the-shelf models.

This component was removed after pilot testing showed that standard sentiment systems handled Austen's nineteenth-century prose, irony, and free indirect discourse poorly.

Rather than retain a weak measure, the final study focuses on:

- dependency relations;
- verb frames;
- collocations;
- grammatical agency;
- characterization.

This methodological revision became an important part of the project: computational methods should be selected according to the linguistic properties of the data rather than used simply because they are available.

---

## Repository structure

```text
austen-corpus-study/
│
├── README.md
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
│   └── processed/
│
├── figures/
│
└── paper/
    └── Servan_Bakis_Pride_and_Prejudice_Gender_Agency.pdf
