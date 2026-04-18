# GitHub setup — run these on your Windows machine

These are the exact commands to run to get `C:\prideprejudice` onto GitHub as a public repo called **austen-corpus-study**. Run them in **Command Prompt** (not PowerShell, to match your usual workflow).

**Before you start:**
- Make sure you have Git installed. Check with: `git --version`
- Make sure you're signed in to GitHub in a browser (you'll create the empty repo there in Step 5).

---

## Step 1 — Copy the scaffolding files I made into your project

I generated the following files in your Cowork workspace folder. Copy each one into `C:\prideprejudice`:

- `README.md`
- `LICENSE`
- `CITATION.cff`
- `.gitignore`
- `requirements.txt`

You can either drag-and-drop from File Explorer, or run this from Command Prompt (adjust the source path if your Cowork workspace is elsewhere):

```cmd
set COWORK=%USERPROFILE%\Documents\Cowork\Pride and Prejudice corpus study
copy "%COWORK%\README.md"         C:\prideprejudice\
copy "%COWORK%\LICENSE"           C:\prideprejudice\
copy "%COWORK%\CITATION.cff"      C:\prideprejudice\
copy "%COWORK%\.gitignore"        C:\prideprejudice\
copy "%COWORK%\requirements.txt"  C:\prideprejudice\
```

If the Cowork folder path is different on your machine, just set `COWORK` to the right location.

---

## Step 2 — Reorganize into the standard layout (optional but recommended)

The README describes this layout:

```
src/          Python scripts
data/raw/     Project Gutenberg source
data/processed/   Pipeline outputs (CSV/JSON)
figures/      visualize*.py output PNGs
docs/         proposal.docx, final_paper.docx, presentation.pdf
```

To move your existing flat files into it, run:

```cmd
cd C:\prideprejudice

mkdir src
mkdir data
mkdir data\raw
mkdir data\processed
mkdir figures
mkdir docs

move corpus_builder.py        src\
move analysis.py              src\
move agency_analysis.py       src\
move verb_frame_analysis.py   src\
move visualize.py             src\
move visualize_agency.py      src\
move visualize_verb_frames.py src\
```

Then move whatever raw text, processed CSVs, figures, and .docx files you have into the matching folders. If any of your scripts reference paths like `"corpus.txt"` directly, you'll need to update them to `"data/raw/pg1342.txt"` (or whatever you named the file) — otherwise the pipeline will break when you run it fresh.

**Skip this step if you'd rather keep your flat structure** — just know the README's layout section will be aspirational rather than accurate.

Create empty placeholder files so `data/processed/` and `figures/` show up in the repo even if they're empty:

```cmd
type nul > data\processed\.gitkeep
type nul > figures\.gitkeep
```

---

## Step 3 — Initialize Git

```cmd
cd C:\prideprejudice

git init
git branch -M main
git config user.name  "Servan Gediz Bakış"
git config user.email "servanbakis@gmail.com"
```

(The `config` lines are local to this repo — they don't override any global settings.)

---

## Step 4 — Stage and commit

```cmd
git add .
git status
```

Check the `git status` output. Things to look for:
- Are `.venv/` or `__pycache__/` listed? They shouldn't be — the `.gitignore` handles them. If they are, something went wrong with the copy in Step 1.
- Is the transformer model file listed? It shouldn't be — it's also in `.gitignore`.
- Is anything else surprisingly large? If `git status` shows a file over, say, 50 MB, think twice before committing. GitHub rejects single files over 100 MB.

When the status looks clean:

```cmd
git commit -m "Initial commit: computational gender analysis of Pride and Prejudice"
```

---

## Step 5 — Create the empty repo on GitHub

Go to **https://github.com/new** and:

- **Repository name:** `austen-corpus-study`
- **Description:** *Computational gender analysis of Jane Austen's Pride and Prejudice — dependency parsing, agency ratios, verb-frame statistics.*
- **Visibility:** Public
- **Do NOT** initialize with a README, .gitignore, or license — you already have them locally.

Click "Create repository". GitHub will show you a page with push instructions. The part you need is already in Step 6 below.

---

## Step 6 — Push

Replace `<your-username>` with your GitHub username, then run:

```cmd
git remote add origin https://github.com/<your-username>/austen-corpus-study.git
git push -u origin main
```

You'll be prompted for credentials. GitHub doesn't accept passwords over HTTPS anymore — use a **personal access token** instead of your password:
1. Go to https://github.com/settings/tokens
2. "Generate new token (classic)" → check `repo` scope → generate
3. Copy the token and paste it when Git prompts for a password

Windows Credential Manager will remember it, so you only do this once.

---

## Step 7 — Verify

Visit `https://github.com/<your-username>/austen-corpus-study`. You should see:
- README rendering with the project overview
- File tree showing `src/`, `data/`, `figures/`, `docs/`
- "About" section (click the gear, add the description and topics like `corpus-linguistics`, `digital-humanities`, `spacy`, `jane-austen`)
- A "Cite this repository" button on the right (powered by `CITATION.cff`)

---

## If something goes wrong

**"fatal: remote origin already exists"** — you already ran `git remote add`. Use `git remote set-url origin https://github.com/<your-username>/austen-corpus-study.git` instead.

**Pushed a file you didn't mean to** (e.g., large model, `.env`) — tell me what happened and I'll walk you through removing it from history with `git filter-repo` or BFG.

**Files too big** — GitHub's hard limit is 100 MB per file and a soft warning at 50 MB. If your transformer model or a pickle slipped through `.gitignore`, remove it, commit the removal, then re-push.

---

## What I did NOT do

- I did not push anything on your behalf (I can't authenticate to GitHub from here).
- I did not touch your actual `.py` files, `.docx` files, or corpus text — those are all local to your machine and I don't have copies.
- I did not add your GitHub username to the README or scaffolding — do a find-and-replace on `<your-username>` in the README once you know the URL, or just leave it as a placeholder since reproducing from your public repo will use whatever URL shows in the address bar.
