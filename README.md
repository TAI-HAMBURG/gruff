# GRUFF

This repository contains the files needed to generate and score the GRUFF German pronoun evaluation data.

## Repository contents

- `data/` – source TSV files for task and context generation
  - `task.tsv`
  - `context.tsv`
- `scripts/` – Python scripts for generation, sampling, and model scoring
  - `add_context.py`
  - `article.py`
  - `baseline_task.py`
  - `constants.py`
  - `occupations.py`
  - `pairings.py`
  - `participants.py`
  - `prompt.py`
  - `pronouns.py`
  - `sample_for_humans.py`
  - `sample_templates.py`
  - `score_models.py`
  - `README.md`

## Purpose

The repository supports a workflow for:
- generating German pronoun evaluation items
- adding explicit and implicit contextual distractors
- sampling balanced evaluation subsets
- scoring language models on binary and non-binary pronoun forms

## Main workflow

From the project root, the core script pipeline is:

```bash
python scripts/baseline_task.py
python scripts/add_context.py tasks_no_context.tsv data/context.tsv
python scripts/sample_templates.py
python scripts/sample_for_humans.py
python scripts/score_models.py 11_*.tsv 13_*.tsv 15_*.tsv
```

## Important note

Before sharing or publishing the repository, ensure that `scripts/constants.py` does not contain a real Hugging Face access token.

## Documentation

A more detailed description of the scripts is available in:
- `scripts/README.md`

---

## Note

This README was created with AI assistance based on the files contained in this repository.
