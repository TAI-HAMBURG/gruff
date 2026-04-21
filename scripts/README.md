# GRUFF Scripts

This folder contains the data-generation and model-scoring scripts used for the GRUFF pronoun evaluation setup.

## Overview

The scripts generate German evaluation items with different pronoun systems, add contextual distractors, sample balanced subsets, and score language models on the resulting TSV files.

### Pronoun systems
The code works with four pronoun/article variants:
- masculine: `er`
- feminine: `sie`
- de-e system: `en`
- xier/gender-star system: `xier`

These variants are defined in:
- `pronouns.py`
- `article.py`
- `occupations.py`
- `participants.py`
- `pairings.py`

`pairings.py` controls how article forms and pronoun forms are aligned, especially for the two non-binary variants.

## Files

### Data generation
- `baseline_task.py`  
  Expands `data/task.tsv` into `tasks_no_context.tsv` by replacing `$ARTICLE_AND_OCCUPATION` with all article/occupation variants and adding the corresponding gold pronoun.

- `add_context.py`  
  Adds explicit and implicit context from `data/context.tsv` to an existing task file. It creates several output files with increasing context depth.

- `sample_templates.py`  
  Samples balanced subsets from generated `e*.tsv` files for fixed random seeds (`11`, `13`, `15`).

- `sample_for_humans.py`  
  Creates a masked evaluation sample (`sampled_for_humans.tsv`) for human annotation from the `11_*.tsv` files.

### Model evaluation
- `score_models.py`  
  Scores Hugging Face models on one or more TSV input files.
  - encoder models: pseudo log-likelihood via `minicons`
  - decoder models: average token log-probability over fully verbalized sentences
  - chat/flan models: prompt-based generation via `prompt.py`

- `prompt.py`  
  Contains prompt templates and chat-format wrappers for supported instruction-tuned models.

### Configuration and mappings
- `constants.py`  
  Contains placeholders for:
  - `HF_ACCESS_TOKEN`
  - `huggingface_home`

- `pronouns.py`  
  Maps pronoun placeholders such as `$NOM_PRONOUN` to the four surface forms.

- `article.py`  
  Defines article variants: `Der`, `Die`, `De`, `Dier`.

- `occupations.py`  
  Maps occupations to four gendered / gender-inclusive surface forms.

- `participants.py`  
  Maps participant nouns to four gendered / gender-inclusive surface forms.

- `pairings.py`  
  Provides the helper used to align article, referent, and pronoun variants consistently.

## Expected input files

The scripts expect the repository structure used in this project:

- `data/task.tsv`
- `data/context.tsv`

### `data/task.tsv`
Expected columns:
- `occupation`
- `participant`
- `sentence`
- `pronoun_type`
- `word`
- `article_and_occupation`

### `data/context.tsv`
Expected columns:
- `pronoun_type`
- `polarity`
- `explicit_template`
- `implicit_template`

## Installation

Recommended Python version: **3.10+**

Install dependencies, e.g.:

```bash
pip install pandas numpy torch transformers minicons
```

Depending on your analysis workflow, you may also want:

```bash
pip install matplotlib seaborn scipy jupyter
```

## Configuration

Before running `score_models.py`, adjust `constants.py`:

```python
HF_ACCESS_TOKEN='YOUR_HF_ACCESS_TOKEN_HERE'
huggingface_home='/path/to/huggingface_cache'
```

For public uploads, do **not** commit a real access token.

## Usage

Run all commands from the project root.

### 1. Generate tasks without context
```bash
python scripts/baseline_task.py
```
Creates:
- `tasks_no_context.tsv`

### 2. Add context to tasks
```bash
python scripts/add_context.py tasks_no_context.tsv data/context.tsv
```
Creates multiple files such as:
- `eo_task.tsv`
- `eo_ep_task.tsv`
- `eo_ep_ip_task.tsv`
- `eo_ep_ip_ip_task.tsv`
- `eo_ep_ip_ip_ip_task.tsv`
- `eo_ep_ip_ip_ip_ip_task.tsv`

## File name conventions
The generated filenames encode the context structure:
- `e` = explicit context
- `i` = implicit context
- `o` = occupation is the referenced entity
- `p` = participant is the referenced entity

Example:
- `eo_ep_ip_task.tsv` = explicit occupation context, then explicit participant context, then implicit participant context

### 3. Sample balanced subsets
```bash
python scripts/sample_templates.py
```
Creates sampled files for seeds `11`, `13`, and `15`, for example:
- `11_eo_task.tsv`
- `13_eo_ep_task.tsv`
- `15_eo_ep_ip_task.tsv`

### 4. Create a human-evaluation sample
```bash
python scripts/sample_for_humans.py
```
Creates:
- `sampled_for_humans.tsv`

### 5. Score language models
```bash
python scripts/score_models.py 11_*.tsv 13_*.tsv 15_*.tsv
```
For each input file, the script creates an output directory named after the file stem and writes one TSV per model.

Example output:
- `11_eo_task/deepset_gbert-base.tsv`
- `11_eo_task/FacebookAI_xlm-roberta-base.tsv`
- `11_eo_task/mayflowergmbh_Llama-3-SauerkrautLM-8b-Instruct-AWQ.tsv`

## Output format

### Generated task files
Typical columns:
- `occupation`
- `participant`
- `sentence`
- `pronoun_type`
- `word`
- `article_and_occupation`
- `pronoun`
- `uid`
- `confuse_pronoun`

### Model score files
Typical columns:
- `sentence`
- `verbalized_token`
- `pronoun_type`
- `occupation`
- `participant`
- `word`
- `article_and_occupation`
- `p_er`
- `p_sie`
- `p_en`
- `p_xier`
- `pronoun` (if available in the input)

## Notes

- `baseline_task.py` runs its logic at top level, so it writes `tasks_no_context.tsv` immediately when executed.
- `score_models.py` may require a GPU for larger decoder or AWQ models.
- The current model list is defined directly inside `score_models.py`.
- `prompt.py` contains prompt wrappers, but prompt-based evaluation is only used for model names containing `chat` or `flan`.

## Minimal reproduction pipeline

```bash
python scripts/baseline_task.py
python scripts/add_context.py tasks_no_context.tsv data/context.tsv
python scripts/sample_templates.py
python scripts/sample_for_humans.py
python scripts/score_models.py 11_*.tsv 13_*.tsv 15_*.tsv
```

---

## Note

This README was created with AI assistance based on the scripts and project structure in this repository.
