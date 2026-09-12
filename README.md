# gpt2-ptbr-218m

A Portuguese GPT-2-like language model trained from scratch and subsequently instruction-tuned with supervised fine-tuning (SFT). This repository provides the modular pipeline for data preparation, tokenizer training, pretraining, SFT, evaluation, and Hugging Face export.

🤗 **Model:** [augustoafleal/gpt2-ptbr-218m](https://huggingface.co/augustoafleal/gpt2-ptbr-218m)

Training path: **Portuguese Wikipedia → SFT Alpaca PT-BR → response-only SFT Canarim-Instruct-PTBR**.

## Repository structure

Only files and directories tracked by Git are listed below.

```text
llm-project/
├── ingest/
│   ├── __init__.py
│   ├── config.py
│   ├── db.py
│   └── ingest.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataloader.py
│   │   └── sft_dataset.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── generate.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── gpt.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sft_trainer.py
│   │   └── trainer.py
│   └── utils/
│       └── __init__.py
├── scripts/
│   ├── aggregate_judge_results.py
│   ├── audit_benchmark.py
│   ├── build_judge_inputs.py
│   ├── chat_hf.py
│   ├── download_sft_dataset.py
│   ├── download_wiki.py
│   ├── export_extracted_to_training_data.py
│   ├── export_huggingface.py
│   ├── export_training_data.py
│   ├── extract_wiki.py
│   ├── find_best_question.py
│   ├── generate_text.py
│   ├── plot_metrics.py
│   ├── plot_paper_metrics.py
│   ├── prepare_sft_dataset.py
│   ├── prepare_sft_response_only.py
│   ├── run_inference_suite.sh
│   ├── run_local_generation_benchmark.py
│   ├── tokenize_dataset.py
│   ├── train_gpt.py
│   ├── train_sft.py
│   ├── train_tokenizer.py
│   ├── validate_tokenized.py
│   └── validate_tokenizer.py
├── sql/
│   └── init.sql
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Generated artifacts

The directories below are **not tracked by Git**. They are created locally by the pipeline scripts.

| Directory | Purpose | Created by |
|---|---|---|
| `data/` | Dumps, extracted articles, training corpora, tokenized binaries, SFT datasets | `download_wiki.py`, `extract_wiki.py`, `export_training_data.py`, `tokenize_dataset.py`, `prepare_sft_dataset.py` |
| `runs/` | Training checkpoints (`best.pt`, `last.pt`), metric CSVs, configurations, inference outputs | `train_gpt.py`, `train_sft.py`, `run_inference_suite.sh` |
| `exports/` | Model exports in Hugging Face format (`config.json`, `model.safetensors`, tokenizer files) | `export_huggingface.py` |
| `benchmark/` | Model outputs, judge inputs/results, leaderboard, and benchmark audit reports | `run_local_generation_benchmark.py`, `build_judge_inputs.py`, `aggregate_judge_results.py`, `audit_benchmark.py` |
| `artifacts/` | Trained SentencePiece tokenizer model and vocabulary | `train_tokenizer.py` |

All scripts create their required directories automatically — manually running `mkdir` is not necessary.

## Prerequisites

- Docker and Docker Compose
- Python 3.10+ (note: `wikiextractor` may fail on Python 3.11+ with a regex error)
- Python dependencies:

```bash
pip install -r requirements.txt
```

Then install PyTorch separately according to your hardware:

```bash
# CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

> PyTorch is not included in `requirements.txt` because the installation URL varies according to the hardware (CPU vs CUDA).

## Configuration

1. Create the environment file:

```bash
cp .env.example .env
```

2. Adjust the credentials if necessary.

## Start PostgreSQL (Docker)

The download and extraction run in Python and do not require Docker. PostgreSQL is only required for the ingestion step.

Start the database from the project root:

```bash
docker compose up -d
```

The `wiki_articles` table will be created automatically using the [`sql/init.sql`](sql/init.sql) script.

## 1. Download the Wikipedia dump

By default, the script downloads the latest Portuguese Wikipedia dump (`ptwiki`) and saves it to `data/raw/wiki_dump.xml.bz2`.

```bash
python scripts/download_wiki.py
```

For another language/project, pass the desired code:

```bash
python scripts/download_wiki.py enwiki
```

In Docker-first mode, the command is the same, but runs inside the `app` service:

```bash
docker compose --profile app run --rm app python scripts/download_wiki.py enwiki
```

## 2. Extract the dump with WikiExtractor

By default, the script generates JSON output in `data/extracted/`.

```bash
python scripts/extract_wiki.py
```

Command executed internally:

```bash
python -m wikiextractor.WikiExtractor \
    --json \
    --no-templates \
    -o data/extracted \
    data/raw/wiki_dump.xml.bz2
```

## 3. Ingest the articles into PostgreSQL

Run ingestion from the project root:

```bash
python ingest/ingest.py
```

The pipeline:

- reads JSON files from the extracted directory defined in `WIKI_EXTRACTED_DIR`
- extracts `id`, `title`, and `text`
- calculates `length`
- ignores texts shorter than 200 characters
- performs batch inserts
- ignores duplicates with `ON CONFLICT (id) DO NOTHING`
- commits per batch
- safely handles `rollback()` for invalid connections
- automatically attempts to reconnect on a connection error with a single batch retry
- continues execution even if a batch fails

## 4. Export a general-purpose training dataset

Generates a continuous text file (`data/training/dataset_general.txt`) ready for BPE/SentencePiece tokenization and GPT-like pretraining.

```bash
python scripts/export_training_data.py
```

The script:

- connects to PostgreSQL (reuses `ingest/db.py`)
- runs `SELECT title, text FROM wiki_articles WHERE length > 200 ORDER BY id`
- exports in streaming mode with `fetchmany(1000)` — without loading everything into memory
- applies light cleaning: removes line breaks and normalizes whitespace
- formats each article as `Title\nText\n\n<eos>\n`
- generates `data/training/dataset_general.txt` in UTF-8

Example of the generated format:

```text
Astronomia
Astronomia é uma ciência natural que estuda corpos celestes...

<eos>
Brasil
Brasil é um país localizado na América do Sul...

<eos>
```

## 4a. Export dataset directly (PostgreSQL-free alternative)

Alternative route that generates the same output format as `export_training_data.py`, but reads
directly from the WikiExtractor JSON lines, **without going through PostgreSQL**.

This flow branches from the main pipeline after extraction:

```text
WikiExtractor                      WikiExtractor
    ↓                                    ↓
data/extracted/             vs.   data/extracted/
    ↓                                    ↓
ingest.py (DB)                          export_extracted_to_training_data.py
    ↓                                    ↓
PostgreSQL                              dataset_general_full.txt
    ↓
export_training_data.py
    ↓
dataset_general.txt
```

```bash
python scripts/export_extracted_to_training_data.py \
    --input-dir data/extracted \
    --output-file data/training/dataset_general_full.txt
```

The script:

- recursively traverses the directory provided in `--input-dir`, looking for `wiki_*` files
- processes the JSON lines one line at a time (WikiExtractor format)
- reuses the same filtering and normalization functions from the traditional pipeline:
  - `normalize_article()` from `ingest/ingest.py` — validates `id`, `title`, `length >= MIN_TEXT_LENGTH`
  - `clean_field()` from `scripts/export_training_data.py` — removes internal line breaks and normalizes whitespace
- deduplicates articles by `id` in memory (`set()`)
- sorts by `id` (deterministic, equivalent to PostgreSQL's `ORDER BY id`)
- writes in the format `{title}\n{text}\n\n<eos>\n` — **exactly** the same format as the DB pipeline
- generates metadata in `<output>.metadata.json`

Example of the generated metadata:

```json
{
  "input_dir": "data/extracted",
  "output_file": "data/training/dataset_general_full.txt",
  "total_files": 2906,
  "total_articles_seen": 896342,
  "total_articles_written": 896342,
  "duplicates": 0,
  "skipped_empty": 0,
  "skipped_short": 0,
  "total_characters": 2382480029,
  "min_length": 200,
  "elapsed_seconds": 180.5
}
```

| Argument | Default | Description |
|---|---|---|
| `--input-dir` | `data/extracted` | Directory with WikiExtractor `wiki_*` files |
| `--output-file` | `data/training/dataset_general_full.txt` | Output file in `TITLE\nTEXT\n\n<eos>\n` format |
| `--min-length` | `MIN_TEXT_LENGTH` from `.env` (or `200`) | Minimum text length |

### Differences from the DB pipeline

| Feature | DB pipeline | Direct pipeline |
|---|---|---|
| Dependency | PostgreSQL running | None |
| Intermediate storage | PostgreSQL table (~6-7 GB for the complete corpus) | Only the final TXT file |
| Deduplication | `ON CONFLICT (id) DO NOTHING` in the database | `set()` in memory |
| Sorting | `ORDER BY id` in SQL | `sorted()` in Python |
| Streaming | Cursor streaming (`fetchmany`) | **Accumulates in memory** for sorting |
| Fault tolerance | Automatic reconnection, retry per batch | Fails on file read error |

### Known limitation

Due to sorting by `id`, the script loads all articles into memory before writing.
For the complete corpus (896,342 articles, ~2.38 GB of text), peak memory usage is
approximately **2.5 GB** (text + Python overhead). Machines with less than 6 GB of RAM may be unable to perform the sorting without swapping.

### Compatibility

The generated file (`dataset_general_full.txt`) is format-compatible with the downstream pipeline:

- `scripts/tokenize_dataset.py` — consumes the same format and generates `train.bin`/`val.bin`, but reads `data/training/dataset_general.txt` by default. Rename this file or update `DATASET_PATH` before tokenization.
- `scripts/train_gpt.py` — trains the model

## 5. Train the BPE tokenizer

Trains a BPE tokenizer (SentencePiece) with the exported general-purpose dataset.

```bash
python scripts/train_tokenizer.py
```

The script:

- reads `data/training/dataset_general.txt`
- trains a BPE tokenizer with `vocab_size=16000`
- defines special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`
- generates `artifacts/tokenizer/tokenizer.model` and `artifacts/tokenizer/tokenizer.vocab`

Expected output:

```text
Starting tokenizer training (bpe)
Dataset: /path/to/data/training/dataset_general.txt
Vocab size: 16000
Artifacts: /path/to/artifacts/tokenizer

Tokenizer trained successfully!
Model: /path/to/artifacts/tokenizer/tokenizer.model
Vocabulary: /path/to/artifacts/tokenizer/tokenizer.vocab
```

Generated files:

| File | Description |
|---|---|
| `artifacts/tokenizer/tokenizer.model` | SentencePiece binary model |
| `artifacts/tokenizer/tokenizer.vocab` | Readable BPE vocabulary |

## 6. Validate the tokenizer

Validates tokenization with real words extracted from the corpus itself.

```bash
python scripts/validate_tokenizer.py
```

The script:

- scans `data/training/dataset_general.txt` and collects titles and frequent terms
- shows how many tokens each word produces and its fragmentation (subwords)
- displays the special tokens and their IDs

Expected output:

```text
Word                         Tokens Pieces
----------------------------------------------------------------------
Elvas                           2  ▁El + vas
São                             1  ▁São
Condado                         1  ▁Condado
moeda                           1  ▁moeda
Casa                            1  ▁Casa
Reforma                         1  ▁Reforma
...

Special tokens:
  <eos>        -> id=2, encode=[13501, 2]
  <bos>        -> id=1, encode=[13501, 1]
  <pad>        -> id=3, encode=[13501, 3]
  <unk>        -> id=0, encode=[13501, 0, 12668, 0]
```

## 7. Tokenize the dataset into binaries

Converts the text corpus into binary token IDs (`.bin`) for consumption by the DataLoader and GPT-like training.

```bash
python scripts/tokenize_dataset.py
```

The script:

- loads `artifacts/tokenizer/tokenizer.model` (SentencePiece BPE)
- reads `data/training/dataset_general.txt` in streaming mode (line by line)
- detects articles separated by textual `<eos>`
- tokenizes each article with `sp.encode()` and manually inserts `EOS_ID` (2) at the end of each one
- deterministically splits: first 90% of articles → train, 10% → val
- saves as contiguous `uint16` arrays (`vocab_size=16000` fits perfectly)
- generates metadata with token counts and configuration

Expected output:

```text
Tokenizer: /path/to/artifacts/tokenizer/tokenizer.model
EOS ID:    2
Dataset:   /path/to/data/training/dataset_general.txt

Counting articles...
Total articles: 81050
Train articles: 72945
Val articles:   8105

Tokenizing...
  processed 50000/81050 articles

Total tokens:      105617933
Train tokens:       98729599
Val tokens:          6888334

Saving binary files...
Train:    data/tokenized/train.bin
Val:      data/tokenized/val.bin
Metadata: data/tokenized/metadata.json
```

Generated files:

| File | Description |
|---|---|
| `data/tokenized/train.bin` | ~189 MB — training tokens in `uint16` |
| `data/tokenized/val.bin` | ~13 MB — validation tokens in `uint16` |
| `data/tokenized/metadata.json` | Metadata: `vocab_size`, `dtype`, `train_tokens`, `val_tokens`, `eos_id`, `tokenizer_path` |

### Validate tokenization

After generating the binaries, validate the integrity of the files:

```bash
python scripts/validate_tokenized.py
```

The script checks:

- consistency of `metadata.json` with the binaries
- token ID range (all < `vocab_size`)
- count and correct position of EOS markers
- encode/decode roundtrip on samples
- readable decode of the first and last article
- format of the stream `[tokens][EOS][tokens][EOS]...`

## 8. Train a GPT-like model

Trains a decoder-only Transformer (GPT-2 style) with the tokenized binaries.

```bash
python scripts/train_gpt.py --device cpu
```

For GPU (if available):

```bash
python scripts/train_gpt.py --device cuda
```

The script:

- loads `data/tokenized/train.bin` and `val.bin` via `numpy.memmap` (without loading everything into RAM)
- uses the default lightweight configuration: `vocab_size=16000`, `block_size=256`, `n_embd=384`, `n_head=6`, `n_layer=6`
- uses AdamW with warmup + cosine learning rate decay
- calculates validation loss and perplexity at regular intervals
- saves checkpoints in `runs/<timestamp>/`

### Published model architecture

The published `gpt2-ptbr-218m` model has **218,040,320 parameters** and the following configuration:

| Setting | Value |
|---|---:|
| Vocabulary size | 16,000 |
| Context length | 256 |
| Embedding dimension | 1,024 |
| Attention heads | 16 |
| Transformer layers | 16 |

Its architecture (`src/model/gpt.py`) is:

| Component | Description |
|---|---|
| `token_embedding` | Embedding lookup 16000 → 1024 (shared with `lm_head`) |
| `position_embedding` | Learned positional embedding 256 → 1024 |
| `CausalSelfAttention` | Multi-head attention (16 heads) with causal mask |
| `FeedForward` | MLP 1024 → 4096 → 1024 with GELU |
| `Block` | LayerNorm → Attention → residual → LayerNorm → FFN → residual |
| `lm_head` | Projection 1024 → 16000 (shared weights) |

To train with the published-model dimensions, pass the architecture flags explicitly:

```bash
python scripts/train_gpt.py \
    --n-embd 1024 \
    --n-head 16 \
    --n-layer 16 \
    --block-size 256 \
    --device cuda
```

### Training script defaults

The `train_gpt.py` defaults remain a smaller configuration (6 layers, 6 heads, embedding dimension 384, and ~16.9M parameters), suitable for lightweight experiments and smoke tests.

Expected output with the default configuration:

```text
Training on cpu
Parameters: 16,869,120
Run dir: /path/to/runs/20260528_204405

step      1 | loss 9.7819 | tok/s 1090 | lr 3.00e-06
  └─ val loss 9.7253 | perplexity 16735.00

step   1000 | loss 6.2341 | tok/s 1150 | lr 2.84e-04
  └─ val loss 6.5102 | perplexity 672.34
...

Training complete. Best val loss: 5.2341
Checkpoints saved to /path/to/runs/20260528_204405
```

Files generated in `runs/<timestamp>/`:

| File | Description |
|---|---|
| `best.pt` | Checkpoint with the lowest val loss (model + optimizer) |
| `last.pt` | Checkpoint from the last step |
| `config.json` | Training hyperparameters |
| `train_metrics.csv` | Loss and tokens/sec per step |
| `eval_metrics.csv` | Val loss and perplexity per evaluation |

### Resume interrupted training

If training is interrupted (power loss, OOM, Ctrl+C), it is possible to resume from the last checkpoint without losing progress:

```bash
python scripts/train_gpt.py \
  --resume runs/20260603_220422/last.pt \
  --device cpu \
  --max-iters 100000
```

The checkpoint restores the model weights, optimizer state, current step, and best val loss. Training continues from the next step and the learning rate scheduler (warmup + cosine decay) proceeds exactly from where it stopped. `--max-iters` defines the new final step — useful for extending training beyond the initially planned duration.

### Script arguments

| Argument | Default | Description |
|---|---|---|
| `--device` | `cpu` | `cpu` or `cuda` |
| `--batch-size` | `32` | Batch size |
| `--block-size` | `256` | Context size |
| `--lr` | `3e-4` | Learning rate |
| `--max-iters` | `10000` | Training iterations (defines the final step when resuming) |
| `--eval-interval` | `500` | Interval between evaluations |
| `--eval-iters` | `100` | Iterations for averaging val loss |
| `--n-embd` | `384` | Embedding dimension |
| `--n-head` | `6` | Number of attention heads |
| `--n-layer` | `6` | Number of Transformer layers |
| `--dropout` | `0.1` | Dropout rate |
| `--resume` | — | Path to `last.pt` from previous training. Restores model, optimizer, step, and best val loss. `--max-iters` defines the new final step. |

## 9. Generate text with the trained model

After training, generate text autoregressively with a saved checkpoint.

```bash
python scripts/generate_text.py \
    --checkpoint runs/<timestamp>/best.pt \
    --prompt "Astronomia" \
    --max-new-tokens 200 \
    --temperature 0.8 \
    --top-k 40 \
    --stop-at-eos
```

The script:

- loads the checkpoint and the saved `model_config`
- loads the SentencePiece tokenizer
- encodes the prompt with the tokenizer
- generates one token at a time using sampling with `temperature` and `top_k`
- decodes the generated tokens into text
- optionally stops generation upon finding the `<eos>` token (`--stop-at-eos`)

### Script arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | (required) | Path to the checkpoint `.pt` |
| `--prompt` | `""` | Initial text for generation |
| `--max-new-tokens` | `200` | Maximum number of tokens to generate |
| `--temperature` | `0.8` | Sampling temperature |
| `--top-k` | `40` | Top-k sampling |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--stop-at-eos` | `false` | Stops generation when emitting `<eos>` |

## 10. Run batch inference

Generates text for multiple predefined prompts from a saved checkpoint, automating `generate_text.py` in batch.

```bash
./scripts/run_inference_suite.sh <run_id>
```

Example:

```bash
./scripts/run_inference_suite.sh 20260531_232031
```

The script:

- uses the checkpoint `runs/<run_id>/best.pt`
- runs `generate_text.py` for 8 fixed prompts (astronomia, Brasil, IA, história, etc.)
- saves each output to `runs/<run_id>/inference/<name>.txt`
- generates `runs/<run_id>/inference/metadata.json` with the produced files

## 11. Visualize training metrics

Generates loss and perplexity charts from the CSVs saved during training.

```bash
python scripts/plot_metrics.py <run_id>
```

Example:

```bash
python scripts/plot_metrics.py 20260531_232031
```

The script:

- reads `runs/<run_id>/train_metrics.csv` and `runs/<run_id>/eval_metrics.csv`
- generates `train_loss.png` (blue `step × loss` curve) and `val_perplexity.png` (orange `step × perplexity` curve)
- saves the PNGs inside the run directory itself

| Argument | Description |
|---|---|
| `--no-grid` | Removes the grid from the charts (enabled by default) |

## 12. Download SFT (Supervised Fine-Tuning) dataset

Downloads the `dominguesm/alpaca-data-pt-br` dataset from Hugging Face for fine-tuning preparation.

```bash
python scripts/download_sft_dataset.py
```

The script:

- downloads the dataset with `datasets.load_dataset`
- saves it to `data/sft/alpaca_ptbr/raw/alpaca_data_ptbr.json`
- preserves the `instruction`, `input`, and `output` fields
- is idempotent — if the file already exists, it does not download it again

| Argument | Default | Description |
|---|---|---|
| `--dataset-name` | `dominguesm/alpaca-data-pt-br` | Dataset name on Hugging Face |
| `--output-path` | `data/sft/alpaca_ptbr/raw/alpaca_data_ptbr.json` | Output path |

## 13. Prepare the SFT dataset for training

Normalizes, formats, splits into train/val, tokenizes with SentencePiece, and generates uint16 bins for causal LM SFT.

```bash
python scripts/prepare_sft_dataset.py
```
> Use --max-examples 1000 for quick tests

The script:

- loads the JSON downloaded to `data/sft/alpaca_ptbr/raw/alpaca_data_ptbr.json`
- normalizes examples: validates `instruction` and `output`, discards invalid examples, and normalizes whitespace
- shuffles deterministically with a fixed seed
- formats each example using the instruction/response pattern with `eos_id` manually appended:

```text
### Instrução:
{instruction}

### Resposta:
{output}
<eos>
```

When `input` is not empty:

```text
### Instrução:
{instruction}

### Entrada:
{input}

### Resposta:
{output}
<eos>
```

- splits into train/val (90/10 by default)
- tokenizes with SentencePiece (`artifacts/tokenizer/tokenizer.model`)
- saves `sft_train.txt`, `sft_val.txt`, `train.bin`, `val.bin` (uint16), and `metadata.json`

| Argument | Default | Description |
|---|---|---|
| `--input-path` | `data/sft/alpaca_ptbr/raw/alpaca_data_ptbr.json` | Input JSON |
| `--output-dir` | `data/sft/alpaca_ptbr/processed` | Output directory |
| `--tokenizer-path` | `artifacts/tokenizer/tokenizer.model` | SentencePiece tokenizer |
| `--val-ratio` | `0.1` | Validation proportion |
| `--seed` | `42` | Shuffle seed |
| `--max-examples` | (all) | Limit the number of examples |

Files generated in `data/sft/alpaca_ptbr/processed/`:

| File | Description |
|---|---|
| `sft_train.txt` | Formatted training examples |
| `sft_val.txt` | Formatted validation examples |
| `train.bin` | Training tokens in `uint16` |
| `val.bin` | Validation tokens in `uint16` |
| `metadata.json` | Metadata: `vocab_size`, `dtype`, `eos_id`, `train_tokens`, `val_tokens`, etc. |

### 13.1 Prepare the SFT Response-Only dataset

Variant of the dataset above that generates loss masks to train only the response tokens.

```bash
python scripts/prepare_sft_response_only.py
```

Differences compared to `prepare_sft_dataset.py`:

- generates `train_loss_mask.bin` and `val_loss_mask.bin` (`uint8`: 0 = ignore loss, 1 = calculate loss)
- metadata includes `training_format: "response_only"`, `loss_tokens_train/val`, and `response_tokens_train/val`
- instruction/input tokens are masked (loss = 0) — only the response and `<eos>` contribute to the loss

| Argument | Default | Description |
|---|---|---|
| `--input-path` | `data/sft/alpaca_ptbr/raw/alpaca_data_ptbr.json` | Input JSON |
| `--output-dir` | `data/sft/alpaca_ptbr/processed_response_only` | Output directory |
| `--tokenizer-path` | `artifacts/tokenizer/tokenizer.model` | SentencePiece tokenizer |
| `--val-ratio` | `0.1` | Validation proportion |
| `--seed` | `42` | Shuffle seed |
| `--max-examples` | (all) | Limit the number of examples |

Files generated in `data/sft/alpaca_ptbr/processed_response_only/`:

| File | Description |
|---|---|
| `train.bin` | Training tokens in `uint16` |
| `val.bin` | Validation tokens in `uint16` |
| `train_loss_mask.bin` | Loss mask (training) in `uint8` |
| `val_loss_mask.bin` | Loss mask (validation) in `uint8` |
| `metadata.json` | Metadata with `training_format: "response_only"` |

### 13.2 Prepare the Canarim response-only SFT stage

The released model's second SFT stage uses [Canarim-Instruct-PTBR-Dataset](https://huggingface.co/datasets/dominguesm/Canarim-Instruct-PTBR-Dataset), a Portuguese instruction dataset with `instruction`, `input`, and `output` fields. Download, prepare it with response-only loss masks, then fine-tune from the Alpaca SFT checkpoint:

```bash
python scripts/download_sft_dataset.py \
    --dataset-name dominguesm/Canarim-Instruct-PTBR-Dataset \
    --output-path data/sft/canarim_ptbr/raw/canarim_instruct_ptbr.json

python scripts/prepare_sft_response_only.py \
    --input-path data/sft/canarim_ptbr/raw/canarim_instruct_ptbr.json \
    --output-dir data/sft/canarim_ptbr/processed_response_only

python scripts/train_sft.py \
    --pretrained-run-id sft_<alpaca_run_id> \
    --data-dir data/sft/canarim_ptbr/processed_response_only \
    --response-only \
    --device cuda
```

This produces the final stage of the training path: the loss is computed only over the response and `<eos>` tokens. The preparation and training scripts currently use Alpaca PT-BR labels in their generated metadata. This does not change the Canarim data used for training.

## 14. Train SFT (Supervised Fine-Tuning)

Fine-tunes a pretrained model on an instruction dataset.
Supports two modes:

- **full_loss** (default): loss calculated over all tokens
- **response_only**: loss calculated only over response tokens (instruction/input ignored)

```bash
# Full-loss (default)
python scripts/train_sft.py --pretrained-run-id <run_id> --device cuda

# Response-only
python scripts/train_sft.py --pretrained-run-id <run_id> --response-only --device cuda
```

Example with smoke test:

```bash
python scripts/train_sft.py \
  --pretrained-run-id 20260531_232031 \
  --max-iters 100 \
  --batch-size 2 \
  --device cpu
```

Example with real training (GPU):

```bash
python scripts/train_sft.py \
  --pretrained-run-id 20260531_232031 \
  --batch-size 16 \
  --block-size 256 \
  --max-iters 1000 \
  --eval-interval 50 \
  --eval-iters 20 \
  --lr 5e-5 \
  --min-lr 5e-6 \
  --device cuda
```

The script:

- loads the checkpoint from `runs/<pretrained_run_id>/best.pt`
- reconstructs the GPT model with the same configuration as the base checkpoint
- keeps **all trainable parameters** (full fine-tuning, without LoRA)
- loads `train.bin` and `val.bin` from the specified directory
- with `--response-only`: loads `processed_response_only/` and uses masks to ignore instruction/input
- uses `get_batch` with `np.memmap` (streaming, without loading everything into RAM)
- uses a lower LR than pretraining (default `5e-5`) with warmup + cosine decay
- saves everything in `runs/sft_<timestamp>/`

| Argument | Default | Description |
|---|---|---|
| `--pretrained-run-id` | (required) | Run ID of the pretrained model |
| `--checkpoint-name` | `best.pt` | Checkpoint name in the base run |
| `--data-dir` | `processed` or `processed_response_only` | SFT dataset directory (auto: `processed` without the flag, `processed_response_only` with `--response-only`) |
| `--batch-size` | `16` | Batch size |
| `--block-size` | `256` | Context size |
| `--max-iters` | `1000` | Training iterations |
| `--eval-interval` | `50` | Interval between evaluations |
| `--eval-iters` | `20` | Iterations for averaging val loss |
| `--lr` | `5e-5` | Learning rate |
| `--min-lr` | `5e-6` | Minimum learning rate |
| `--warmup-iters` | `100` | Warmup iterations |
| `--lr-decay-iters` | `max_iters` | Iterations over which to decay the LR |
| `--weight-decay` | `0.1` | Weight decay |
| `--grad-clip` | `1.0` | Gradient clipping |
| `--device` | auto | `cpu` or `cuda` |
| `--seed` | `42` | Random seed |
| `--response-only` | `false` | Trains only response tokens (ignores instruction/input in the loss) |

Files generated in `runs/sft_<timestamp>/`:

| File | Description |
|---|---|
| `best.pt` | Checkpoint with the lowest val loss |
| `last.pt` | Checkpoint from the last step |
| `config.json` | Complete SFT training configuration |
| `run_metadata.json` | Lineage: base checkpoint, dataset, training type |
| `train_metrics.csv` | Loss and tokens/sec per step |
| `eval_metrics.csv` | Val loss and perplexity per evaluation |

## 15. Export checkpoint to Hugging Face

Exports a trained checkpoint (pretraining or SFT) to Hugging Face format,
generating `config.json`, `model.safetensors`, tokenizer files (`tokenizer.model`, `tokenizer_config.json`,
`special_tokens_map.json`), `generation_config.json`, and `README.md` — all ready for upload
to the Hub or use with `AutoModelForCausalLM`.

```bash
# Export pretraining checkpoint
python scripts/export_huggingface.py \
    --checkpoint runs/<run_id>/best.pt \
    --output exports/huggingface/pretrained_<run_id>

# Export SFT checkpoint
python scripts/export_huggingface.py \
    --checkpoint runs/sft_<run_id>/best.pt \
    --output exports/huggingface/sft_<run_id>
```

To validate the export (loads the exported model with `AutoModelForCausalLM` and compares
the tokenizer encoding with the original SentencePiece):

```bash
python scripts/export_huggingface.py \
    --checkpoint runs/sft_<run_id>/best.pt \
    --output exports/huggingface/sft_<run_id> \
    --validate
```

| Flag | Description |
|---|---|
| `--checkpoint` | Path to the `.pt` or run ID (e.g. `sft_20260610_172000`) |
| `--output` | Output directory for artifacts |
| `--overwrite` | Overwrites existing output directory |
| `--validate` | Post-export: loads model + tokenizer and runs inference |
| `--verbose` | Detailed log per tensor |

Files generated in `exports/huggingface/<run_id>/`:

| File | Description |
|---|---|
| `config.json` | Model configuration (`GPT2Config`) |
| `model.safetensors` | Exported weights in safe format |
| `tokenizer.model` | Original SentencePiece model |
| `tokenizer_config.json` | Tokenizer configuration |
| `special_tokens_map.json` | Special token mapping |
| `tokenizer.json` | Complete tokenizer (vocab + merges) |
| `generation_config.json` | Default generation configuration |
| `README.md` | Model card with pipeline, datasets, and citation |

The exported model can be loaded directly with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("exports/huggingface/sft_<run_id>")
tokenizer = AutoTokenizer.from_pretrained("exports/huggingface/sft_<run_id>")
```

## 16. Interactive chat with a Hugging Face model

After exporting the checkpoint to Hugging Face format, it is possible to chat
interactively with the model directly in the terminal.

```bash
# Local mode (loads the model with transformers)
python scripts/chat_hf.py --model exports/huggingface/sft_<run_id>

# API mode (uses Hugging Face InferenceClient)
python scripts/chat_hf.py --backend api --model augustoafleal/gpt2-ptbr-218m

# Single turn (non-interactive) — useful for testing
python scripts/chat_hf.py --model exports/huggingface/sft_<run_id> \
    --max-new-tokens 120 --temperature 0.7 <<< "O que é Python?"
```

The script offers two backends:

| Backend | Description |
|---|---|
| `local` (default) | Loads the model with `transformers.AutoModelForCausalLM` on the local device |
| `api` | Uses `huggingface_hub.InferenceClient` — no GPU required, only an HF token |

In **local** mode, the model is loaded into RAM/VRAM and responses are generated
locally. In **api** mode, requests are sent to Hugging Face infrastructure (free, but limited).

The instruction template follows the same SFT pattern:

```text
### Instrução:
{user_input}

### Resposta:
```

To disable the template (useful for base models without fine-tuning):

```bash
python scripts/chat_hf.py --model exports/huggingface/pretrained_<run_id> \
    --no-instruction-template
```

Special commands during interactive chat:

| Command | Action |
|---|---|
| `exit`, `quit`, `sair` | Ends the session |
| `Ctrl+C` / `Ctrl+D` | Ends the session |

### Script arguments

| Argument | Default | Description |
|---|---|---|
| `--backend` | `local` | `local` (transformers) or `api` (InferenceClient) |
| `--model` | (required) | Local path or Hub ID (e.g. `exports/huggingface/...` or `augustoafleal/gpt2-ptbr-218m`) |
| `--instruction-template` | `True` | Formats input as an instruction using `### Instrução:\n...\n\n### Resposta:\n` |
| `--max-new-tokens` | `80` | Maximum number of tokens to generate per response |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-k` | `40` | Top-k sampling |
| `--top-p` | `1.0` | Top-p (nucleus) sampling |
| `--do-sample` | `True` | Use stochastic sampling (vs. greedy) |
| `--repetition-penalty` | `1.1` | Token repetition penalty |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` (local backend only) |
| `--dtype` | `auto` | `auto`, `float32`, `float16`, or `bfloat16` (local backend only) |

The Hugging Face token is resolved automatically in the following order:

1. `HF_TOKEN` environment variable
2. `HUGGINGFACE_HUB_TOKEN` environment variable
3. Token stored by `huggingface-cli login` (`huggingface_hub.get_token()`)

## 17. Run an automated generation benchmark

Runs 20 fixed questions in 2 generation modes (normal and creative) and saves all answers in structured JSON with fields for manual evaluation.

```bash
# With the full checkpoint path
python scripts/run_local_generation_benchmark.py runs/sft_20260607_230617/best.pt

# Or just with the run ID (automatically resolves runs/<id>/best.pt)
python scripts/run_local_generation_benchmark.py sft_20260607_230617

# With custom output
python scripts/run_local_generation_benchmark.py sft_20260607_230617 benchmark.json
```

File generated in `runs/<run_id>/benchmark_<timestamp>.json`:

| Field | Description |
|---|---|
| `checkpoint` | Checkpoint used |
| `modes.normal` | Parameters for normal mode (temp=0.3, top_k=20) |
| `modes.creative` | Parameters for creative mode (temp=0.7, top_k=40) |
| `results[].normal.raw_output` | Complete generation stdout |
| `results[].normal.answer` | Extracted answer (after `### Resposta:`) |
| `results[].manual_eval` | Fields for manual evaluation (score 0-2, notes) |

Example of generated JSON:

```json
{
  "checkpoint": "runs/sft_20260607_230617/best.pt",
  "num_questions": 20,
  "modes": { "normal": { "temperature": 0.3, "top_k": 20 }, "creative": { ... } },
  "results": [
    {
      "id": 1,
      "question": "O que é inteligência artificial?",
      "normal": { "raw_output": "...", "answer": "...", "exit_code": 0 },
      "creative": { "raw_output": "...", "answer": "...", "exit_code": 0 },
      "manual_eval": {
        "normal_score": null,
        "creative_score": null,
        "notes": "",
        "repetition": null,
        "format_followed": null,
        "factual_error": null
      }
    }
  ]
}
```

## 18. Build inputs for evaluation by LLM judges

After generating the benchmark outputs, prepare anonymized inputs for blind evaluation by
LLM judges (GPT, Gemini, Claude). The script shuffles the models for each question using
random aliases (A/B/C/D) and generates JSON + Markdown files ready for the judge.

```bash
python scripts/build_judge_inputs.py
```

The script:

- reads all `benchmark/model_outputs/benchmark_run_*.json`
- validates consistency: same number of questions and same text across models
- extracts the `answer` (or `response`, `output`, `generated_text`) field from each result
- builds a blind mapping `{A, B, C, ...} → model_id` per question (seed=42, deterministic shuffle)
- generates JSON files per question in `benchmark/judge_inputs/`
- generates Markdown files per question in `benchmark/judge_inputs_md/` with an embedded evaluation prompt
- saves the mapping in `benchmark/judge_mapping.json`

| Argument | Default | Description |
|---|---|---|
| `--input-dir` | `benchmark/model_outputs` | Directory with `benchmark_run_*.json` |
| `--output-dir` | `benchmark/judge_inputs` | JSON output per question |
| `--output-dir-md` | `benchmark/judge_inputs_md` | Markdown output per question |
| `--mapping` | `benchmark/judge_mapping.json` | Path to the alias → model mapping |

The evaluation prompt included in the Markdown files defines the criteria:

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Correctness | incorrect | partially correct | correct |
| Instruction Following | not followed | partially followed | completely followed |
| Factuality | significant factual errors | mix of correct/incorrect | correct facts |
| Conciseness | inadequate | acceptable | objective |
| Repetition | excessive repetition | some repetition | no relevant repetition |
| Overall | poor | acceptable | excellent |

Each Markdown file (`benchmark/judge_inputs_md/question_001.md`) contains:

```
# Pergunta 1

## Instrução
{pergunta}

---

## Modelo A
{resposta do modelo A}

---

## Modelo B
{resposta do modelo B}

...

## Avaliação
{prompt completo com formato JSON de saída}
```

Generated files:

| File | Description |
|---|---|
| `benchmark/judge_inputs/question_*.json` | JSON input per question (fields: question_id, question, answers) |
| `benchmark/judge_inputs_md/question_*.md` | Markdown input per question (ready to copy to the judge) |
| `benchmark/judge_mapping.json` | Mapping `{question: {alias: model_id}}` to decode results |

## 19. Aggregate judge results into a leaderboard

Processes the judge evaluation results (manually saved in
`benchmark/judge_results/`) and generates a consolidated leaderboard with metrics
per model.

```bash
python scripts/aggregate_judge_results.py
```

The script:

- reads `benchmark/judge_results/{gpt,gemini,claude}/question_*.json` from each judge
- loads `benchmark/judge_mapping.json` to decode aliases → model_id
- extracts normalized scores (correctness, instruction_following, factuality, conciseness, repetition, overall)
- extracts ranking, winner, and loser for each evaluation
- validates consistency: `n_scores == n_rankings == num_questions × num_judges`, `total_wins == total_losses`
- calculates per model: average overall, wins, losses, average_rank
- generates a leaderboard sorted by overall (desc) and wins (desc)

| Argument | Default | Description |
|---|---|---|
| `--judge-results-dir` | `benchmark/judge_results` | Directory with `gpt/`, `gemini/`, `claude/` subfolders |
| `--mapping` | `benchmark/judge_mapping.json` | Alias → model mapping |
| `--reports-dir` | `benchmark/reports` | Output directory for reports |

Files generated in `benchmark/reports/`:

| File | Description |
|---|---|
| `leaderboard.json` | Complete leaderboard in JSON with all metrics |
| `leaderboard.csv` | Leaderboard in CSV (rank, model, overall, wins, losses, average_rank) |
| `leaderboard.md` | Leaderboard in Markdown formatted as a table |
| `judge_summary.json` | Judge summary (model used, number of questions) |
| `question_level_results.json` | Results per question (winner/loser per judge) |

Example of the generated leaderboard:

```text
| Rank | Model               | Overall | Wins | Losses | Avg Rank |
| ---- | ------------------- | ------: | ---: | -----: | -------: |
| 1    | sft_20260610_172000 |    1.98 |   60 |      0 |     1.00 |
| 2    | sft_20260607_230617 |    1.87 |   56 |      4 |     1.07 |
| 3    | 20260603_224519     |    0.42 |    4 |     56 |     3.55 |
| 4    | 20260531_232031     |    0.18 |    0 |     60 |     4.00 |
```

Expected console output:

```text
Validation checks:
  All checks passed (expected=60)

Total evaluations processed: 60
Judges found:
  gpt: gpt-4o (20 questions)
  gemini: gemini-2.0-flash (20 questions)
  claude: claude-3-5-sonnet-20241022 (20 questions)

Top models by overall:
  1. sft_20260610_172000 — overall=1.98, wins=60, losses=0, avg_rank=1.0
  2. sft_20260607_230617 — overall=1.87, wins=56, losses=4, avg_rank=1.07
  3. 20260603_224519 — overall=0.42, wins=4, losses=56, avg_rank=3.55
  4. 20260531_232031 — overall=0.18, wins=0, losses=60, avg_rank=4.0
```

## 20. Audit benchmark results

Generates an audit report checking the consistency and coherence of
the judge results.

```bash
python scripts/audit_benchmark.py
```

The script:

- loads all results from `benchmark/judge_results/{gpt,gemini,claude}/`
- loads `benchmark/judge_mapping.json`
- generates `benchmark/reports/audit_report.md` with the following analyses:

| Section | Analysis |
|---|---|
| 1. Score Distribution | Count of 0/1/2 scores per model and metric, with average |
| 2. Winner × Overall | Checks whether the winner has the highest overall score |
| 3. Winner × Ranking[0] | Checks whether the winner is at the top of the ranking |
| 4. Loser × Ranking[-1] | Checks whether the loser is at the bottom of the ranking |
| 5. Average Rank | Confirms the average_rank calculation |
| 6. Distribution by Judge | Average and distribution of overall by judge |
| 7. Judge Severity | Comparison of the global average vs. each judge |
| 8. Correlation between Metrics | Pearson correlation between overall, average_rank, and wins |
| 9. Ranking Anomalies | Detects missing, extra, or duplicate aliases |
| 10. Final Conclusion | Summary of data integrity |

No command-line arguments are required.

Expected output:

```text
Audit report generated: benchmark/reports/audit_report.md

EXECUTIVE SUMMARY — Benchmark Audit

Key findings:
  1. Aggregator is CORRECT — all numerical validations pass
  2. Judges are severe but internally consistent
  3. Overall and average_rank produce the same ordering
  4. Detected ranking anomaly: Gemini Q10 has 'Delta' instead of 'D'
  5. SFT models clearly dominate the leaderboard
```

## 21. Generate figures for the paper

Generates publication-ready figures (PNG + PDF, 300 dpi) with improved visual finish,
using all completed pretraining and SFT runs.

```bash
python scripts/plot_paper_metrics.py <run_id1> [<run_id2> ...]
```

Example with all runs:

```bash
python scripts/plot_paper_metrics.py 20260603_224519 sft_20260607_230617 sft_20260610_172000 20260531_232031
```

Figures generated per run (in the `runs/<run_id>/` directory):

| File | Content |
|---|---|
| `paper_train_loss.png` / `.pdf` | Training loss with marker at the lowest value |
| `paper_val_loss.png` / `.pdf` | Validation loss with marker at the best checkpoint |
| `paper_val_perplexity.png` / `.pdf` | Validation perplexity with marker at the lowest value |

Comparison figure (in `runs/`):

| File | Content |
|---|---|
| `paper_perplexity_comparison.png` / `.pdf` | Bar chart comparing the best perplexity across runs |

## Environment variables

File [`.env.example`](.env.example):

- `POSTGRES_HOST`: PostgreSQL host
- `POSTGRES_PORT`: PostgreSQL port
- `POSTGRES_DB`: database name
- `POSTGRES_USER`: database user
- `POSTGRES_PASSWORD`: database password
- `BATCH_SIZE`: insertion batch size (safer default: `100`)
- `MIN_TEXT_LENGTH`: minimum text length accepted by the pipeline
- `WIKI_JSON_GLOB`: search pattern for extracted files
- `WIKI_EXTRACTED_DIR`: extracted directory (WikiExtractor output) used for ingestion

## Pipeline (local Python + Postgres in Docker)

This flow runs the Python scripts on your host and uses PostgreSQL via Docker Compose.

```bash
cp .env.example .env
docker compose up -d
pip install -r requirements.txt
python scripts/download_wiki.py
python scripts/extract_wiki.py
python ingest/ingest.py
python scripts/export_training_data.py
python scripts/train_tokenizer.py
python scripts/validate_tokenizer.py
python scripts/tokenize_dataset.py
python scripts/train_gpt.py --device cpu
python scripts/export_huggingface.py \
    --checkpoint runs/<run_id>/best.pt \
    --output exports/huggingface/pretrained_<run_id> \
    --validate                              # optional: export to Hugging Face
python scripts/generate_text.py --checkpoint runs/<timestamp>/best.pt --prompt "Astronomia"
python scripts/plot_metrics.py <run_id>          # optional: visualize metrics
./scripts/run_inference_suite.sh <run_id>         # optional: batch inference
```

## Alternative pipeline (without PostgreSQL)

Route that skips the database and generates the corpus directly from the WikiExtractor JSON lines.

```bash
cp .env.example .env
pip install -r requirements.txt
python scripts/download_wiki.py
python scripts/extract_wiki.py
python scripts/export_extracted_to_training_data.py \
    --input-dir data/extracted \
    --output-file data/training/dataset_general_full.txt
python scripts/tokenize_dataset.py
python scripts/train_gpt.py --device cpu
```

**Attention**: `tokenize_dataset.py` currently reads from `data/training/dataset_general.txt`.
To use the new corpus, manually point `DATASET_PATH` in the script to it or rename the file.

## SFT pipeline (post-pretraining)

After pretraining, prepare the dataset for supervised fine-tuning:

```bash
pip install -r requirements.txt
python scripts/download_sft_dataset.py
python scripts/prepare_sft_dataset.py
python scripts/prepare_sft_dataset.py --max-examples 1000
python scripts/prepare_sft_response_only.py
python scripts/train_sft.py --pretrained-run-id <run_id> --device cuda
python scripts/train_sft.py --pretrained-run-id <run_id> --response-only --device cuda
python scripts/export_huggingface.py \
    --checkpoint runs/sft_<run_id>/best.pt \
    --output exports/huggingface/sft_<run_id> \
    --validate                              # optional: export to Hugging Face
python scripts/chat_hf.py \
    --model exports/huggingface/sft_<run_id>    # optional: interactive chat
```

After SFT, evaluate the models with the automated benchmark pipeline:

```bash
# 1. Generate answers for 20 fixed questions (2 modes: normal and creative)
python scripts/run_local_generation_benchmark.py sft_<run_id>

# 2. Build anonymized inputs for judges
python scripts/build_judge_inputs.py

# 3. Aggregate judge results into a leaderboard
python scripts/aggregate_judge_results.py

# 4. Audit result consistency
python scripts/audit_benchmark.py
```

## Pipeline (Docker-first)

This flow runs *everything* (download, extraction, and ingestion) inside the `app` container, together with PostgreSQL in Compose.

```bash
cp .env.example .env
docker compose up -d postgres
docker compose --profile app run -d --name llm_download app python scripts/download_wiki.py
docker compose --profile app run -d --name llm_extract app python scripts/extract_wiki.py
docker compose --profile app run -d --name llm_ingest app python ingest/ingest.py
docker compose --profile app run --rm app python scripts/export_training_data.py
docker compose --profile app run --rm app python scripts/train_tokenizer.py
docker compose --profile app run --rm app python scripts/validate_tokenizer.py
docker compose --profile app run --rm app python scripts/tokenize_dataset.py
docker compose --profile app run --rm app python scripts/train_gpt.py --device cpu
docker compose --profile app run --rm app python scripts/download_sft_dataset.py
docker compose --profile app run --rm app python scripts/prepare_sft_dataset.py
docker compose --profile app run --rm app python scripts/train_sft.py --pretrained-run-id <run_id> --device cpu
docker compose --profile app run --rm app python scripts/export_huggingface.py \
    --checkpoint runs/sft_<run_id>/best.pt \
    --output exports/huggingface/sft_<run_id> \
    --validate                              # optional: export to Hugging Face
docker compose --profile app run --rm app python scripts/chat_hf.py \
    --model exports/huggingface/sft_<run_id>    # optional: interactive chat
```

Logs/state:

```bash
docker ps -a | rg "llm_(download|extract|ingest)"
docker logs -f llm_download
```

For another language/project:

```bash
docker compose --profile app run -d --name llm_download app python scripts/download_wiki.py enwiki
```

## Ingestion behavior

- `BATCH_SIZE=100` is the recommended default to avoid statements that are too large with `execute_values`.
- If PostgreSQL goes down during a batch, the script safely attempts to call `rollback()`.
- If the connection is broken, the script closes the old connection, opens a new one, and retries the same batch once.
- If the retry also fails, the batch is marked as failed and the pipeline continues to the next batch.
- Logs show processed batches, failed batches, reconnection, and a final ingestion summary.

Examples of expected logs:

```text
Using extracted data from /path/to/data/extracted
Found 128 files to ingest
Processed batch 1 from wiki_00: 100 rows inserted
Connection error on batch 4 from wiki_00: server closed the connection unexpectedly
Database connection re-established
Retried batch 4 from wiki_00: 100 rows inserted
Finished file /path/to/wiki_00: 1200 rows inserted across 12 batches
Ingestion summary: 50000 rows inserted, 520 batches processed, 2 batches failed
```

## Notes

- PostgreSQL is exposed on the port defined in `.env`.
- The ingestion script automatically loads `.env` via `python-dotenv`.
- The extraction directory used by ingestion is the one defined in `WIKI_EXTRACTED_DIR`.
- The download script fails on HTTP errors and attempts to resume interrupted downloads via the `.part` file.
- The extraction script does not reuse old directories.
- If the container already exists without the table, recreate the volume or execute the initialization SQL manually.

## License

The source code in this repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Model weights and third-party datasets may be subject to their own licensing terms. This repository uses data derived from Portuguese Wikipedia and instruction datasets such as Alpaca PT-BR and Canarim-Instruct-PTBR. Users should review the respective dataset licenses and terms before redistribution or commercial use.
