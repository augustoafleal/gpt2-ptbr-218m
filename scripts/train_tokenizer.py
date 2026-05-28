from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "training" / "dataset_general.txt"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tokenizer"

VOCAB_SIZE = 16000
MODEL_TYPE = "bpe"
MODEL_PREFIX = str(ARTIFACTS_DIR / "tokenizer")

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


def train() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting tokenizer training ({MODEL_TYPE})")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Artifacts: {ARTIFACTS_DIR}")

    spm.SentencePieceTrainer.train(
        input=str(DATASET_PATH),
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type=MODEL_TYPE,
        character_coverage=1.0,
        user_defined_symbols=["<pad>", "<bos>", "<eos>"],
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        pad_piece="<pad>",
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        num_threads=4,
        input_sentence_size=0,
        shuffle_input_sentence=True,
        remove_extra_whitespaces=True,
        normalization_rule_name="nmt_nfkc",
    )

    model_file = ARTIFACTS_DIR / "tokenizer.model"
    vocab_file = ARTIFACTS_DIR / "tokenizer.vocab"

    print()
    print("Tokenizer trained successfully!")
    print(f"Model: {model_file}")
    print(f"Vocabulary: {vocab_file}")


if __name__ == "__main__":
    train()
