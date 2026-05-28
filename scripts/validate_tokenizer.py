from __future__ import annotations

from collections import Counter
from pathlib import Path

import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "training" / "dataset_general.txt"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
SAMPLE_WORDS = 30


def get_real_words(path: Path, max_samples: int = 30) -> list[str]:
    title_words: Counter = Counter()
    text_words: Counter = Counter()
    articles = 0

    with path.open("r", encoding="utf-8") as f:
        line = f.readline()
        while line and len(title_words) < max_samples * 2:
            stripped = line.strip()
            if not stripped:
                line = f.readline()
                continue
            if stripped == "<eos>":
                line = f.readline()
                continue

            raw = stripped.split()
            if raw:
                for w in raw[:3]:
                    w = w.strip(""".,;:!?()[]{}""""'")
                    if len(w) > 2:
                        title_words[w] += 1

            text_line = f.readline()
            if text_line:
                for w in text_line.strip().split()[:50]:
                    w = w.strip(""".,;:!?()[]{}""""'")
                    if len(w) > 3:
                        text_words[w] += 1

            while text_line and text_line.strip() != "<eos>":
                text_line = f.readline()
                if text_line and text_line.strip():
                    for w in text_line.strip().split()[:20]:
                        w = w.strip(""".,;:!?()[]{}""""'")
                        if len(w) > 3:
                            text_words[w] += 1

            articles += 1
            if articles % 10000 == 0:
                pass

            line = f.readline()

    selected: list[str] = []
    seen = set()
    for w, _ in title_words.most_common(max_samples * 2):
        if w.lower() not in seen and len(selected) < max_samples:
            selected.append(w)
            seen.add(w.lower())

    for w, _ in text_words.most_common(max_samples * 2):
        if w.lower() not in seen and len(selected) < max_samples:
            selected.append(w)
            seen.add(w.lower())

    return selected[:max_samples]


def validate() -> None:
    sp = spm.SentencePieceProcessor()
    sp.load(str(MODEL_PATH))

    print(f"Tokenizer: {MODEL_PATH}")
    print(f"Vocab size: {sp.get_piece_size():>5}")
    print(f"Dataset:    {DATASET_PATH}")
    print()

    print(f"{'Word':<28} {'Tokens':>4} {'Pieces'}")
    print("-" * 70)

    words = get_real_words(DATASET_PATH, SAMPLE_WORDS)
    for w in words:
        pieces = sp.encode_as_pieces(w)
        ids = sp.encode(w)
        frag = " + ".join(pieces)
        print(f"{w:<28} {len(ids):>4}  {frag}")

    print()
    print("Special tokens:")
    for tok in ["<eos>", "<bos>", "<pad>", "<unk>"]:
        pid = sp.piece_to_id(tok)
        ids = sp.encode(tok)
        print(f"  {tok:<12} -> id={pid}, encode={ids}")

    print()
    print(f"bos_id(): {sp.bos_id()}, eos_id(): {sp.eos_id()}, "
          f"pad_id(): {sp.pad_id()}, unk_id(): {sp.unk_id()}")
    print(f"(use piece_to_id() if it returns -1)")


if __name__ == "__main__":
    validate()
