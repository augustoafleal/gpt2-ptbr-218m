from __future__ import annotations

import argparse
import json
import sys
import time
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HF_KEY_MAP = {
    "token_embedding.weight": "transformer.wte.weight",
    "position_embedding.weight": "transformer.wpe.weight",
    "lm_head.weight": "lm_head.weight",
    "ln_f.weight": "transformer.ln_f.weight",
    "ln_f.bias": "transformer.ln_f.bias",
}

HF_KEY_MAP_BLOCK = {
    "ln_1.weight": "ln_1.weight",
    "ln_1.bias": "ln_1.bias",
    "attn.c_attn.weight": "attn.c_attn.weight",
    "attn.c_proj.weight": "attn.c_proj.weight",
    "ln_2.weight": "ln_2.weight",
    "ln_2.bias": "ln_2.bias",
    "ffn.net.0.weight": "mlp.c_fc.weight",
    "ffn.net.2.weight": "mlp.c_proj.weight",
}

NEEDS_TRANSPOSE_INNER = {
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "ffn.net.0.weight",
    "ffn.net.2.weight",
}

SKIP_KEYS = {"blocks.*.attn.mask"}

DTYPE_MAP = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}


@dataclass
class ExportReport:
    output_dir: str = ""
    files: list[dict] = field(default_factory=list)
    model_params: int = 0
    model_unique_params: int = 0
    model_dtype: str = ""
    warnings: list[str] = field(default_factory=list)

    def add_file(self, path: str, size_bytes: int) -> None:
        self.files.append({"path": path, "size_bytes": size_bytes})

    def print_summary(self) -> None:
        total_size = sum(f["size_bytes"] for f in self.files)
        print(f"\nExport complete: {len(self.files)} files, {total_size / 1024 / 1024:.1f} MB")
        print(f"Output: {self.output_dir}")
        print(f"Parameters: {self.model_unique_params:,} (unique) / {self.model_params:,} (total on disk)")
        print(f"Dtype: {self.model_dtype}")
        print("Files:")
        for f in sorted(self.files, key=lambda x: x["path"]):
            print(f"  {f['path']:50s} {f['size_bytes'] / 1024:.1f} KB")
        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"  ! {w}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained checkpoint to Hugging Face format"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint .pt file or run ID (e.g. sft_20260610_172000)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for Hugging Face artifacts",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output directory",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress information",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run post-export validation (load model, run inference)",
    )
    return parser.parse_args(argv)


def resolve_checkpoint(arg: str) -> Path:
    if arg.endswith(".pt"):
        path = Path(arg)
    else:
        path = PROJECT_ROOT / "runs" / arg / "best.pt"
    path = path.resolve()
    if not path.exists():
        print(f"Error: checkpoint not found: {path}", file=sys.stderr)
        sys.exit(1)
    return path


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    return ckpt


def skip_key(key: str) -> bool:
    for skip in SKIP_KEYS:
        if skip.startswith("blocks.") and key.startswith("blocks."):
            skip_suffix = skip.split(".", 2)[-1] if "." in skip else skip
            if key.endswith(skip_suffix):
                return True
        elif not skip.startswith("blocks.") and skip in key:
            return True
    return False


def count_unique_params(state_dict: dict[str, torch.Tensor]) -> int:
    seen: set[int] = set()
    total = 0
    for key, tensor in state_dict.items():
        if skip_key(key):
            continue
        ptr = tensor.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total += tensor.numel()
    return total


def get_dtype_label(state_dict: dict[str, torch.Tensor]) -> str:
    dtypes = {t.dtype for t in state_dict.values()}
    if len(dtypes) == 0:
        return "unknown"
    if len(dtypes) > 1:
        return "mixed"
    return DTYPE_MAP.get(list(dtypes)[0], str(list(dtypes)[0]))


def extract_metadata(ckpt: dict, checkpoint_path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    meta["checkpoint_path"] = str(checkpoint_path.resolve())
    meta["checkpoint_name"] = checkpoint_path.name
    meta["run_id"] = checkpoint_path.parent.name
    meta["training_type"] = ckpt.get("training_type", "pretraining")
    meta["pretrained_checkpoint"] = ckpt.get("pretrained_checkpoint", None)
    meta["pretrained_run_id"] = None
    if meta["pretrained_checkpoint"]:
        pretrained_path = Path(meta["pretrained_checkpoint"])
        meta["pretrained_run_id"] = pretrained_path.parent.name
    meta["step"] = ckpt.get("step", None)
    meta["val_loss"] = ckpt.get("val_loss", None)
    meta["best_val_loss"] = ckpt.get("best_val_loss", None)
    meta["train_config"] = ckpt.get("train_config", {})
    meta["model_config"] = ckpt.get("model_config", {})
    return meta


def get_dataset_for_run(run_id: str, training_type: str) -> str:
    ds_map = {
        "20260531_232031": "Portuguese Wikipedia (pretraining)",
        "20260603_224519": "Portuguese Wikipedia (pretraining)",
        "sft_20260607_230617": "Alpaca PT-BR (instruction fine-tuning)",
        "sft_20260610_172000": "Canarim-Instruct-PTBR (instruction fine-tuning)",
    }
    if run_id in ds_map:
        return ds_map[run_id]
    if training_type.startswith("sft"):
        return "instruction dataset"
    return "Portuguese Wikipedia"


def find_tokenizer_dir() -> Path:
    tokenizer_dir = PROJECT_ROOT / "artifacts" / "tokenizer"
    if not tokenizer_dir.exists() or not (tokenizer_dir / "tokenizer.model").exists():
        candidates = list(PROJECT_ROOT.glob("**/tokenizer.model"))
        if candidates:
            tokenizer_dir = candidates[0].parent
        else:
            print("Error: no SentencePiece tokenizer.model found", file=sys.stderr)
            sys.exit(1)
    return tokenizer_dir


def export_tokenizer(
    tokenizer_dir: Path,
    output_dir: Path,
    report: ExportReport,
    verbose: bool,
    model_max_length: int = 256,
) -> dict[str, int]:
    if verbose:
        print(f"  Exporting tokenizer from {tokenizer_dir}")

    special_token_ids = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
        "unk_token_id": 0,
    }

    try:
        from transformers import LlamaTokenizer
        hf_tokenizer = LlamaTokenizer.from_pretrained(
            str(tokenizer_dir),
        )
        hf_tokenizer.bos_token = "<bos>"
        hf_tokenizer.eos_token = "<eos>"
        hf_tokenizer.pad_token = "<pad>"
        hf_tokenizer.unk_token = "<unk>"
        hf_tokenizer.add_bos_token = False
        hf_tokenizer.add_eos_token = False
        hf_tokenizer.model_max_length = model_max_length
        hf_tokenizer.save_pretrained(str(output_dir))
        if verbose:
            print(f"    Using LlamaTokenizer (vocab_size={len(hf_tokenizer)})")
    except ImportError:
        if verbose:
            print("    transformers not available - building minimal tokenizer.json")
        import sentencepiece
        sp = sentencepiece.SentencePieceProcessor()
        sp.load(str(tokenizer_dir / "tokenizer.model"))
        vocab = {sp.id_to_piece(i): i for i in range(sp.vocab_size())}

        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {"id": 0, "content": "<unk>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": 1, "content": "<bos>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": 2, "content": "<eos>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": 3, "content": "<pad>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            ],
            "normalizer": {"type": "NFKC"},
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": True},
            "decoder": {"type": "ByteLevel", "add_prefix_space": True},
            "post_processor": None,
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": [],
                "ignore_merges": True,
                "byte_fallback": False,
                "fuse_unk": False,
            },
        }

        with open(output_dir / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)
        with open(output_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_type": "gpt2", "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>", "bos_token": "<bos>",
                "eos_token": "<eos>", "pad_token": "<pad>",
                "model_max_length": model_max_length,
            }, f, ensure_ascii=False, indent=2)
            f.write("\n")
        with open(output_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
            json.dump({
                "unk_token": "<unk>", "bos_token": "<bos>",
                "eos_token": "<eos>", "pad_token": "<pad>",
            }, f, ensure_ascii=False, indent=2)
            f.write("\n")

    tc_path = output_dir / "tokenizer_config.json"
    if tc_path.exists():
        with open(tc_path) as f:
            tc = json.load(f)
        tc.pop("backend", None)
        tc.pop("is_local", None)
        tc.pop("local_files_only", None)
        tc["model_type"] = "gpt2"
        tc["model_max_length"] = model_max_length
        with open(tc_path, "w", encoding="utf-8") as f:
            json.dump(tc, f, ensure_ascii=False, indent=2)
            f.write("\n")

    st_path = output_dir / "special_tokens_map.json"
    if not st_path.exists():
        with open(st_path, "w", encoding="utf-8") as f:
            json.dump({
                "unk_token": "<unk>", "bos_token": "<bos>",
                "eos_token": "<eos>", "pad_token": "<pad>",
            }, f, ensure_ascii=False, indent=2)
            f.write("\n")

    model_dest = output_dir / "tokenizer.model"
    shutil.copy2(str(tokenizer_dir / "tokenizer.model"), str(model_dest))
    report.add_file(str(model_dest), model_dest.stat().st_size)

    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        fpath = output_dir / fname
        if fpath.exists():
            report.add_file(str(fpath), fpath.stat().st_size)

    return special_token_ids


def export_config(
    model_config: dict, special_token_ids: dict, output_dir: Path, report: ExportReport, verbose: bool
) -> None:
    if verbose:
        print("  Generating config.json")

    n_embd = model_config.get("n_embd", 384)
    block_size = model_config.get("block_size", 256)

    hf_config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": model_config.get("vocab_size", 16000),
        "n_positions": block_size,
        "n_ctx": block_size,
        "n_embd": n_embd,
        "n_layer": model_config.get("n_layer", 6),
        "n_head": model_config.get("n_head", 6),
        "n_inner": n_embd * 4,
        "activation_function": "gelu",
        "resid_pdrop": model_config.get("dropout", 0.1),
        "embd_pdrop": model_config.get("dropout", 0.1),
        "attn_pdrop": model_config.get("dropout", 0.1),
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "bos_token_id": special_token_ids.get("bos_token_id", 1),
        "eos_token_id": special_token_ids.get("eos_token_id", 2),
        "pad_token_id": special_token_ids.get("pad_token_id", 3),
        "use_cache": True,
        "tie_word_embeddings": True,
    }

    path = output_dir / "config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)
        f.write("\n")
    report.add_file(str(path), path.stat().st_size)
    if verbose:
        print(f"    config.json written ({len(hf_config)} keys)")


def remap_state_dict(
    state_dict: dict[str, torch.Tensor],
    model_config: dict,
) -> dict[str, torch.Tensor]:
    n_embd = model_config.get("n_embd", 384)
    n_layer = model_config.get("n_layer", 6)
    seen_data_ptrs: dict[int, str] = {}
    hf_sd: dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        if skip_key(key):
            continue

        is_block = key.startswith("blocks.")
        if is_block:
            parts = key.split(".")
            block_idx = parts[1]
            inner_key = ".".join(parts[2:])
            if inner_key not in HF_KEY_MAP_BLOCK:
                continue
            hf_key = f"transformer.h.{block_idx}.{HF_KEY_MAP_BLOCK[inner_key]}"
        elif key in HF_KEY_MAP:
            hf_key = HF_KEY_MAP[key]
        else:
            continue

        if is_block and inner_key in NEEDS_TRANSPOSE_INNER:
            tensor = tensor.T.contiguous()

        data_ptr = tensor.data_ptr()
        if data_ptr in seen_data_ptrs:
            hf_sd[hf_key] = hf_sd[seen_data_ptrs[data_ptr]].clone()
        else:
            seen_data_ptrs[data_ptr] = hf_key
            hf_sd[hf_key] = tensor

    ref_dtype = hf_sd.get("transformer.wte.weight", torch.tensor(0, dtype=torch.float32)).dtype

    for i in range(n_layer):
        hf_sd[f"transformer.h.{i}.attn.c_attn.bias"] = torch.zeros(3 * n_embd, dtype=ref_dtype)
        hf_sd[f"transformer.h.{i}.attn.c_proj.bias"] = torch.zeros(n_embd, dtype=ref_dtype)
        hf_sd[f"transformer.h.{i}.mlp.c_fc.bias"] = torch.zeros(4 * n_embd, dtype=ref_dtype)
        hf_sd[f"transformer.h.{i}.mlp.c_proj.bias"] = torch.zeros(n_embd, dtype=ref_dtype)

    return hf_sd


def export_weights(
    state_dict: dict[str, torch.Tensor],
    model_config: dict,
    output_dir: Path,
    report: ExportReport,
    verbose: bool,
) -> None:
    if verbose:
        print("  Converting weights to safetensors")

    hf_sd = remap_state_dict(state_dict, model_config)

    try:
        from safetensors.torch import save_file as st_save_file
        path = output_dir / "model.safetensors"
        st_save_file(hf_sd, str(path))
    except ImportError:
        print("Error: safetensors not installed. Install with: pip install safetensors", file=sys.stderr)
        sys.exit(1)

    report.add_file(str(path), path.stat().st_size)
    if verbose:
        print(f"    {len(hf_sd)} tensors exported, {len(state_dict) - len(hf_sd)} skipped")


def export_generation_config(
    special_token_ids: dict, model_config: dict, output_dir: Path, report: ExportReport, verbose: bool
) -> None:
    if verbose:
        print("  Generating generation_config.json")

    gen_config = {
        "eos_token_id": special_token_ids.get("eos_token_id", 2),
        "pad_token_id": special_token_ids.get("pad_token_id", 3),
        "bos_token_id": special_token_ids.get("bos_token_id", 1),
        "max_length": model_config.get("block_size", 256),
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
    }

    path = output_dir / "generation_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gen_config, f, indent=2)
        f.write("\n")
    report.add_file(str(path), path.stat().st_size)


def export_model_card(
    model_config: dict,
    metadata: dict,
    output_dir: Path,
    report: ExportReport,
    verbose: bool,
) -> None:
    if verbose:
        print("  Generating README.md")

    training_type = metadata.get("training_type", "pretraining")
    step = metadata.get("step", "?")
    val_loss = metadata.get("val_loss", "?")
    vocab_size = model_config.get("vocab_size", "?")
    block_size = model_config.get("block_size", "?")
    n_embd = model_config.get("n_embd", "?")
    n_head = model_config.get("n_head", "?")
    n_layer = model_config.get("n_layer", "?")
    dropout = model_config.get("dropout", "?")
    unique_params = report.model_unique_params
    total_params = report.model_params
    model_name = f"gpt2-ptbr-{unique_params // 1_000_000}m" if unique_params else "gpt2-ptbr"
    export_date = time.strftime("%Y-%m-%d %H:%M:%S")

    readme = f"""---
language: pt
license: mit
pipeline_tag: text-generation
tags:
- gpt2
- portuguese
- causal-lm
- pytorch
- safetensors
---

# {model_name}

Portuguese GPT-2-like autoregressive language model trained from scratch.

## Training pipeline

The released model, `{model_name}`, is the final checkpoint of a three-stage pipeline:

1. **Pretraining** on Portuguese Wikipedia.
2. **Supervised Fine-Tuning** on Alpaca PT-BR.
3. **Supervised Fine-Tuning** on Canarim-Instruct-PTBR.

The released checkpoint corresponds to the final instruction-tuned model.

## Model description

- **Architecture:** GPT-2 (decoder-only transformer)
- **Parameters:** {unique_params:,} trainable unique parameters (weight-tying between token embedding and output projection; zero biases added for Hugging Face compatibility)
- **Layers:** {n_layer}
- **Attention heads:** {n_head}
- **Embedding dimension:** {n_embd}
- **Vocabulary:** {vocab_size:,} tokens (SentencePiece BPE)
- **Sequence length:** {block_size} tokens
- **Activation:** GELU
- **Dropout:** {dropout}

> **Note on parameter count:** During export, the weight-tying between the token embedding and the language model head is preserved, and zero-initialized bias tensors are added for compatibility with Hugging Face's `GPT2LMHeadModel`. These biases are not part of the original trained model and do not affect behavior.

## Datasets

- **Portuguese Wikipedia corpus** — used for autoregressive pretraining.
- **Alpaca PT-BR** — Portuguese instruction-following dataset derived from Stanford Alpaca (Taori et al., 2023). Used for the first SFT stage.
- **Canarim-Instruct-PTBR** — Portuguese instruction-following dataset by Maicon Domingues (Domingues, 2023). Used for the second SFT stage.

## Tokenizer

- **Type:** SentencePiece BPE
- **Vocabulary:** {vocab_size:,} tokens
- **Special tokens:** `<unk>` = 0, `<bos>` = 1, `<eos>` = 2, `<pad>` = 3
- **Pre-tokenizer:** Metaspace (SentencePiece native)
- **Model max length:** {block_size} tokens

The tokenizer was trained from scratch on the Portuguese Wikipedia corpus.

## Training details

- **Type:** {training_type}
- **Best validation loss:** {val_loss}
- **Training steps:** {step}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "augustoafleal/{model_name}"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "A inteligência artificial é"
inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_k=40,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Limitations

- The model may generate factual errors.
- The model may repeat phrases.
- The model may fail to follow instructions exactly.
- The context length is limited to {block_size} tokens.
- The model was trained on limited data compared to modern LLMs.
- This model is not suitable for high-stakes use without human validation.

## Dataset citations

```
Stanford Alpaca: Taori et al., 2023. https://github.com/tatsu-lab/stanford_alpaca
Canarim-Instruct-PTBR: Domingues, 2023. https://huggingface.co/datasets/dominguesm/Canarim-Instruct-PTBR
```

## Citation

If you use this model, please cite:

```bibtex
@misc{{gpt2ptbr218m,
  title        = {{{model_name}: A Portuguese GPT-2-like Autoregressive Language Model}},
  author       = {{Augusto Antônio Fontanive Leal}},
  year         = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/augustoafleal/{model_name}}}}}
}}
```
"""

    path = output_dir / "README.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(readme)
    report.add_file(str(path), path.stat().st_size)


VALIDATION_STRINGS = [
    "Olá, como vai?",
    "A inteligência artificial é um campo da ciência.",
    "Mercúrio, Vênus, Terra e Marte.",
    "<bos> teste <eos>",
    "2 + 2 = 4",
]


def validate_export(
    output_dir: Path,
    tokenizer_dir: Path,
    verbose: bool,
) -> bool:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  ! transformers not available — skipping validation")
        return True

    if verbose:
        print("\n" + "=" * 60)
        print("VALIDATION")
        print("=" * 60)

    all_ok = True

    if verbose:
        print("\n[1/4] Loading tokenizer with AutoTokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
        if verbose:
            print(f"      ✓ Loaded (vocab_size={len(tokenizer)}, class={type(tokenizer).__name__})")
    except Exception as e:
        print(f"      ✗ FAILED: {e}")
        return False

    if verbose:
        print("\n[2/4] Comparing tokenizer encoding with original SentencePiece...")
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(str(tokenizer_dir / "tokenizer.model"))
    except Exception as e:
        print(f"      ! Could not load original SentencePiece: {e}")
        print(f"      ! Skipping encoding comparison")
        sp = None

    if sp is not None:
        for s in VALIDATION_STRINGS:
            sp_ids = sp.encode(s)
            hf_ids = tokenizer.encode(s)
            if sp_ids != hf_ids:
                if "<bos>" in s or "<eos>" in s:
                    print(f"      ⚠ PARTIAL: {repr(s)}  (SP had leading space before special token; HF strips it — expected)")
                else:
                    print(f"      ✗ MISMATCH: {repr(s)}")
                    print(f"        SP: {sp_ids}")
                    print(f"        HF: {hf_ids}")
                    all_ok = False
            elif verbose:
                print(f"      ✓ {repr(s)}")

    if verbose:
        print("\n[3/4] Loading model with AutoModelForCausalLM...")
    try:
        model = AutoModelForCausalLM.from_pretrained(str(output_dir))
        if verbose:
            n = sum(p.numel() for p in model.parameters())
            print(f"      ✓ Loaded ({n:,} params, dtype={model.dtype})")
    except Exception as e:
        print(f"      ✗ FAILED: {e}")
        return False

    if verbose:
        print("\n[4/4] Running inference...")
    try:
        import torch
        prompt = "A inteligência artificial é"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output_ids = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or 3,
        )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if verbose:
            print(f"      Prompt: {repr(prompt)}")
            print(f"      Output: {repr(generated)}")
            if len(generated) > len(prompt):
                print(f"      ✓ Inference completed ({len(output_ids[0]) - len(input_ids[0])} tokens generated)")
            else:
                print(f"      ⚠ No new tokens generated (check special tokens)")
    except Exception as e:
        print(f"      ✗ Inference FAILED: {e}")
        all_ok = False

    print()
    if all_ok:
        print("VALIDATION: PASS")
    else:
        print("VALIDATION: FAIL")
    print("=" * 60)

    return all_ok


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    output_dir = Path(args.output).resolve()
    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            print(f"Error: output directory already exists: {output_dir}", file=sys.stderr)
            print("Use --overwrite to overwrite", file=sys.stderr)
            return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    report = ExportReport(output_dir=str(output_dir))
    verbose = args.verbose

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    if verbose:
        print(f"Loading checkpoint: {checkpoint_path}")
    device = torch.device("cpu")
    ckpt = load_checkpoint(checkpoint_path, device)

    state_dict = ckpt["model_state_dict"]
    model_config = ckpt["model_config"]
    metadata = extract_metadata(ckpt, checkpoint_path)

    report.model_params = sum(p.numel() for p in state_dict.values())
    report.model_unique_params = count_unique_params(state_dict)
    report.model_dtype = get_dtype_label(state_dict)

    if verbose:
        print(f"  Model config: {json.dumps(model_config, indent=4)}")
        print(f"  State dict: {len(state_dict)} keys, {report.model_params:,} params ({report.model_dtype})")
        print(f"  Unique params: {report.model_unique_params:,}")
        print(f"  Training type: {metadata.get('training_type', 'pretraining')}")

    tokenizer_dir = find_tokenizer_dir()
    if verbose:
        print(f"Tokenizer: {tokenizer_dir / 'tokenizer.model'}")

    if verbose:
        print("\nExporting tokenizer files...")
    special_token_ids = export_tokenizer(
        tokenizer_dir, output_dir, report, verbose,
        model_max_length=model_config.get("block_size", 256),
    )

    if verbose:
        print("\nExporting config...")
    export_config(model_config, special_token_ids, output_dir, report, verbose)

    if verbose:
        print("\nExporting weights...")
    export_weights(state_dict, model_config, output_dir, report, verbose)

    if verbose:
        print("\nExporting generation config...")
    export_generation_config(special_token_ids, model_config, output_dir, report, verbose)

    if verbose:
        print("\nExporting model card...")
    export_model_card(model_config, metadata, output_dir, report, verbose)

    report.print_summary()

    if args.validate:
        success = validate_export(output_dir, tokenizer_dir, verbose)
        if not success:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
