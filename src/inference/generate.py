from __future__ import annotations

from pathlib import Path

import sentencepiece as spm
import torch

from src.model.gpt import GPT


def load_model_for_generation(
    checkpoint: dict,
    model_config: dict,
    device: torch.device,
) -> GPT:
    model = GPT(
        vocab_size=model_config["vocab_size"],
        block_size=model_config["block_size"],
        n_embd=model_config["n_embd"],
        n_head=model_config["n_head"],
        n_layer=model_config["n_layer"],
        dropout=model_config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def generate_text(
    model: GPT,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: torch.device = torch.device("cpu"),
) -> str:
    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        idx = model.generate(idx, max_new_tokens, temperature, top_k)

    generated = idx[0].tolist()
    text = tokenizer.decode(generated)
    return text
