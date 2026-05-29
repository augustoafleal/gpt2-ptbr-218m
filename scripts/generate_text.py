from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.generate import load_model_for_generation, generate_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate text with trained GPT model")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--prompt", type=str, default="",
                        help="Prompt text to start generation")
    parser.add_argument("--max-new-tokens", type=int, default=200,
                        help="Maximum tokens to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=40,
                        help="Top-k sampling (default: 40)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device (default: cpu)")

    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return 1

    tokenizer_path = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
    if not tokenizer_path.exists():
        print(f"Tokenizer not found: {tokenizer_path}")
        return 1

    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model_config = checkpoint["model_config"]
    model_config["dropout"] = 0.0

    model = load_model_for_generation(checkpoint, model_config, device)

    prompt = args.prompt if args.prompt else sp.decode([sp.bos_id()])

    print(f"Prompt: {prompt}")
    print(f"Generating ({args.max_new_tokens} tokens, "
          f"temp={args.temperature}, top_k={args.top_k})...")
    print()

    text = generate_text(
        model, sp, prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
