from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

load_dotenv()
hf_logging.set_verbosity_error()


def get_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token as hf_get_token
        token = hf_get_token()
        return token if token else None
    except Exception:
        return None


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def resolve_dtype(dtype_arg: str) -> torch.dtype | str | None:
    mapping = {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(dtype_arg)


def format_prompt(user_input: str, use_template: bool) -> str:
    if not use_template:
        return user_input
    return f"### Instrução:\n{user_input}\n\n### Resposta:\n"


def postprocess_response(response: str, prompt: str | None = None) -> str:
    cleaned = response.strip()
    if prompt and cleaned.startswith(prompt):
        cleaned = cleaned[len(prompt):].strip()
    return cleaned


def load_local_backend(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype | str | None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    token = get_hf_token()
    token_kwargs = {"token": token} if token else {}

    tokenizer = AutoTokenizer.from_pretrained(model_id, **token_kwargs)

    load_kwargs = dict(token_kwargs)
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def generate_local(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    repetition_penalty: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def load_api_backend(model_id: str) -> tuple:
    token = get_hf_token()
    from huggingface_hub import InferenceClient
    client = InferenceClient(model=model_id, token=token)
    return client, None


def generate_api(
    client,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    repetition_penalty: float,
) -> str:
    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            return_full_text=False,
        )
    except StopIteration:
        raise ConnectionError(
            "API de inferência inacessível — verifique conexão com "
            "api-inference.huggingface.co"
        ) from None
    return postprocess_response(response)


def main_loop_local(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    use_template: bool,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    repetition_penalty: float,
) -> None:
    quit_commands = {"exit", "quit", "sair"}
    print()
    while True:
        try:
            user_input = input("Você > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in quit_commands:
            break

        prompt = format_prompt(user_input, use_template)

        try:
            response = generate_local(
                model, tokenizer, prompt, device,
                max_new_tokens, temperature, top_k, top_p,
                do_sample, repetition_penalty,
            )
            if not response:
                print("Modelo > [resposta vazia — tente reformular a pergunta]")
            else:
                print(f"Modelo > {response}")
        except torch.cuda.OutOfMemoryError:
            print("Modelo > [erro: memória CUDA insuficiente. Tente reduzir max-new-tokens.]")
        except Exception as e:
            msg = str(e) if str(e) else f"{type(e).__name__}"
            print(f"Modelo > [erro durante geração: {msg}]")
        print()


def main_loop_api(
    client,
    use_template: bool,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    repetition_penalty: float,
) -> None:
    quit_commands = {"exit", "quit", "sair"}
    print()
    while True:
        try:
            user_input = input("Você > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in quit_commands:
            break

        prompt = format_prompt(user_input, use_template)

        try:
            response = generate_api(
                client, prompt,
                max_new_tokens, temperature, top_k, top_p,
                do_sample, repetition_penalty,
            )
            if not response:
                print("Modelo > [resposta vazia — tente reformular a pergunta]")
            else:
                print(f"Modelo > {response}")
        except ConnectionError as e:
            print(f"Modelo > [erro: {e}]")
        except Exception as e:
            msg = str(e) if str(e) else f"{type(e).__name__}"
            print(f"Modelo > [erro durante geração: {msg}]")
        print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI interativo para inferência com modelo Hugging Face"
    )
    parser.add_argument(
        "--backend", type=str, default="local",
        choices=["local", "api"],
        help="Backend de inferência: local (transformers) ou api (InferenceClient) (default: local)",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Caminho local ou Hub ID (ex: exports/huggingface/... ou augustoafleal/gpt2-ptbr-218m)",
    )
    parser.add_argument(
        "--instruction-template", action=argparse.BooleanOptionalAction,
        default=True, dest="use_template",
        help="Formata entrada como instrução (default: True)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    device = resolve_device(args.device) if args.backend == "local" else ""
    torch_dtype = resolve_dtype(args.dtype) if args.backend == "local" else None

    print(f"  Backend:     {args.backend}")
    print(f"  Modelo:      {args.model}")
    if args.backend == "local":
        print(f"  Device:      {device}")
        print(f"  Dtype:       {args.dtype}")
    print(f"  Max tokens:  {args.max_new_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k:       {args.top_k}")
    print(f"  Top-p:       {args.top_p}")
    print(f"  Template:    {'sim' if args.use_template else 'não'}")

    try:
        if args.backend == "local":
            model, tokenizer = load_local_backend(args.model, device, torch_dtype)
        else:
            client, _ = load_api_backend(args.model)
    except OSError as e:
        print(f"Erro ao carregar modelo: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Erro inesperado ao carregar modelo: {e}", file=sys.stderr)
        return 1

    try:
        if args.backend == "local":
            main_loop_local(
                model, tokenizer, device, args.use_template,
                args.max_new_tokens, args.temperature, args.top_k,
                args.top_p, args.do_sample, args.repetition_penalty,
            )
        else:
            main_loop_api(
                client, args.use_template,
                args.max_new_tokens, args.temperature, args.top_k,
                args.top_p, args.do_sample, args.repetition_penalty,
            )
    except KeyboardInterrupt:
        pass

    print("Até logo!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
