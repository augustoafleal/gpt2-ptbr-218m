from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"
if VENV_PYTHON.exists():
    _interpreter = str(VENV_PYTHON)
else:
    _interpreter = sys.executable

QUESTIONS = [
    "O que é inteligência artificial?",
    "O que é aprendizado de máquina?",
    "O que é fotossíntese?",
    "O que é um planeta?",
    "O que é astronomia?",
    "Quem foi Albert Einstein?",
    "Quem foi Isaac Newton?",
    "Qual é a capital do Brasil?",
    "O que é Porto Alegre?",
    "Explique o que foi a Segunda Guerra Mundial.",
    "Liste três planetas do Sistema Solar.",
    "Liste três cores primárias.",
    "Liste três linguagens de programação.",
    "Dê três exemplos de animais mamíferos.",
    "Liste dois benefícios de estudar matemática.",
    "Quanto é 2 + 2?",
    "Se João tem 3 maçãs e ganha mais 2, quantas maçãs ele tem?",
    "Responda apenas sim ou não: a Terra é um planeta?",
    "Responda apenas sim ou não: o Sol é uma estrela?",
    'Classifique como positivo ou negativo: "Eu gostei muito do filme."',
]

MODES = {
    "normal": {
        "temperature": 0.3,
        "top_k": 20,
        "max_new_tokens": 120,
        "stop_at_eos": True,
    },
    "creative": {
        "temperature": 0.7,
        "top_k": 40,
        "max_new_tokens": 120,
        "stop_at_eos": True,
    },
}


def resolve_checkpoint(arg: str) -> Path:
    if arg.endswith(".pt"):
        path = Path(arg)
    else:
        path = PROJECT_ROOT / "runs" / arg / "best.pt"
    if not path.exists():
        print(f"Error: checkpoint not found: {path.resolve()}")
        raise SystemExit(1)
    return path.resolve()


def extract_answer(raw_output: str) -> tuple[str, bool]:
    marker = "### Resposta:"
    idx = raw_output.rfind(marker)
    if idx == -1:
        return raw_output, False

    after_marker = raw_output[idx + len(marker) :]
    lines = after_marker.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped.startswith("Generating")
            or stripped.startswith("Prompt:")
            or stripped.startswith("EOS stopping")
        ):
            continue
        if stripped == "" and not cleaned:
            continue
        cleaned.append(line)
    text = "\n".join(cleaned).strip()
    if not text:
        return raw_output, False
    return text, True


def run_generation(
    checkpoint: Path, question: str, params: dict
) -> dict:
    prompt = f"### Instrução:\n{question}\n\n### Resposta:\n"
    generate_script = PROJECT_ROOT / "scripts" / "generate_text.py"
    cmd = [
        _interpreter,
        str(generate_script),
        "--checkpoint",
        str(checkpoint),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(params["max_new_tokens"]),
        "--temperature",
        str(params["temperature"]),
        "--top-k",
        str(params["top_k"]),
    ]
    if params.get("stop_at_eos"):
        cmd.append("--stop-at-eos")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        raw_output = result.stdout
        answer, safe = extract_answer(raw_output)
        if not safe:
            answer = raw_output
        return {
            "raw_output": raw_output,
            "answer": answer,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired as e:
        partial = e.stdout or ""
        return {
            "raw_output": partial,
            "answer": "",
            "stderr": f"Timeout: {e.stderr or ''}",
            "exit_code": -1,
        }
    except Exception as e:
        return {
            "raw_output": "",
            "answer": "",
            "stderr": str(e),
            "exit_code": -1,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run automated generation benchmark for GPT model checkpoints"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint .pt file or run ID (e.g. sft_20260607_230617)",
    )
    parser.add_argument(
        "output_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to output JSON (optional, auto-generated if omitted)",
    )

    args = parser.parse_args(argv)

    checkpoint = resolve_checkpoint(args.checkpoint)

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = checkpoint.parent / f"benchmark_{timestamp}.json"

    results = []
    for i, question in enumerate(QUESTIONS, start=1):
        prompt = f"### Instrução:\n{question}\n\n### Resposta:\n"
        normal_result = run_generation(checkpoint, question, MODES["normal"])
        creative_result = run_generation(checkpoint, question, MODES["creative"])

        results.append(
            {
                "id": i,
                "question": question,
                "prompt": prompt,
                "normal": normal_result,
                "creative": creative_result,
                "manual_eval": {
                    "normal_score": None,
                    "creative_score": None,
                    "notes": "",
                    "repetition": None,
                    "format_followed": None,
                    "factual_error": None,
                },
            }
        )

    output = {
        "checkpoint": str(checkpoint),
        "created_at": datetime.now().isoformat(),
        "num_questions": len(QUESTIONS),
        "modes": {
            name: {
                "temperature": p["temperature"],
                "top_k": p["top_k"],
                "max_new_tokens": p["max_new_tokens"],
                "stop_at_eos": p.get("stop_at_eos", False),
            }
            for name, p in MODES.items()
        },
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Benchmark saved to:")
    print(output_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
