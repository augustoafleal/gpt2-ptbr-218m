"""Build judge input files from benchmark model outputs."""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

MODEL_OUTPUTS_DIR = Path("benchmark/model_outputs")
JUDGE_INPUTS_DIR = Path("benchmark/judge_inputs")
JUDGE_INPUTS_MD_DIR = Path("benchmark/judge_inputs_md")
JUDGE_MAPPING_PATH = Path("benchmark/judge_mapping.json")

ANSWER_FIELD_NAMES = {"answer", "response", "output", "generated_text"}

ALIASES = ["A", "B", "C", "D", "E", "F", "G", "H"]


def extract_model_id(filename: str) -> str:
    stem = Path(filename).stem
    prefix = "benchmark_run_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected filename format: {filename}")
    return stem[len(prefix):]


def find_answer_field(item: Dict[str, Any]) -> Optional[str]:
    for key, value in item.items():
        if isinstance(value, dict):
            result = find_answer_field(value)
            if result is not None:
                return result
        if key in ANSWER_FIELD_NAMES and isinstance(value, str):
            return value
    return None


def collect_fields(item: Dict[str, Any], prefix: str = "") -> set:
    fields = set()
    for key, value in item.items():
        full_key = f"{prefix}.{key}" if prefix else key
        fields.add(full_key)
        if isinstance(value, dict):
            fields |= collect_fields(value, full_key)
    return fields


def load_benchmark_data(input_dir: Path) -> List[Dict[str, Any]]:
    models_data = []
    files = sorted(input_dir.glob("benchmark_run_*.json"))

    if not files:
        print("ERROR: No benchmark_run_*.json files found.", file=sys.stderr)
        sys.exit(1)

    for filepath in files:
        model_id = extract_model_id(filepath.name)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        models_data.append({"model_id": model_id, "data": data})

    return models_data


def validate_consistency(models_data: List[Dict[str, Any]]) -> int:
    num_questions = None
    question_by_id: Dict[int, str] = {}

    for entry in models_data:
        results = entry["data"].get("results", [])
        if num_questions is None:
            num_questions = len(results)
        elif len(results) != num_questions:
            print(
                f"ERROR: Model '{entry['model_id']}' has {len(results)} questions, "
                f"expected {num_questions}.",
                file=sys.stderr,
            )
            sys.exit(1)

        for result in results:
            qid = result.get("id")
            question = result.get("question", "")
            if qid is not None:
                prev = question_by_id.get(qid)
                if prev is None:
                    question_by_id[qid] = question
                elif prev != question:
                    print(
                        f"ERROR: Question ID {qid} has different text across models:\n"
                        f"  '{prev}' vs\n"
                        f"  '{question}'",
                        file=sys.stderr,
                    )
                    sys.exit(1)

    if num_questions is None:
        print("ERROR: No results found in benchmark files.", file=sys.stderr)
        sys.exit(1)

    return num_questions


def build_judge_data(
    models_data: List[Dict[str, Any]], num_questions: int
) -> List[Dict[str, Any]]:
    questions_data = []

    for q_idx in range(num_questions):
        first_result = models_data[0]["data"]["results"][q_idx]
        qid = first_result.get("id", q_idx + 1)
        question = first_result.get("question", "")

        answers: Dict[str, str] = {}
        for entry in models_data:
            model_id = entry["model_id"]
            r = entry["data"]["results"][q_idx]
            answer = find_answer_field(r)
            if answer is None:
                available = collect_fields(r)
                print(
                    f"ERROR: Could not find answer field in model '{model_id}', "
                    f"question ID {qid}.\n"
                    f"Available fields: {', '.join(sorted(available))}",
                    file=sys.stderr,
                )
                sys.exit(1)
            answers[model_id] = answer

        questions_data.append({
            "question_id": qid,
            "question": question,
            "answers": answers,
        })

    return questions_data


def build_blind_mapping(
    models_data: List[Dict[str, Any]], num_questions: int
) -> Dict[str, Dict[str, str]]:
    rng = random.Random(42)
    model_ids = [entry["model_id"] for entry in models_data]
    mapping: Dict[str, Dict[str, str]] = {}

    for q_idx in range(num_questions):
        first_result = models_data[0]["data"]["results"][q_idx]
        qid = first_result.get("id", q_idx + 1)
        key = f"question_{qid:03d}"

        shuffled = model_ids.copy()
        rng.shuffle(shuffled)

        entry = {}
        for i, model_id in enumerate(shuffled):
            entry[ALIASES[i]] = model_id
        mapping[key] = entry

    return mapping


def save_json_files(
    questions_data: List[Dict[str, Any]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for q in questions_data:
        filename = f"question_{q['question_id']:03d}.json"
        filepath = output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(q, f, ensure_ascii=False, indent=2)
            f.write("\n")


def save_markdown_files(
    questions_data: List[Dict[str, Any]],
    output_dir: Path,
    mapping: Dict[str, Dict[str, str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    judge_prompt = """Você é um avaliador especializado em modelos de linguagem.

IMPORTANTE:

- Os modelos foram anonimizados.
- Avalie apenas a qualidade das respostas.
- Ignore completamente tokens especiais como:
  <eos>, <bos>, <pad>.
- Não penalize a presença desses tokens.

Critérios:

Correctness:
2 = correta
1 = parcialmente correta
0 = incorreta

Instruction Following:
2 = seguiu completamente a instrução
1 = seguiu parcialmente
0 = não seguiu

Factuality:
2 = fatos corretos
1 = mistura fatos corretos e incorretos
0 = erros factuais importantes

Conciseness:
2 = objetiva
1 = aceitável
0 = inadequada

Repetition:
2 = sem repetição relevante
1 = alguma repetição
0 = repetição excessiva

Overall:
2 = excelente
1 = aceitável
0 = ruim

Retorne APENAS JSON válido:

{
  "scores": {
    "A": {...},
    "B": {...},
    "C": {...},
    "D": {...}
  },

  "ranking": [
    "A",
    "B",
    "C",
    "D"
  ],

  "winner": "A",

  "loser": "D",

  "justification": "..."
}"""

    for q in questions_data:
        filename = f"question_{q['question_id']:03d}.md"
        filepath = output_dir / filename
        key = f"question_{q['question_id']:03d}"
        q_mapping = mapping[key]

        lines = [f"# Pergunta {q['question_id']}\n\n"]
        lines.append("## Instrução\n\n")
        lines.append(f"{q['question']}\n\n")
        lines.append("---\n\n")

        for alias in sorted(q_mapping):
            model_id = q_mapping[alias]
            answer = q["answers"][model_id]
            lines.append(f"## Modelo {alias}\n\n")
            lines.append(f"{answer}\n\n")
            lines.append("---\n\n")

        lines.append("## Avaliação\n\n")
        lines.append(f"{judge_prompt}\n")

        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)


def save_mapping_file(
    mapping: Dict[str, Dict[str, str]], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
        f.write("\n")


def print_summary(
    models_data: List[Dict[str, Any]], num_questions: int
) -> None:
    print(f"Models loaded: {len(models_data)}")
    print(f"Questions: {num_questions}")
    print(f"JSON files generated: {num_questions}")
    print(f"Markdown files generated: {num_questions}")
    print(f"Mapping file generated: benchmark/judge_mapping.json")
    print()
    print("Models:")
    for entry in models_data:
        print(f"* {entry['model_id']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build judge input files from benchmark model outputs."
    )
    parser.add_argument(
        "--input-dir",
        default=str(MODEL_OUTPUTS_DIR),
        help=f"Input directory with benchmark_run_*.json files (default: {MODEL_OUTPUTS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(JUDGE_INPUTS_DIR),
        help=f"Output directory for JSON files (default: {JUDGE_INPUTS_DIR})",
    )
    parser.add_argument(
        "--output-dir-md",
        default=str(JUDGE_INPUTS_MD_DIR),
        help=f"Output directory for Markdown files (default: {JUDGE_INPUTS_MD_DIR})",
    )
    parser.add_argument(
        "--mapping",
        default=str(JUDGE_MAPPING_PATH),
        help=f"Output path for mapping file (default: {JUDGE_MAPPING_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    models_data = load_benchmark_data(input_dir)
    num_questions = validate_consistency(models_data)
    questions_data = build_judge_data(models_data, num_questions)
    mapping = build_blind_mapping(models_data, num_questions)

    save_json_files(questions_data, Path(args.output_dir))
    save_markdown_files(questions_data, Path(args.output_dir_md), mapping)
    save_mapping_file(mapping, Path(args.mapping))

    print_summary(models_data, num_questions)


if __name__ == "__main__":
    main()
