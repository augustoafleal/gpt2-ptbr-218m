import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

JUDGE_RESULTS_DIR = Path("benchmark/judge_results")
JUDGE_MAPPING_PATH = Path("benchmark/judge_mapping.json")
REPORTS_DIR = Path("benchmark/reports")

JUDGE_NAMES = ["gpt", "gemini", "claude"]

CANONICAL_KEYS = [
    "correctness",
    "instruction_following",
    "factuality",
    "conciseness",
    "repetition",
    "overall",
]

ALIASES = {"A", "B", "C", "D", "E", "F", "G", "H"}


def _camel_to_snake(name: str) -> str:
    result = ""
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and name[i - 1].islower():
            result += "_"
        result += ch.lower()
    return result


def normalize_key(key: str) -> str:
    s = key.strip().replace(" ", "_")
    return _camel_to_snake(s)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mapping(path: Path) -> Dict[str, Dict[str, str]]:
    return load_json(path)


def load_judge_results(
    judge_dir: Path,
) -> Tuple[Optional[str], Dict[str, Dict[str, Any]]]:
    model_txt = judge_dir / "model.txt"
    model_name: Optional[str] = None
    if model_txt.exists():
        model_name = model_txt.read_text(encoding="utf-8").strip()

    results: Dict[str, Dict[str, Any]] = {}
    for fpath in sorted(judge_dir.glob("question_*.json")):
        key = fpath.stem
        results[key] = load_json(fpath)

    return model_name, results


def validate_data(
    all_judge_results: Dict[str, Tuple[Optional[str], Dict[str, Any]]],
    mapping: Dict[str, Dict[str, str]],
) -> int:
    question_sets = []
    for judge_name, (_, results) in all_judge_results.items():
        keys = sorted(results.keys())
        question_sets.append(keys)
        for qkey in keys:
            if qkey not in mapping:
                print(
                    f"ERROR: '{qkey}' from judge '{judge_name}' not found in mapping.",
                    file=sys.stderr,
                )
                sys.exit(1)

    ref = question_sets[0]
    for judge_name, keys in zip(all_judge_results, question_sets):
        if keys != ref:
            print(
                f"ERROR: Judge '{judge_name}' has different question set. "
                f"Expected {len(ref)} questions, got {len(keys)}.",
                file=sys.stderr,
            )
            sys.exit(1)

    return len(ref)


def normalize_scores(raw_scores: Dict[str, Any]) -> Dict[str, float]:
    normalized = {}
    for raw_key, value in raw_scores.items():
        norm_key = normalize_key(raw_key)
        if norm_key in CANONICAL_KEYS:
            normalized[norm_key] = float(value)
    return normalized


def process_all(
    all_judge_results: Dict[str, Tuple[Optional[str], Dict[str, Any]]],
    mapping: Dict[str, Dict[str, str]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    model_metrics: Dict[str, Dict[str, Any]] = {}
    question_level: Dict[str, List[Dict[str, str]]] = {}
    judge_summary: Dict[str, Dict[str, Any]] = {}

    for judge_name, (model_name, results) in all_judge_results.items():
        judge_summary[judge_name] = {
            "model": model_name or "unknown",
            "questions": len(results),
        }

        for qkey, data in results.items():
            q_mapping = mapping[qkey]
            scores = data.get("scores", {})
            ranking = data.get("ranking", [])
            winner = data.get("winner", "")
            loser = data.get("loser", "")

            if qkey not in question_level:
                question_level[qkey] = []

            winner_real = q_mapping.get(winner, winner)
            loser_real = q_mapping.get(loser, loser)
            question_level[qkey].append({
                "judge": judge_name,
                "winner": winner_real,
                "loser": loser_real,
            })

            for alias, raw_score in scores.items():
                model_id = q_mapping.get(alias)
                if model_id is None:
                    print(
                        f"ERROR: Alias '{alias}' in {qkey} (judge: {judge_name}) "
                        f"not found in mapping.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                if model_id not in model_metrics:
                    model_metrics[model_id] = {
                        "scores": {k: [] for k in CANONICAL_KEYS},
                        "wins": 0,
                        "losses": 0,
                        "ranks": [],
                    }

                norm = normalize_scores(raw_score)
                for k in CANONICAL_KEYS:
                    if k in norm:
                        model_metrics[model_id]["scores"][k].append(norm[k])

                if model_id == winner_real:
                    model_metrics[model_id]["wins"] += 1
                if model_id == loser_real:
                    model_metrics[model_id]["losses"] += 1

                if ranking:
                    if alias in ranking:
                        rank = ranking.index(alias) + 1
                    else:
                        rank = len(ranking)
                    model_metrics[model_id]["ranks"].append(rank)

    return model_metrics, question_level, judge_summary


def build_leaderboard(
    model_metrics: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for model_id, metrics in model_metrics.items():
        row: Dict[str, Any] = {"model": model_id}

        n_scores = None
        score_sums = {}
        for k in CANONICAL_KEYS:
            vals = metrics["scores"][k]
            score_sums[k] = round(sum(vals), 2)
            if n_scores is None:
                n_scores = len(vals)
            row[k] = round(statistics.mean(vals), 2) if vals else 0.0

        row["n_scores"] = n_scores or 0
        row["score_sums"] = score_sums

        ranks = metrics["ranks"]
        row["n_rankings"] = len(ranks)
        row["rank_sum"] = round(sum(ranks), 2)
        row["average_rank"] = round(statistics.mean(ranks), 2) if ranks else 0.0

        row["wins"] = metrics["wins"]
        row["losses"] = metrics["losses"]
        rows.append(row)

    rows.sort(key=lambda r: (-r["overall"], -r["wins"]))
    return rows


def validate_metrics(
    rows: List[Dict[str, Any]], num_questions: int, num_judges: int
) -> None:
    expected_count = num_questions * num_judges
    total_wins = 0
    total_losses = 0
    all_ok = True

    print("Validation checks:")
    print()

    for row in rows:
        model = row["model"]
        n_scores = row["n_scores"]
        n_rankings = row["n_rankings"]

        scores_ok = n_scores == expected_count
        if not scores_ok:
            print(
                f"  FAIL: {model} — n_scores={n_scores}, expected {expected_count}"
            )
            all_ok = False

        rankings_ok = n_rankings == expected_count
        if not rankings_ok:
            print(
                f"  FAIL: {model} — n_rankings={n_rankings}, expected {expected_count}"
            )
            all_ok = False

        total_wins += row["wins"]
        total_losses += row["losses"]

    sum_wins_ok = total_wins == expected_count
    sum_losses_ok = total_losses == expected_count

    if not sum_wins_ok:
        print(f"  FAIL: total_wins={total_wins}, expected {expected_count}")
        all_ok = False
    if not sum_losses_ok:
        print(f"  FAIL: total_losses={total_losses}, expected {expected_count}")
        all_ok = False

    if all_ok:
        print(f"  All checks passed (expected={expected_count})")
    print()


def save_leaderboard_json(rows: List[Dict[str, Any]], path: Path) -> None:
    models = {}
    for row in rows:
        model_id = row["model"]
        entry = {k: v for k, v in row.items() if k != "model"}
        models[model_id] = entry
    result = {"models": models}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_leaderboard_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "rank", "model", "overall", "wins", "losses", "average_rank",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            writer.writerow({
                "rank": i,
                "model": row["model"],
                "overall": row["overall"],
                "wins": row["wins"],
                "losses": row["losses"],
                "average_rank": row["average_rank"],
            })


def save_leaderboard_md(rows: List[Dict[str, Any]], path: Path) -> None:
    lines = ["# Benchmark Leaderboard\n\n"]
    lines.append(
        "| Rank | Modelo              | Overall | Wins | Losses | Avg Rank |\n"
    )
    lines.append(
        "| ---- | ------------------- | ------: | ---: | -----: | -------: |\n"
    )
    for i, row in enumerate(rows, start=1):
        lines.append(
            f"| {i:<4} | {row['model']:<19} | "
            f"{row['overall']:>6.2f} | "
            f"{row['wins']:>3} | "
            f"{row['losses']:>4} | "
            f"{row['average_rank']:>7.2f} |\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def save_judge_summary(summary: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_question_level(ql: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ql, f, ensure_ascii=False, indent=2)
        f.write("\n")


def print_summary(
    rows: List[Dict[str, Any]],
    judge_summary: Dict[str, Any],
    total_evaluations: int,
) -> None:
    print(f"Total evaluations processed: {total_evaluations}")
    print()
    print("Judges found:")
    for name, info in judge_summary.items():
        print(f"  {name}: {info['model']} ({info['questions']} questions)")
    print()
    print("Top models by overall:")
    for i, row in enumerate(rows, start=1):
        print(
            f"  {i}. {row['model']} — "
            f"overall={row['overall']}, wins={row['wins']}, "
            f"losses={row['losses']}, avg_rank={row['average_rank']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate judge results into leaderboard and reports."
    )
    parser.add_argument(
        "--judge-results-dir",
        default=str(JUDGE_RESULTS_DIR),
    )
    parser.add_argument(
        "--mapping",
        default=str(JUDGE_MAPPING_PATH),
    )
    parser.add_argument(
        "--reports-dir",
        default=str(REPORTS_DIR),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = Path(args.judge_results_dir)
    if not results_dir.exists():
        print(f"ERROR: Judge results directory '{results_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        print(f"ERROR: Mapping file '{mapping_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    mapping = load_mapping(mapping_path)

    all_judge_results: Dict[str, Tuple[Optional[str], Dict[str, Any]]] = {}
    for judge_name in JUDGE_NAMES:
        judge_dir = results_dir / judge_name
        if not judge_dir.exists():
            print(f"ERROR: Judge directory '{judge_dir}' does not exist.", file=sys.stderr)
            sys.exit(1)
        model_name, results = load_judge_results(judge_dir)
        if not results:
            print(
                f"ERROR: No question_*.json files found in '{judge_dir}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        all_judge_results[judge_name] = (model_name, results)

    num_questions = validate_data(all_judge_results, mapping)

    model_metrics, question_level, judge_summary = process_all(
        all_judge_results, mapping
    )

    leaderboard_rows = build_leaderboard(model_metrics)

    reports_dir = Path(args.reports_dir)
    save_leaderboard_json(leaderboard_rows, reports_dir / "leaderboard.json")
    save_leaderboard_csv(leaderboard_rows, reports_dir / "leaderboard.csv")
    save_leaderboard_md(leaderboard_rows, reports_dir / "leaderboard.md")
    save_judge_summary(judge_summary, reports_dir / "judge_summary.json")
    save_question_level(question_level, reports_dir / "question_level_results.json")

    total_evaluations = num_questions * len(JUDGE_NAMES)
    validate_metrics(leaderboard_rows, num_questions, len(JUDGE_NAMES))
    print_summary(leaderboard_rows, judge_summary, total_evaluations)


if __name__ == "__main__":
    main()
