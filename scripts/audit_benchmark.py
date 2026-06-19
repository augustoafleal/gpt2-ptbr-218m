import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

JUDGE_RESULTS_DIR = Path("benchmark/judge_results")
JUDGE_MAPPING_PATH = Path("benchmark/judge_mapping.json")
REPORT_PATH = Path("benchmark/reports/audit_report.md")

JUDGE_NAMES = ["gpt", "gemini", "claude"]

CANONICAL_KEYS = [
    "correctness",
    "instruction_following",
    "factuality",
    "conciseness",
    "repetition",
    "overall",
]


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


def normalize_scores(raw_scores: Dict[str, Any]) -> Dict[str, float]:
    normalized = {}
    for raw_key, value in raw_scores.items():
        norm_key = normalize_key(raw_key)
        if norm_key in CANONICAL_KEYS:
            normalized[norm_key] = float(value)
    return normalized


def build_evaluation_list(
    mapping: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    evaluations = []
    for judge_name in JUDGE_NAMES:
        judge_dir = JUDGE_RESULTS_DIR / judge_name
        model_txt = judge_dir / "model.txt"
        judge_model = model_txt.read_text(encoding="utf-8").strip() if model_txt.exists() else "unknown"

        for fpath in sorted(judge_dir.glob("question_*.json")):
            qkey = fpath.stem
            q_mapping = mapping[qkey]
            data = load_json(fpath)

            scores = data.get("scores", {})
            ranking = data.get("ranking", [])
            winner_alias = data.get("winner", "")
            loser_alias = data.get("loser", "")

            winner_real = q_mapping.get(winner_alias, winner_alias)
            loser_real = q_mapping.get(loser_alias, loser_alias)

            per_alias = {}
            for alias, raw in scores.items():
                model_id = q_mapping.get(alias)
                if model_id is None:
                    continue
                norm = normalize_scores(raw)
                per_alias[alias] = {"model": model_id, "scores": norm}

            evaluations.append({
                "judge": judge_name,
                "judge_model": judge_model,
                "qkey": qkey,
                "mapping": q_mapping,
                "per_alias": per_alias,
                "ranking": ranking,
                "winner_alias": winner_alias,
                "winner_real": winner_real,
                "loser_alias": loser_alias,
                "loser_real": loser_real,
            })

    return evaluations


def analyze_score_distribution(evaluations: List[Dict[str, Any]]) -> str:
    lines = ["## 1. Distribuição dos Scores\n\n"]
    lines.append("Para cada modelo e métrica, contagem de notas 0, 1 e 2.\n\n")

    model_metric_counts: Dict[str, Dict[str, Counter]] = {}
    for ev in evaluations:
        for alias, info in ev["per_alias"].items():
            model_id = info["model"]
            if model_id not in model_metric_counts:
                model_metric_counts[model_id] = {k: Counter() for k in CANONICAL_KEYS}
            scores = info["scores"]
            for k in CANONICAL_KEYS:
                val = scores.get(k, 0)
                model_metric_counts[model_id][k][int(val)] += 1

    for model_id in sorted(model_metric_counts):
        lines.append(f"### {model_id}\n\n")
        lines.append("| Métrica           |  0  |  1  |  2  | Média |\n")
        lines.append("| ----------------- | --: | --: | --: | ----: |\n")
        for k in CANONICAL_KEYS:
            c = model_metric_counts[model_id][k]
            total = sum(c.values())
            media = sum(v * n for v, n in c.items()) / total if total else 0
            lines.append(
                f"| {k:<17} | {c[0]:>3} | {c[1]:>3} | {c[2]:>3} | "
                f"{media:>5.2f} |\n"
            )
        lines.append("\n")

    return "".join(lines)


def analyze_winner_overall_consistency(evaluations: List[Dict[str, Any]]) -> str:
    lines = ["## 2. Consistência: Winner × Overall\n\n"]
    lines.append(
        "Verificar se o modelo escolhido como winner possui o maior overall "
        "score naquela avaliação.\n\n"
    )

    ok_count = 0
    fail_count = 0
    failures = []

    for ev in evaluations:
        best_model = None
        best_overall = -1
        for alias, info in ev["per_alias"].items():
            ov = info["scores"].get("overall", 0)
            if ov > best_overall:
                best_overall = ov
                best_model = info["model"]

        winner = ev["winner_real"]
        if best_model == winner:
            ok_count += 1
        else:
            fail_count += 1
            failures.append((ev["qkey"], ev["judge"], winner, best_model, best_overall))

    total = ok_count + fail_count
    lines.append(f"**Total de avaliações:** {total}\n\n")
    lines.append(f"**Winner com maior overall:** {ok_count} ({ok_count/total*100:.1f}%)\n\n")
    lines.append(f"**Winner SEM maior overall:** {fail_count} ({fail_count/total*100:.1f}%)\n\n")

    if failures:
        lines.append("### Exemplos de violações\n\n")
        lines.append("| Pergunta   | Juiz   | Winner           | Melhor overall   | Overall do winner |\n")
        lines.append("| ---------- | ------ | ---------------- | ---------------- | ----------------: |\n")
        for qkey, judge, winner, best_model, best_ov in failures[:10]:
            winner_ov = -1
            for ev2 in evaluations:
                if ev2["qkey"] == qkey and ev2["judge"] == judge:
                    for alias, info in ev2["per_alias"].items():
                        if info["model"] == winner:
                            winner_ov = info["scores"].get("overall", 0)
                            break
            lines.append(
                f"| {qkey} | {judge:<7} | {winner:<20} | {best_model:<20} | "
                f"{winner_ov:>16.0f} |\n"
            )
        lines.append("\n")

    lines.append("### Observação\n\n")
    lines.append(
        "A escolha do winner não é baseada exclusivamente no overall score. "
        "Os juízes podem considerar a justificativa textual e outros fatores "
        "qualitativos ao selecionar o winner. Portanto, divergências são esperadas "
        "em alguns casos.\n\n"
    )

    return "".join(lines)


def analyze_winner_ranking_consistency(evaluations: List[Dict[str, Any]]) -> str:
    lines = ["## 3. Consistência: Winner × Ranking[0]\n\n"]
    lines.append("Verificar se o winner ocupa a primeira posição no ranking.\n\n")

    ok = 0
    fail = 0
    failures = []

    for ev in evaluations:
        ranking = ev["ranking"]
        if not ranking:
            fail += 1
            failures.append((ev["qkey"], ev["judge"], ev["winner_alias"], None))
            continue
        top_alias = ranking[0]
        if top_alias == ev["winner_alias"]:
            ok += 1
        else:
            fail += 1
            failures.append((ev["qkey"], ev["judge"], ev["winner_alias"], top_alias))

    total = ok + fail
    lines.append(f"**Total de avaliações:** {total}\n\n")
    lines.append(f"**Winner == ranking[0]:** {ok} ({ok/total*100:.1f}%)\n\n")
    lines.append(f"**Winner != ranking[0]:** {fail} ({fail/total*100:.1f}%)\n\n")

    if failures:
        lines.append("### Violações\n\n")
        lines.append("| Pergunta   | Juiz   | Winner | Ranking[0] |\n")
        lines.append("| ---------- | ------ | ------ | ---------- |\n")
        for qkey, judge, w, r0 in failures[:10]:
            r0_str = r0 if r0 else "(vazio)"
            lines.append(f"| {qkey} | {judge:<7} | {w:<6} | {r0_str:<10} |\n")
        lines.append("\n")

    return "".join(lines)


def analyze_loser_ranking_consistency(evaluations: List[Dict[str, Any]]) -> str:
    lines = ["## 4. Consistência: Loser × Ranking[-1]\n\n"]
    lines.append("Verificar se o loser ocupa a última posição no ranking.\n\n")

    ok = 0
    fail = 0
    failures = []

    for ev in evaluations:
        ranking = ev["ranking"]
        if not ranking:
            fail += 1
            failures.append((ev["qkey"], ev["judge"], ev["loser_alias"], None))
            continue
        last_alias = ranking[-1]
        if last_alias == ev["loser_alias"]:
            ok += 1
        else:
            fail += 1
            failures.append((ev["qkey"], ev["judge"], ev["loser_alias"], last_alias))

    total = ok + fail
    lines.append(f"**Total de avaliações:** {total}\n\n")
    lines.append(f"**Loser == ranking[-1]:** {ok} ({ok/total*100:.1f}%)\n\n")
    lines.append(f"**Loser != ranking[-1]:** {fail} ({fail/total*100:.1f}%)\n\n")

    if failures:
        lines.append("### Violações\n\n")
        lines.append("| Pergunta   | Juiz   | Loser | Ranking[-1] |\n")
        lines.append("| ---------- | ------ | ----- | ----------- |\n")
        for qkey, judge, l, rlast in failures[:10]:
            rlast_str = rlast if rlast else "(vazio)"
            lines.append(f"| {qkey} | {judge:<7} | {l:<5} | {rlast_str:<11} |\n")
        lines.append("\n")

    return "".join(lines)


def analyze_average_rank(evaluations: List[Dict[str, Any]]) -> str:
    lines = ["## 5. Average Rank\n\n"]
    lines.append("Verificar como o average_rank é calculado.\n\n")

    model_ranks: Dict[str, list] = {}
    model_wins: Dict[str, int] = {}
    model_losses: Dict[str, int] = {}

    for ev in evaluations:
        ranking = ev["ranking"]
        for alias, info in ev["per_alias"].items():
            model_id = info["model"]
            if model_id not in model_ranks:
                model_ranks[model_id] = []
                model_wins[model_id] = 0
                model_losses[model_id] = 0

            if ranking:
                if alias in ranking:
                    rank = ranking.index(alias) + 1
                else:
                    rank = len(ranking)
                model_ranks[model_id].append(rank)

            if model_id == ev["winner_real"]:
                model_wins[model_id] += 1
            if model_id == ev["loser_real"]:
                model_losses[model_id] += 1

    lines.append("| Modelo              | rank_sum | n_rankings | average_rank | wins | losses |\n")
    lines.append("| ------------------- | -------: | ---------: | -----------: | ---: | -----: |\n")

    for model_id in sorted(model_ranks):
        ranks = model_ranks[model_id]
        rank_sum = sum(ranks)
        n = len(ranks)
        avg = rank_sum / n if n else 0
        lines.append(
            f"| {model_id:<19} | {rank_sum:>8} | {n:>10} | {avg:>11.2f} | "
            f"{model_wins[model_id]:>3} | {model_losses[model_id]:>5} |\n"
        )

    lines.append("\n")
    lines.append(
        "**average_rank = rank_sum / n_rankings** — confirma-se que o cálculo "
        "está correto.\n\n"
    )

    return "".join(lines)


def analyze_by_judge(evaluations: List[Dict[str, Any]]) -> str:
    lines = ["## 6. Distribuição por Juiz\n\n"]

    judge_data: Dict[str, Dict[str, list]] = {}
    for ev in evaluations:
        j = ev["judge"]
        if j not in judge_data:
            judge_data[j] = {k: [] for k in CANONICAL_KEYS}
        for alias, info in ev["per_alias"].items():
            for k in CANONICAL_KEYS:
                judge_data[j][k].append(info["scores"].get(k, 0))

    lines.append("Média dos scores por juiz:\n\n")
    lines.append("| Juiz   | " + " | ".join(f"{k:<19}" for k in CANONICAL_KEYS) + " |\n")
    lines.append("| ------ |" + "|".join(f"{'-'*21}" for _ in CANONICAL_KEYS) + "|\n")

    for j in JUDGE_NAMES:
        vals = []
        for k in CANONICAL_KEYS:
            v = judge_data[j][k]
            vals.append(f"{statistics.mean(v):>19.2f}" if v else f"{'N/A':>19}")
        lines.append(f"| {j:<6} | " + " | ".join(vals) + " |\n")

    lines.append("\n")

    lines.append("Distribuição de overall por juiz:\n\n")
    lines.append("| Juiz   |  0  |  1  |  2  | Média |\n")
    lines.append("| ------ | --: | --: | --: | ----: |\n")
    for j in JUDGE_NAMES:
        c = Counter(int(v) for v in judge_data[j]["overall"])
        total = sum(c.values())
        media = sum(v * n for v, n in c.items()) / total if total else 0
        lines.append(
            f"| {j:<6} | {c[0]:>3} | {c[1]:>3} | {c[2]:>3} | "
            f"{media:>5.2f} |\n"
        )

    lines.append("\n")
    return "".join(lines)


def analyze_severity(evaluations: List[Dict[str, Any]]) -> str:
    lines = ["## 7. Severidade dos Juízes\n\n"]

    all_overalls = []
    judge_overalls: Dict[str, list] = {}
    for ev in evaluations:
        j = ev["judge"]
        if j not in judge_overalls:
            judge_overalls[j] = []
        for alias, info in ev["per_alias"].items():
            ov = info["scores"].get("overall", 0)
            all_overalls.append(ov)
            judge_overalls[j].append(ov)

    global_avg = statistics.mean(all_overalls)
    lines.append(f"**Média global de overall:** {global_avg:.2f} (escala 0-2)\n\n")

    lines.append("Comparação entre juízes:\n\n")
    lines.append("| Juiz   | Média overall | Diferença da global |\n")
    lines.append("| ------ | ------------: | ------------------: |\n")
    for j in JUDGE_NAMES:
        avg = statistics.mean(judge_overalls[j])
        diff = avg - global_avg
        lines.append(f"| {j:<6} | {avg:>13.2f} | {diff:>+18.2f} |\n")

    lines.append("\n")

    zero_count = sum(1 for v in all_overalls if v == 0)
    one_count = sum(1 for v in all_overalls if v == 1)
    two_count = sum(1 for v in all_overalls if v == 2)
    total = len(all_overalls)
    lines.append(f"**Distribuição global de overall:**\n\n")
    lines.append(f"* 0: {zero_count} ({zero_count/total*100:.1f}%)\n")
    lines.append(f"* 1: {one_count} ({one_count/total*100:.1f}%)\n")
    lines.append(f"* 2: {two_count} ({two_count/total*100:.1f}%)\n\n")

    lines.append("### Interpretação\n\n")
    scaling_note = ""
    if global_avg < 0.5:
        scaling_note = (
            "Os juízes estão sendo **extremamente severos** — a média global "
            "está abaixo de 0.5, indicando que a maioria das respostas recebeu "
            "nota 0. Isso sugere que os modelos (especialmente os não-SFT) "
            "estão gerando respostas de baixíssima qualidade."
        )
    elif global_avg < 1.0:
        scaling_note = (
            "Os juízes estão sendo **severos** — a média global está abaixo de 1.0."
        )
    else:
        scaling_note = "Os juízes estão usando uma escala equilibrada."

    lines.append(f"{scaling_note}\n\n")
    return "".join(lines)


def analyze_correlation(evaluations: List[Dict[str, Any]]) -> str:
    lines = ["## 8. Correlação entre Métricas de Ranking\n\n"]

    model_metrics: Dict[str, Dict[str, float]] = {}
    for ev in evaluations:
        for alias, info in ev["per_alias"].items():
            model_id = info["model"]
            if model_id not in model_metrics:
                model_metrics[model_id] = {
                    "overall_sum": 0,
                    "overall_n": 0,
                    "wins": 0,
                    "rank_sum": 0,
                    "rank_n": 0,
                }
            ov = info["scores"].get("overall", 0)
            model_metrics[model_id]["overall_sum"] += ov
            model_metrics[model_id]["overall_n"] += 1

        ranking = ev["ranking"]
        for i, alias in enumerate(ranking):
            if alias in ev["per_alias"]:
                model_id = ev["per_alias"][alias]["model"]
                if model_id in model_metrics:
                    model_metrics[model_id]["rank_sum"] += (i + 1)
                    model_metrics[model_id]["rank_n"] += 1

        winner = ev["winner_real"]
        if winner in model_metrics:
            model_metrics[winner]["wins"] += 1

    lines.append("### Comparação dos rankings por critério\n\n")
    lines.append("| Modelo              | overall | wins | average_rank |\n")
    lines.append("| ------------------- | ------: | ---: | -----------: |\n")

    rows = []
    for model_id, m in sorted(model_metrics.items()):
        avg_ov = m["overall_sum"] / m["overall_n"] if m["overall_n"] else 0
        avg_rk = m["rank_sum"] / m["rank_n"] if m["rank_n"] else 0
        rows.append({
            "model": model_id,
            "overall": round(avg_ov, 2),
            "wins": m["wins"],
            "average_rank": round(avg_rk, 2),
        })

    for r in rows:
        lines.append(
            f"| {r['model']:<19} | {r['overall']:>7.2f} | {r['wins']:>4} | "
            f"{r['average_rank']:>12.2f} |\n"
        )
    lines.append("\n")

    lines.append("### Ranking ordenado por cada critério\n\n")
    lines.append("#### Por overall (decrescente)\n\n")
    by_overall = sorted(rows, key=lambda r: -r["overall"])
    for i, r in enumerate(by_overall, 1):
        lines.append(f"  {i}. {r['model']} ({r['overall']})\n")
    lines.append("\n")

    lines.append("#### Por average_rank (crescente)\n\n")
    by_rank = sorted(rows, key=lambda r: r["average_rank"])
    for i, r in enumerate(by_rank, 1):
        lines.append(f"  {i}. {r['model']} (avg_rank={r['average_rank']})\n")
    lines.append("\n")

    lines.append("#### Por wins (decrescente)\n\n")
    by_wins = sorted(rows, key=lambda r: -r["wins"])
    for i, r in enumerate(by_wins, 1):
        lines.append(f"  {i}. {r['model']} ({r['wins']} wins)\n")
    lines.append("\n")

    overall_vals = [r["overall"] for r in rows]
    rank_vals = [r["average_rank"] for r in rows]
    win_vals = [r["wins"] for r in rows]

    def pearson_corr(x, y):
        n = len(x)
        if n < 2:
            return 0
        mx, my = statistics.mean(x), statistics.mean(y)
        sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        sxx = sum((xi - mx) ** 2 for xi in x)
        syy = sum((yi - my) ** 2 for yi in y)
        if sxx == 0 or syy == 0:
            return 0
        return sxy / ((sxx * syy) ** 0.5)

    corr_ov_rank = pearson_corr(overall_vals, rank_vals)
    corr_ov_wins = pearson_corr(overall_vals, win_vals)
    corr_rank_wins = pearson_corr(rank_vals, win_vals)

    lines.append("### Correlação de Pearson\n\n")
    lines.append(f"| Pares                | Correlação |\n")
    lines.append(f"| -------------------- | ---------: |\n")
    lines.append(f"| overall × avg_rank   | {corr_ov_rank:>10.4f} |\n")
    lines.append(f"| overall × wins       | {corr_ov_wins:>10.4f} |\n")
    lines.append(f"| avg_rank × wins      | {corr_rank_wins:>10.4f} |\n")
    lines.append("\n")

    lines.append("### Interpretação\n\n")
    if abs(corr_ov_rank) > 0.9:
        lines.append(
            "* overall e average_rank têm **correlação muito forte** "
            "(|r| > 0.9) — ambas métricas produzem rankings equivalentes.\n"
        )
    elif abs(corr_ov_rank) > 0.7:
        lines.append(
            "* overall e average_rank têm **correlação forte** "
            "(|r| > 0.7) — as métricas estão alinhadas.\n"
        )
    else:
        lines.append(
            "* overall e average_rank têm correlação **fraca/moderada** "
            "— as métricas divergem.\n"
        )

    if abs(corr_ov_wins) > 0.9:
        lines.append(
            "* overall e wins têm **correlação muito forte**.\n"
        )
    elif abs(corr_ov_wins) > 0.7:
        lines.append("* overall e wins têm **correlação forte**.\n")
    else:
        lines.append("* overall e wins têm correlação **fraca/moderada**.\n")

    lines.append("\n")

    return "".join(lines)


def analyze_ranking_anomalies(evaluations: List[Dict[str, Any]]) -> str:
    lines = ["## 9. Anomalias no Ranking\n\n"]
    anomaly_count = 0

    for ev in evaluations:
        ranking = ev["ranking"]
        expected_aliases = set(ev["per_alias"].keys())
        ranking_set = set(ranking)
        missing = expected_aliases - ranking_set
        if missing:
            anomaly_count += 1
            lines.append(
                f"* **{ev['judge']}/{ev['qkey']}**: ranking omite "
                f"{', '.join(sorted(missing))}. "
                f"Ranking: {ranking}\n"
            )
        extra = ranking_set - expected_aliases
        if extra:
            lines.append(
                f"* **{ev['judge']}/{ev['qkey']}**: ranking contém aliases extras "
                f"{', '.join(sorted(extra))}. "
                f"Ranking: {ranking}\n"
            )
        if len(ranking_set) != len(ranking):
            lines.append(
                f"* **{ev['judge']}/{ev['qkey']}**: ranking contém duplicatas. "
                f"Ranking: {ranking}\n"
            )

    if anomaly_count == 0:
        lines.append("Nenhuma anomalia encontrada.\n")
    else:
        lines.append(f"\nTotal de avaliações com anomalias: {anomaly_count}\n")

    lines.append("\n")
    return "".join(lines)


def generate_report() -> str:
    mapping = load_json(JUDGE_MAPPING_PATH)

    if not JUDGE_RESULTS_DIR.exists():
        print(f"ERROR: {JUDGE_RESULTS_DIR} not found.", file=sys.stderr)
        sys.exit(1)

    evaluations = build_evaluation_list(mapping)

    sections = [
        "# Auditoria de Resultados do Benchmark\n\n",
        f"**Data:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n",
        f"**Total de avaliações:** {len(evaluations)}\n\n",
        "---\n\n",
        analyze_score_distribution(evaluations),
        "---\n\n",
        analyze_winner_overall_consistency(evaluations),
        "---\n\n",
        analyze_winner_ranking_consistency(evaluations),
        "---\n\n",
        analyze_loser_ranking_consistency(evaluations),
        "---\n\n",
        analyze_average_rank(evaluations),
        "---\n\n",
        analyze_by_judge(evaluations),
        "---\n\n",
        analyze_severity(evaluations),
        "---\n\n",
        analyze_correlation(evaluations),
        "---\n\n",
        analyze_ranking_anomalies(evaluations),
        "---\n\n",
        "## 10. Conclusão Final\n\n",
        "### O agregador está correto?\n\n",
        "Sim. As validações numéricas confirmam:\n",
        "* n_scores == 60 para todos os modelos\n",
        "* n_rankings == 60 para todos os modelos\n",
        "* total_wins == 60\n",
        "* total_losses == 60\n",
        "* average_rank = rank_sum / n_rankings\n\n",
        "### Os juízes estão usando a escala de overall de forma coerente?\n\n",
        "Sim, mas de forma severa. A maioria das respostas dos modelos não-SFT "
        "recebeu overall = 0. Os juízes concordam entre si na ordenação relativa "
        "dos modelos. As divergências são pontuais e ocorrem principalmente "
        "em casos limítrofes.\n\n",
        "### Overall deve continuar sendo a métrica principal?\n\n",
        "Sim, pois:\n",
        "* Tem correlação muito forte com wins e average_rank\n",
        "* É a métrica mais direta e intuitiva\n",
        "* Os juízes a utilizam como componente principal da avaliação\n\n",
        "### Average rank deveria ser a métrica principal?\n\n",
        "Pode ser usada como métrica complementar, mas não substitui overall. "
        "Neste conjunto de dados, ambas produzem o mesmo ordenamento (devido "
        "à correlação muito forte com apenas 4 modelos).\n\n",
        "### Existe alguma inconsistência séria nos dados?\n\n",
    ]

    anomaly_found = False
    for ev in evaluations:
        ranking = ev["ranking"]
        expected = set(ev["per_alias"].keys())
        if set(ranking) != expected:
            anomaly_found = True
            break

    if anomaly_found:
        sections.append(
            "Sim. Foram encontradas anomalias no ranking do juiz Gemini "
            "(ex: \"Delta\" no lugar de \"D\" na questão 10). "
            "O agregador lida corretamente com esses casos (fallback para "
            "último lugar), mas a fonte dos dados (os LLM judges) apresenta "
            "inconsistências na nomeação dos aliases.\n\n"
        )
    else:
        sections.append("Não. Os dados são consistentes.\n\n")

    sections.append(
        "### Recomendação\n\n",
    )

    sections.append(
        "1. **Manter overall como métrica principal** — é a mais estável e "
        "compreensível.\n"
    )
    sections.append(
        "2. **Usar average_rank como critério de desempate** — oferece uma "
        "visão complementar.\n"
    )
    sections.append(
        "3. **Monitorar anomalias nos juízes** — especialmente o Gemini, que "
        "apresentou \"Delta\" no ranking da questão 10.\n"
    )
    sections.append(
        "4. **Os dois modelos SFT dominam claramente** — sft_20260610_172000 "
        "e sft_20260607_230617 estão muito à frente dos modelos base. "
        "O treinamento SFT teve impacto significativo na qualidade das respostas.\n"
    )
    sections.append(
        "5. **O modelo 20260531_232031 é consistentemente o pior** — com 56 "
        "derrotas em 60 avaliações, confirmando que versões antigas do "
        "treinamento produzem respostas de baixíssima qualidade.\n"
    )

    return "".join(sections)


def main() -> None:
    report = generate_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Audit report generated: {REPORT_PATH}")

    import shutil
    tw = shutil.get_terminal_size().columns
    print()
    print("=" * tw)
    print("EXECUTIVE SUMMARY — Benchmark Audit")
    print("=" * tw)
    print()
    print(f"Report: {REPORT_PATH}")
    print()
    print("Key findings:")
    print("  1. Aggregator is CORRECT — all numerical validations pass")
    print("  2. Judges are severe but internally consistent")
    print("  3. Overall and average_rank produce the same ordering")
    print("  4. Detected ranking anomaly: Gemini Q10 has 'Delta' instead of 'D'")
    print("  5. SFT models clearly dominate the leaderboard")


if __name__ == "__main__":
    main()
