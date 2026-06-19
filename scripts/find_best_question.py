import json
import pathlib
import sys
from statistics import mean

BASE = pathlib.Path(__file__).resolve().parent.parent
JUDGE_RESULTS = BASE / "benchmark" / "judge_results"
MAPPING_FILE = BASE / "benchmark" / "judge_mapping.json"

JUDGES = ["gpt", "gemini", "claude"]
N_QUESTIONS = 20

PT30 = "20260531_232031"
PT100 = "20260603_224519"
ALPACA = "sft_20260607_230617"
CANARIM = "sft_20260610_172000"
MODELS = [PT30, PT100, ALPACA, CANARIM]

QUESTIONS = [
    "O que e inteligencia artificial?",
    "Explique o que e a gravidade.",
    "O que e a fotossintese?",
    "Qual e a funcao do figado no corpo humano?",
    "O que e ciencia?",
    "Qual e a importancia da leitura?",
    "O que e energia renovavel?",
    "Explique o que e a tabela periodica.",
    "O que e a democracia?",
    "Explique o que foi a Segunda Guerra Mundial.",
    "Liste tres planetas do Sistema Solar.",
    "Liste duas caracteristicas dos mamiferos.",
    "Cite dois tipos de energia renovavel.",
    "Cite tres animais domesticos.",
    "Liste dois beneficios de estudar matematica.",
    "Quanto e 2 + 2?",
    "Quanto e 3 + 2?",
    "Responda apenas sim ou nao: A Terra e um planeta?",
    "Responda apenas sim ou nao: O Sol e uma estrela?",
    "Classifique como positivo ou negativo: 'Eu gostei muito do filme.'",
]

def normalize_key(k):
    return k.lower().replace(" ", "_")

def load_mapping():
    with open(MAPPING_FILE) as f:
        return json.load(f)

def load_judge_results(q_num):
    results = {}
    for judge in JUDGES:
        path = JUDGE_RESULTS / judge / f"question_{q_num:03d}.json"
        with open(path) as f:
            data = json.load(f)
        results[judge] = data
    return results

def score_to_model(scores_by_alias, mapping_q):
    model_scores = {m: {} for m in MODELS}
    for alias, scores in scores_by_alias.items():
        mid = mapping_q.get(alias, alias)
        if mid in model_scores:
            model_scores[mid] = scores
    return model_scores

def avg_overall_by_judge(judge_data, mapping_q):
    alias_scores = judge_data.get("scores", {})
    model_scores = score_to_model(alias_scores, mapping_q)
    return {m: s.get("Overall", s.get("overall", 0)) for m, s in model_scores.items()}

def overall_from_judge_data(judge_data, mapping_q):
    return avg_overall_by_judge(judge_data, mapping_q)

def get_model_order(overall_dict):
    return sorted(overall_dict, key=lambda m: overall_dict[m], reverse=True)

def score_avg_across_judges(metric, judge_results, mapping_q):
    vals = {m: [] for m in MODELS}
    for judge_name, jd in judge_results.items():
        alias_scores = jd.get("scores", {})
        ms = score_to_model(alias_scores, mapping_q)
        for m in MODELS:
            s = ms[m]
            raw = s.get(metric, s.get(metric.lower().replace(" ", "_"), 0))
            vals[m].append(raw)
    return {m: mean(vals[m]) for m in MODELS}

def main():
    mapping = load_mapping()

    candidates = []

    for q_num in range(1, N_QUESTIONS + 1):
        q_key = f"question_{q_num:03d}"
        mapping_q = mapping.get(q_key, {})
        judge_results = load_judge_results(q_num)

        overall_avg = score_avg_across_judges("Overall", judge_results, mapping_q)
        correctness_avg = score_avg_across_judges("Correctness", judge_results, mapping_q)
        repetition_avg = score_avg_across_judges("Repetition", judge_results, mapping_q)

        o30 = overall_avg[PT30]
        o100 = overall_avg[PT100]
        oAlp = overall_avg[ALPACA]
        oCan = overall_avg[CANARIM]

        r30 = repetition_avg[PT30]
        r100 = repetition_avg[PT100]
        rAlp = repetition_avg[ALPACA]
        rCan = repetition_avg[CANARIM]

        ordering_ok = (o30 < o100 < oAlp <= oCan)
        sft_rep_ok = (rAlp >= 1.0 and rCan >= 1.0)
        sft_correct_ok = (correctness_avg[ALPACA] >= 0.5 or correctness_avg[CANARIM] >= 1.0)
        pt30_bad = (o30 < 0.5)

        is_open = (q_num <= 15)

        score = 0
        reasons = []
        if ordering_ok:
            score += 10
            reasons.append("ordering_ok")
        if sft_rep_ok:
            score += 5
            reasons.append("sft_rep_ok")
        if sft_correct_ok:
            score += 3
            reasons.append("sft_correct_ok")
        if pt30_bad:
            score += 2
            reasons.append("pt30_bad")
        if is_open:
            score += 5
            reasons.append("open_ended")

        candidates.append({
            "q_num": q_num,
            "question": QUESTIONS[q_num - 1],
            "score": score,
            "overall": overall_avg,
            "correctness": correctness_avg,
            "repetition": repetition_avg,
            "ordering": [PT30, PT100, ALPACA, CANARIM],
            "ordering_ok": ordering_ok,
            "sft_rep_ok": sft_rep_ok,
            "sft_correct_ok": sft_correct_ok,
            "pt30_bad": pt30_bad,
            "is_open": is_open,
            "reasons": reasons,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)

    print("=" * 90)
    print(f"{'Rank':<5} {'Q#':<4} {'Score':<6} {'Ordering':<10} {'SFT_Rep':<8} {'SFT_Corr':<8} {'Open':<6} Question")
    print("=" * 90)
    for i, c in enumerate(candidates):
        ok = "OK" if c["ordering_ok"] else "NO"
        rep = "OK" if c["sft_rep_ok"] else "NO"
        corr = "OK" if c["sft_correct_ok"] else "NO"
        op = "YES" if c["is_open"] else "NO"
        q = c["question"][:55]
        print(f"{i+1:<5} {c['q_num']:<4} {c['score']:<6} {ok:<10} {rep:<8} {corr:<8} {op:<6} {q}")

    print("\n" + "=" * 90)
    print("DETAILED TOP CANDIDATES")
    print("=" * 90)
    for c in candidates[:5]:
        print(f"\n--- Q{c['q_num']:02d}: {c['question']} (Score={c['score']}) ---")
        print(f"  Ordering OK: {c['ordering_ok']} | SFT Rep OK: {c['sft_rep_ok']} | SFT Corr OK: {c['sft_correct_ok']} | Open: {c['is_open']}")
        print(f"  Reasons: {c['reasons']}")
        print(f"  Overall  | PT30={c['overall'][PT30]:.3f} | PT100={c['overall'][PT100]:.3f} | Alpaca={c['overall'][ALPACA]:.3f} | Canarim={c['overall'][CANARIM]:.3f}")
        print(f"  Correct. | PT30={c['correctness'][PT30]:.3f} | PT100={c['correctness'][PT100]:.3f} | Alpaca={c['correctness'][ALPACA]:.3f} | Canarim={c['correctness'][CANARIM]:.3f}")
        print(f"  Repetit. | PT30={c['repetition'][PT30]:.3f} | PT100={c['repetition'][PT100]:.3f} | Alpaca={c['repetition'][ALPACA]:.3f} | Canarim={c['repetition'][CANARIM]:.3f}")

    print("\n" + "=" * 90)
    print("VERBOSE: All 20 questions with raw overall per judge")
    print("=" * 90)
    for q_num in range(1, N_QUESTIONS + 1):
        q_key = f"question_{q_num:03d}"
        mapping_q = mapping.get(q_key, {})
        judge_results = load_judge_results(q_num)
        print(f"\nQ{q_num:02d}: {QUESTIONS[q_num-1]}")
        for jn in JUDGES:
            ov = overall_from_judge_data(judge_results[jn], mapping_q)
            print(f"  {jn:8s}:  PT30={ov[PT30]:.0f}  PT100={ov[PT100]:.0f}  Alpaca={ov[ALPACA]:.0f}  Canarim={ov[CANARIM]:.0f}")

if __name__ == "__main__":
    main()
