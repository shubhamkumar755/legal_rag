# file: scripts/eval_retriever.py

from typing import List, Dict, Tuple
from collections import Counter
from retriever import retrieve


# ───────────────────────────────────────────────
# DATASET
# ───────────────────────────────────────────────

EVAL_DATA: Dict[str, List[Tuple[str, int]]] = {
    "tenant cannot sublet without permission": [
        ("Transfer Of Property", 108)
    ],
    "data leak from company": [
        ("It_Act_2000_Updated", 43),
        ("It_Act_2000_Updated", 72)
    ],
    "assault in public place punishment": [
        ("Ipc_Act", 351)
    ],
    "landlord evicted me without notice": [
        ("Transfer Of Property", 106)
    ],
    "company shared my personal data without consent": [
        ("It_Act_2000_Updated", 72)
    ],
}


# ───────────────────────────────────────────────
# MATCH LOGIC
# ───────────────────────────────────────────────

def normalize(text: str) -> str:
    return text.lower().replace("_", " ").replace("act", "").strip()


def is_match(result: dict, expected: Tuple[str, int]) -> bool:
    try:
        act_match = normalize(expected[0]) in normalize(result["act_name"])
        section_match = int(result["section_number"]) == int(expected[1])
        return act_match and section_match
    except Exception:
        return False


# ───────────────────────────────────────────────
# METRICS
# ───────────────────────────────────────────────

def hit_at_k(results, expected_list, k):
    for r in results[:k]:
        for exp in expected_list:
            if is_match(r, exp):
                return 1
    return 0


def reciprocal_rank(results, expected_list):
    for i, r in enumerate(results):
        for exp in expected_list:
            if is_match(r, exp):
                return 1.0 / (i + 1)
    return 0.0


def precision_at_k(results, expected_list, k):
    matched = set()
    for r in results[:k]:
        for exp in expected_list:
            if is_match(r, exp):
                matched.add(exp)
    return len(matched) / k


def recall_at_k(results, expected_list, k):
    matched = set()
    for r in results[:k]:
        for exp in expected_list:
            if is_match(r, exp):
                matched.add(exp)
    return len(matched) / len(expected_list) if expected_list else 0.0


# ───────────────────────────────────────────────
# NEW: CHUNK ANALYSIS
# ───────────────────────────────────────────────

def analyze_chunk_types(results):
    counter = Counter()
    for r in results:
        counter[r.get("chunk_type", "unknown")] += 1
    return dict(counter)


def get_hit_chunk_type(results, expected):
    for r in results:
        for exp in expected:
            if is_match(r, exp):
                return r.get("chunk_type", "unknown")
    return None


def duplicate_ratio(results):
    texts = [r["text"] for r in results]
    unique = len(set(texts))
    return 1 - (unique / len(texts)) if texts else 0


# ───────────────────────────────────────────────
# MAIN EVALUATION
# ───────────────────────────────────────────────

def evaluate(mode="default"):
    total = len(EVAL_DATA)

    hit5 = mrr = precision5 = recall5 = 0

    print(f"\n===== MODE: {mode} =====")

    for query, expected in EVAL_DATA.items():
        print(f"\n🔍 Query: {query}")
        debug_recall(query, expected)  # debug recall in top-50
        # 🔥 pass mode to retriever
        results = retrieve(query, mode=mode)

        h5 = hit_at_k(results, expected, 5)
        rr = reciprocal_rank(results, expected)
        p5 = precision_at_k(results, expected, 5)
        r5 = recall_at_k(results, expected, 5)

        hit5 += h5
        mrr += rr
        precision5 += p5
        recall5 += r5

        print(f"Hit@5: {h5} | RR: {rr:.2f} | P@5: {p5:.2f} | R@5: {r5:.2f}")

        # 🔥 chunk analysis
        chunk_stats = analyze_chunk_types(results[:5])
        hit_type = get_hit_chunk_type(results, expected)
        dup_ratio = duplicate_ratio(results[:5])

        print("Chunk types:", chunk_stats)
        print("Correct chunk type:", hit_type)
        print(f"Duplicate ratio: {dup_ratio:.2f}")

        print("Top results:")
        for r in results[:5]:
            print(
                f"  - {r['act_name']} Sec {r['section_number']} "
                f"[{r.get('chunk_type')}] "
                f"(score={r.get('rerank_score', r.get('similarity', 0))})"
            )

    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS")
    print("=" * 50)

    print(f"Hit@5: {hit5 / total:.2f}")
    print(f"MRR:   {mrr / total:.2f}")
    print(f"P@5:   {precision5 / total:.2f}")
    print(f"R@5:   {recall5 / total:.2f}")


# ───────────────────────────────────────────────
# CHUNKING ABLATION TEST
# ───────────────────────────────────────────────

def evaluate_ablation():
    print("\n\n===== CHUNKING ABLATION =====")

    modes = ["section_only", "default"]

    for mode in modes:
        evaluate(mode=mode)

def debug_recall(query, expected):
    results = retrieve(query, mode="default")  # use k=50 inside

    for r in results:
        for exp in expected:
            if is_match(r, exp):
                print("✅ FOUND in top-50")
                print("Rank:", results.index(r) + 1)
                print("Chunk type:", r["chunk_type"])
                return True

    print("❌ NOT FOUND in top-50")
    return False
# ───────────────────────────────────────────────
# ENTRY
# ───────────────────────────────────────────────

if __name__ == "__main__":
    evaluate_ablation()