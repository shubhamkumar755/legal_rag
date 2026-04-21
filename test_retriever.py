"""
Evaluation Script for the 5-Stage Modular Legal Retriever
==========================================================
Metrics: Hit@K, MRR, Precision@K, Recall, Latency per stage
Query Types: Layman, Direct Citation, Cross-Act Ambiguity, Procedural
"""

import time
import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from retriever import retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. GOLDEN DATASET
# ─────────────────────────────────────────────────────────────

@dataclass
class GoldenQuery:
    query: str
    expected_citations: list[str]     # Primary + acceptable aliases
    query_type: str                    # "layman" | "citation" | "ambiguity" | "procedural"
    notes: str = ""


GOLDEN_DATASET: list[GoldenQuery] = [

    # ── LAYMAN QUERIES ──────────────────────────────────────
    GoldenQuery(
        query="Someone took my phone by force",
        expected_citations=["BNS Section 309", "IPC Section 390", "IPC Section 392", "BNS Section 310"],
        query_type="layman",
        notes="Tests Stage 1 (Gemini expansion): should map 'force' → Robbery"
    ),
    GoldenQuery(
        query="My landlord threw my things out of the house",
        expected_citations=["Transfer of Property Act Section 108", "IPC Section 441", "BNS Section 329"],
        query_type="layman",
        notes="Tests synonym expansion: eviction / forcible entry"
    ),
    GoldenQuery(
        query="Someone is sending me threatening messages online",
        expected_citations=["IT Act Section 66A", "IT Act Section 67", "IPC Section 503", "BNS Section 351"],
        query_type="layman",
        notes="Tests cross-act mapping: cyber harassment scenario"
    ),
    GoldenQuery(
        query="My employer has not paid my salary for 3 months",
        expected_citations=["Payment of Wages Act Section 5", "Payment of Wages Act Section 15"],
        query_type="layman",
        notes="Tests domain routing to labour law"
    ),
    GoldenQuery(
        query="A shopkeeper sold me a fake branded product",
        expected_citations=["Consumer Protection Act Section 2", "Trade Marks Act Section 102", "IPC Section 420"],
        query_type="layman",
        notes="Tests multi-act retrieval for consumer fraud"
    ),

    # ── DIRECT CITATION QUERIES ──────────────────────────────
    GoldenQuery(
        query="Section 43 of IT Act",
        expected_citations=["IT Act Section 43"],
        query_type="citation",
        notes="Tests regex boost + ACT_BOOST: exact citation must rank #1"
    ),
    GoldenQuery(
        query="BNS Section 303",
        expected_citations=["BNS Section 303"],
        query_type="citation",
        notes="New code reference — tests BNS metadata tagging"
    ),
    GoldenQuery(
        query="Section 138 Negotiable Instruments Act",
        expected_citations=["Negotiable Instruments Act Section 138"],
        query_type="citation",
        notes="Cheque bounce — popular citation, tests precision"
    ),
    GoldenQuery(
        query="Article 21 of the Constitution of India",
        expected_citations=["Constitution of India Article 21"],
        query_type="citation",
        notes="Constitutional reference — tests constitutional corpus"
    ),
    GoldenQuery(
        query="BNSS Section 173",
        expected_citations=["BNSS Section 173", "CrPC Section 173"],
        query_type="citation",
        notes="New procedural code — FIR investigation report"
    ),

    # ── CROSS-ACT AMBIGUITY QUERIES ───────────────────────────
    GoldenQuery(
        query="What is Section 10?",
        expected_citations=["Indian Contract Act Section 10", "Consumer Protection Act Section 10"],
        query_type="ambiguity",
        notes="Tests Context Enrichment: system must NOT conflate different Section 10s"
    ),
    GoldenQuery(
        query="What does Section 17 say about fraud?",
        expected_citations=["Indian Contract Act Section 17"],
        query_type="ambiguity",
        notes="'fraud' keyword should disambiguate Contract Act over others"
    ),
    GoldenQuery(
        query="Explain Section 2 definition",
        expected_citations=["Consumer Protection Act Section 2", "IT Act Section 2", "BNS Section 2"],
        query_type="ambiguity",
        notes="Without act name, multiple Section 2s are valid — tests recall breadth"
    ),
    GoldenQuery(
        query="What is Section 420?",
        expected_citations=["IPC Section 420", "BNS Section 318"],
        query_type="ambiguity",
        notes="Tests legacy IPC ↔ new BNS cross-mapping"
    ),

    # ── PROCEDURAL QUERIES ────────────────────────────────────
    GoldenQuery(
        query="How do I file an FIR online?",
        expected_citations=["NLSIU Handbook", "BNSS Section 173", "BNSS Section 179"],
        query_type="procedural",
        notes="Tests Book Chunker: should find handbook/guide, not bare act"
    ),
    GoldenQuery(
        query="Steps to approach consumer forum for complaint",
        expected_citations=["Consumer Protection Act Section 35", "Consumer Protection Act Section 36", "NLSIU Handbook"],
        query_type="procedural",
        notes="Tests procedural chunking from handbooks"
    ),
    GoldenQuery(
        query="What documents do I need for anticipatory bail application?",
        expected_citations=["BNSS Section 482", "CrPC Section 438"],
        query_type="procedural",
        notes="Tests Stage 1: legal-to-procedural translation"
    ),
    GoldenQuery(
        query="How long does a court take to decide a cheque bounce case?",
        expected_citations=["Negotiable Instruments Act Section 143", "Negotiable Instruments Act Section 138"],
        query_type="procedural",
        notes="Tests timeline/procedural extraction from act text"
    ),
]


# ─────────────────────────────────────────────────────────────
# 2. RESULT NORMALISER & CITATION EXTRACTOR
# ─────────────────────────────────────────────────────────────

# Aliases so "IPC 420" == "IPC Section 420" == "Indian Penal Code Section 420"
ACT_ALIASES: dict[str, list[str]] = {
    "IPC":       ["Indian Penal Code", "IPC"],
    "BNS":       ["Bharatiya Nyaya Sanhita", "BNS"],
    "CrPC":      ["Code of Criminal Procedure", "CrPC"],
    "BNSS":      ["Bharatiya Nagarik Suraksha Sanhita", "BNSS"],
    "IT Act":    ["Information Technology Act", "IT Act"],
    "NI Act":    ["Negotiable Instruments Act", "NI Act"],
}

CITATION_PATTERN = re.compile(
    r"((?:Section|Article|§)\s*\d+[A-Za-z]?(?:\(\d+\))?)",
    re.IGNORECASE
)

ACT_PATTERN = re.compile(
    r"(BNS|BNSS|IPC|CrPC|IT Act|NI Act|"
    r"Indian Penal Code|Bharatiya Nyaya Sanhita|"
    r"Consumer Protection Act|Transfer of Property Act|"
    r"Indian Contract Act|Negotiable Instruments Act|"
    r"Payment of Wages Act|Trade Marks Act|"
    r"Constitution of India|NLSIU Handbook)",
    re.IGNORECASE
)




def normalise(citation: str) -> str:
    """Lowercase + strip whitespace for fuzzy comparison."""
    return re.sub(r"\s+", " ", citation.lower().strip())


def is_hit(retrieved_key: str, expected_citations: list[str]) -> bool:
    """True if retrieved_key matches any expected citation (alias-aware)."""
    r = normalise(retrieved_key)
    for exp in expected_citations:
        if normalise(exp) in r or r in normalise(exp):
            return True
    return False


def extract_citation_key(chunk: dict) -> str | None:
    """
    Accepts the full chunk dict.
    Handles your retriever's native format (act_name / section_number)
    AND the legacy text/metadata format.
    """
    # ── Native retriever format ──
    if isinstance(chunk, dict) and "act_name" in chunk and "section_number" in chunk:
        act = chunk["act_name"].strip()
        sec = str(chunk["section_number"]).strip()
        return f"{act} Section {sec}"

    # ── Legacy text/metadata format ──
    if isinstance(chunk, dict):
        text = chunk.get("text", "") + " " + chunk.get("metadata", {}).get("source", "")
    else:
        text = str(chunk)   # last-resort guard

    if re.search(r"NLSIU|handbook", text, re.IGNORECASE):
        return "NLSIU Handbook"

    act_match = ACT_PATTERN.search(text)
    sec_match  = CITATION_PATTERN.search(text)
    if act_match and sec_match:
        act = act_match.group(1).strip()
        sec = re.sub(r"[§\s]+", " ", sec_match.group(1)).strip().title()
        return f"{act} {sec}"

    return None


def deduplicate_results(results: list) -> list:
    """
    Collapse multiple chunks of the same section into one representative entry.
    Preserves ranking order (first occurrence wins).
    """
    seen: set[str] = set()
    deduped = []
    for chunk in results:
        key = extract_citation_key(chunk)   # ← pass the whole dict, not a string
        if key is None:
            key = chunk.get("id", str(id(chunk))) if isinstance(chunk, dict) else str(id(chunk))
        norm = normalise(key)
        if norm not in seen:
            seen.add(norm)
            if isinstance(chunk, dict):
                chunk["_citation_key"] = key
            deduped.append(chunk)
    return deduped


# ─────────────────────────────────────────────────────────────
# 3. METRIC CALCULATORS
# ─────────────────────────────────────────────────────────────

def compute_hit_at_k(results: list, expected: list[str], k: int = 5) -> bool:
    for chunk in results[:k]:
        if is_hit(chunk.get("_citation_key", ""), expected):
            return True
    return False


def compute_mrr(results: list, expected: list[str]) -> float:
    for rank, chunk in enumerate(results, start=1):
        if is_hit(chunk.get("_citation_key", ""), expected):
            return 1.0 / rank
    return 0.0


def compute_precision_at_k(results: list, expected: list[str], k: int = 5) -> float:
    hits = sum(1 for c in results[:k] if is_hit(c.get("_citation_key", ""), expected))
    return hits / k


def compute_recall(results: list, expected: list[str], k: int = 10) -> float:
    found = set()
    for chunk in results[:k]:
        for exp in expected:
            if is_hit(chunk.get("_citation_key", ""), [exp]):
                found.add(normalise(exp))
    return len(found) / len(expected) if expected else 0.0


# ─────────────────────────────────────────────────────────────
# 4. LATENCY WRAPPER
# ─────────────────────────────────────────────────────────────

@dataclass
class StageTimer:
    stage_name: str
    elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def timed_retrieve(query: str, retrieve_fn) -> tuple[list, dict[str, float]]:
    """
    Wraps your retrieve() function and captures per-stage latency.
    If your retrieve() returns (results, stage_timings) — use that.
    Otherwise, total wall-clock time is recorded.
    """
    t0 = time.perf_counter()
    output = retrieve_fn(query)
    total_ms = (time.perf_counter() - t0) * 1000

    if isinstance(output, tuple) and len(output) == 2:
        results, stage_timings = output
    else:
        results = output
        stage_timings = {"total": total_ms}

    return results, stage_timings


# ─────────────────────────────────────────────────────────────
# 5. ENRICHMENT VERIFIER
# ─────────────────────────────────────────────────────────────

def verify_enrichment(results: list) -> list[str]:
    warnings = []
    for i, chunk in enumerate(results):
        # Use native fields if available
        if "act_name" in chunk:
            if not chunk.get("act_name", "").strip():
                warnings.append(f"  [Chunk #{i+1}] Missing act_name in native chunk")
            continue

        # Fallback to text scanning
        text = chunk.get("text", "")
        has_section = bool(CITATION_PATTERN.search(text))
        has_act     = bool(ACT_PATTERN.search(text))
        if has_section and not has_act:
            preview = text[:80].replace("\n", " ")
            warnings.append(f"  [Chunk #{i+1}] Missing Act identity: '{preview}…'")
    return warnings


# ─────────────────────────────────────────────────────────────
# 6. MAIN EVALUATION RUNNER
# ─────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query: str
    query_type: str
    expected: list[str]
    retrieved_keys: list[str]
    hit_at_5: bool
    mrr: float
    precision_at_5: float
    recall: float
    latency_ms: dict[str, float]
    enrichment_warnings: list[str]
    notes: str = ""


def run_evaluation(
    retrieve_fn,
    dataset: list[GoldenQuery] = GOLDEN_DATASET,
    k: int = 5,
    verbose: bool = True,
) -> list[QueryResult]:
    """
    Args:
        retrieve_fn: Your pipeline's retrieve(query: str) → list[dict] | tuple[list[dict], dict]
        dataset:     List of GoldenQuery objects
        k:           Top-K cutoff for Hit and Precision
        verbose:     Print per-query breakdown to stdout
    """
    results_log: list[QueryResult] = []

    for gq in dataset:
        logger.info(f"Testing [{gq.query_type.upper()}]: {gq.query}")

        # ── Run retrieval with timing ──
        raw_results, latency = timed_retrieve(gq.query, retrieve_fn)

        # ── Deduplication ──
        deduped = deduplicate_results(raw_results)

        # ── Enrichment check ──
        enrich_warnings = verify_enrichment(deduped)
        if enrich_warnings and verbose:
            for w in enrich_warnings:
                logger.warning(w)

        # ── Extract citation keys for display ──
        retrieved_keys = [c.get("_citation_key", "?") for c in deduped]

        # ── Score ──
        hit    = compute_hit_at_k(deduped, gq.expected_citations, k)
        mrr    = compute_mrr(deduped, gq.expected_citations)
        prec   = compute_precision_at_k(deduped, gq.expected_citations, k)
        recall = compute_recall(deduped, gq.expected_citations, k * 2)

        qr = QueryResult(
            query=gq.query,
            query_type=gq.query_type,
            expected=gq.expected_citations,
            retrieved_keys=retrieved_keys[:k],
            hit_at_5=hit,
            mrr=mrr,
            precision_at_5=prec,
            recall=recall,
            latency_ms=latency,
            enrichment_warnings=enrich_warnings,
            notes=gq.notes,
        )
        results_log.append(qr)

        if verbose:
            status = "✓ HIT" if hit else "✗ MISS"
            print(f"\n  {status}  MRR={mrr:.2f}  P@{k}={prec:.2f}  Recall={recall:.2f}")
            print(f"  Expected : {gq.expected_citations}")
            print(f"  Retrieved: {retrieved_keys[:k]}")
            total_ms = latency.get("total", sum(latency.values()))
            print(f"  Latency  : {total_ms:.0f} ms  | stages: {latency}")

    return results_log


# ─────────────────────────────────────────────────────────────
# 7. AGGREGATE REPORT
# ─────────────────────────────────────────────────────────────

def print_report(results: list[QueryResult], k: int = 5):
    total = len(results)
    if total == 0:
        print("No results to report.")
        return

    # ── Overall metrics ──
    hit_rate   = sum(r.hit_at_5 for r in results) / total
    mean_mrr   = sum(r.mrr for r in results) / total
    mean_prec  = sum(r.precision_at_5 for r in results) / total
    mean_recall= sum(r.recall for r in results) / total

    # ── Per query-type breakdown ──
    by_type: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        by_type[r.query_type].append(r)

    # ── Latency stats ──
    all_latencies = []
    for r in results:
        total_ms = r.latency_ms.get("total", sum(r.latency_ms.values()))
        all_latencies.append(total_ms)
    avg_latency = sum(all_latencies) / len(all_latencies)
    max_latency = max(all_latencies)

    # ── Print ──
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  LEGAL RETRIEVER EVALUATION REPORT  (K={k}, N={total} queries)")
    print(sep)
    print(f"  Overall Hit@{k}   : {hit_rate*100:6.1f}%  ({sum(r.hit_at_5 for r in results)}/{total})")
    print(f"  Overall MRR      : {mean_mrr:.4f}")
    print(f"  Overall P@{k}     : {mean_prec*100:6.1f}%")
    print(f"  Overall Recall   : {mean_recall*100:6.1f}%")
    print(f"  Avg Latency      : {avg_latency:.0f} ms")
    print(f"  Max Latency      : {max_latency:.0f} ms")

    print(f"\n  ── By Query Type ──")
    type_labels = {
        "layman":     "Layman (Stage 1 Expansion)",
        "citation":   "Direct Citation (Metadata Boost)",
        "ambiguity":  "Cross-Act Ambiguity (Enrichment)",
        "procedural": "Procedural (Book Chunker)",
    }
    for qtype, label in type_labels.items():
        subset = by_type.get(qtype, [])
        if not subset:
            continue
        n = len(subset)
        h = sum(r.hit_at_5 for r in subset) / n
        m = sum(r.mrr for r in subset) / n
        p = sum(r.precision_at_5 for r in subset) / n
        print(f"  {label:<38} Hit={h*100:.0f}%  MRR={m:.2f}  P@{k}={p*100:.0f}%")

    enrich_issues = sum(1 for r in results if r.enrichment_warnings)
    if enrich_issues:
        print(f"\n  ⚠  Enrichment gaps detected in {enrich_issues}/{total} queries.")
        print(f"     Re-run pipeline.py with the updated store.py fix.")

    # ── Stage-5 Cross-encoder warning ──
    ce_slow = [r for r in results if r.latency_ms.get("stage_5_crossencoder", 0) > 2000]
    if ce_slow:
        print(f"\n  ⚠  Cross-encoder (Stage 5) exceeded 2s in {len(ce_slow)} queries.")
        print(f"     Your config.py guard should auto-disable it — verify this.")

    print(f"\n{sep}")

    # ── Miss analysis ──
    misses = [r for r in results if not r.hit_at_5]
    if misses:
        print(f"\n  MISS ANALYSIS ({len(misses)} misses):")
        for r in misses:
            print(f"\n  Query : {r.query}")
            print(f"  Type  : {r.query_type} — {r.notes}")
            print(f"  Exp   : {r.expected}")
            print(f"  Got   : {r.retrieved_keys}")
    else:
        print("\n  Perfect score — no misses! 🎯")

    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────
# 8. JSON EXPORT
# ─────────────────────────────────────────────────────────────

def export_results(results: list[QueryResult], path: str = "eval_results.json"):
    serialisable = []
    for r in results:
        serialisable.append({
            "query":              r.query,
            "query_type":        r.query_type,
            "expected":          r.expected,
            "retrieved_keys":    r.retrieved_keys,
            "hit_at_5":          r.hit_at_5,
            "mrr":               round(r.mrr, 4),
            "precision_at_5":   round(r.precision_at_5, 4),
            "recall":            round(r.recall, 4),
            "latency_ms":        {k: round(v, 1) for k, v in r.latency_ms.items()},
            "enrichment_issues": len(r.enrichment_warnings) > 0,
            "notes":             r.notes,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)
    logger.info(f"Results exported → {path}")


# ─────────────────────────────────────────────────────────────
# 9. ENTRY POINT — plug in your retrieve() here
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── OPTION A: Import your real pipeline ──
    # from pipeline import retrieve
    # results = run_evaluation(retrieve)


    print("\n  Running evaluation with STUB retriever (replace with your pipeline.retrieve)...")
    results = run_evaluation(retrieve, verbose=True)
    print_report(results)
    export_results(results, "eval_results.json")