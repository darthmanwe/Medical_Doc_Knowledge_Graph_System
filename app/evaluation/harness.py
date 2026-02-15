"""Side-by-side evaluation harness: Vector-only RAG vs Graph-backed RAG.

Runs every gold-standard question through both strategies, collects metrics,
and produces a comparison report.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from app.evaluation.metrics import evaluate_response
from app.evaluation.questions import GOLD_STANDARD_QA
from app.evaluation.report import generate_report
from app.models.schema import (
    EvaluationRecord,
    EvaluationReport,
    MetricScores,
)
from app.rag.graph_rag import graph_rag_query
from app.rag.vector_rag import vector_rag_query

logger = logging.getLogger(__name__)


def run_evaluation() -> EvaluationReport:
    """Execute the full evaluation harness.

    For each gold-standard question:
      1. Run vector-only RAG → collect answer + metrics
      2. Run graph-backed RAG → collect answer + metrics
      3. Store both records for comparison
    """
    records: list[EvaluationRecord] = []

    logger.info("Starting evaluation harness with %d questions…", len(GOLD_STANDARD_QA))
    t0 = time.time()

    for i, qa in enumerate(GOLD_STANDARD_QA, 1):
        logger.info("Question %d/%d [%s]: %s", i, len(GOLD_STANDARD_QA), qa.category.value, qa.question[:60])

        # ── Vector-only RAG ────────────────────────────────────────────────
        try:
            v_resp = vector_rag_query(qa.question)
            v_chunks = v_resp.context_bundle.raw_chunks if v_resp.context_bundle else []
            v_scores = evaluate_response(
                question=qa.question,
                answer=v_resp.answer,
                expected_answer=qa.expected_answer,
                context_chunks=v_chunks,
                citations=None,
            )
            records.append(EvaluationRecord(
                question=qa.question,
                category=qa.category,
                strategy="vector",
                answer=v_resp.answer,
                expected_answer=qa.expected_answer,
                scores=v_scores,
                retrieval_time_ms=v_resp.retrieval_time_ms,
                generation_time_ms=v_resp.generation_time_ms,
            ))
        except Exception as exc:
            logger.error("Vector RAG failed for Q%d: %s", i, exc)
            records.append(_error_record(qa, "vector"))

        # ── Graph-backed RAG ───────────────────────────────────────────────
        try:
            g_resp = graph_rag_query(qa.question)
            g_chunks = g_resp.context_bundle.raw_chunks if g_resp.context_bundle else []
            g_citations = [c.model_dump() for c in g_resp.citations] if g_resp.citations else []
            g_scores = evaluate_response(
                question=qa.question,
                answer=g_resp.answer,
                expected_answer=qa.expected_answer,
                context_chunks=g_chunks,
                citations=g_citations,
            )
            records.append(EvaluationRecord(
                question=qa.question,
                category=qa.category,
                strategy="graph",
                answer=g_resp.answer,
                expected_answer=qa.expected_answer,
                scores=g_scores,
                retrieval_time_ms=g_resp.retrieval_time_ms,
                generation_time_ms=g_resp.generation_time_ms,
            ))
        except Exception as exc:
            logger.error("Graph RAG failed for Q%d: %s", i, exc)
            records.append(_error_record(qa, "graph"))

    elapsed = time.time() - t0
    logger.info("Evaluation complete in %.1fs.", elapsed)

    # Compute aggregate scores
    vector_records = [r for r in records if r.strategy == "vector"]
    graph_records = [r for r in records if r.strategy == "graph"]

    vector_avg = _avg_scores(vector_records) if vector_records else None
    graph_avg = _avg_scores(graph_records) if graph_records else None

    report = EvaluationReport(
        timestamp=datetime.utcnow(),
        records=records,
        vector_avg_scores=vector_avg,
        graph_avg_scores=graph_avg,
    )

    report.summary = generate_report(report)
    return report


def _avg_scores(records: list[EvaluationRecord]) -> MetricScores:
    """Compute average MetricScores across records."""
    n = len(records)
    if n == 0:
        return MetricScores(faithfulness=0, context_precision=0, context_recall=0, answer_correctness=0)

    citation_vals = [r.scores.citation_accuracy for r in records if r.scores.citation_accuracy is not None]

    return MetricScores(
        faithfulness=sum(r.scores.faithfulness for r in records) / n,
        context_precision=sum(r.scores.context_precision for r in records) / n,
        context_recall=sum(r.scores.context_recall for r in records) / n,
        answer_correctness=sum(r.scores.answer_correctness for r in records) / n,
        citation_accuracy=sum(citation_vals) / len(citation_vals) if citation_vals else None,
    )


def _error_record(qa, strategy: str) -> EvaluationRecord:
    """Create an error record with zero scores."""
    return EvaluationRecord(
        question=qa.question,
        category=qa.category,
        strategy=strategy,
        answer="[ERROR: query failed]",
        expected_answer=qa.expected_answer,
        scores=MetricScores(
            faithfulness=0.0,
            context_precision=0.0,
            context_recall=0.0,
            answer_correctness=0.0,
        ),
        retrieval_time_ms=0.0,
        generation_time_ms=0.0,
    )
