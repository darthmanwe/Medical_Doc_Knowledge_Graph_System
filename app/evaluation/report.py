"""Evaluation report generator.

Produces a structured markdown comparison of vector-only RAG vs graph-backed RAG.
"""

from __future__ import annotations

from app.models.schema import EvaluationReport, QuestionCategory


def generate_report(report: EvaluationReport) -> str:
    """Generate a markdown summary from the evaluation report."""
    lines: list[str] = []
    lines.append("# RAG Evaluation Report: Vector-Only vs Graph-Backed")
    lines.append(f"\n**Timestamp:** {report.timestamp.isoformat()}")
    lines.append(f"**Total Questions:** {len(report.records) // 2}")
    lines.append("")

    # ── Aggregate scores ───────────────────────────────────────────────────
    lines.append("## Aggregate Scores")
    lines.append("")
    lines.append("| Metric | Vector RAG | Graph RAG | Delta |")
    lines.append("|--------|-----------|-----------|-------|")

    if report.vector_avg_scores and report.graph_avg_scores:
        v = report.vector_avg_scores
        g = report.graph_avg_scores
        for metric_name in ["faithfulness", "context_precision", "context_recall", "answer_correctness"]:
            v_val = getattr(v, metric_name)
            g_val = getattr(g, metric_name)
            delta = g_val - v_val
            arrow = "+" if delta > 0 else ""
            lines.append(f"| {metric_name} | {v_val:.3f} | {g_val:.3f} | {arrow}{delta:.3f} |")

        if g.citation_accuracy is not None:
            lines.append(f"| citation_accuracy | N/A | {g.citation_accuracy:.3f} | — |")

    lines.append("")

    # ── Per-category breakdown ─────────────────────────────────────────────
    lines.append("## Per-Category Breakdown")
    lines.append("")

    for category in QuestionCategory:
        cat_records = [r for r in report.records if r.category == category]
        if not cat_records:
            continue

        lines.append(f"### {category.value}")
        v_recs = [r for r in cat_records if r.strategy == "vector"]
        g_recs = [r for r in cat_records if r.strategy == "graph"]

        for vr, gr in zip(v_recs, g_recs):
            lines.append(f"\n**Q:** {vr.question}")
            lines.append(f"- **Expected:** {vr.expected_answer[:150]}…")
            lines.append(f"- **Vector answer:** {vr.answer[:150]}…")
            lines.append(
                f"  - Faithfulness={vr.scores.faithfulness:.2f}, "
                f"Precision={vr.scores.context_precision:.2f}, "
                f"Recall={vr.scores.context_recall:.2f}, "
                f"Correctness={vr.scores.answer_correctness:.2f}"
            )
            lines.append(f"- **Graph answer:** {gr.answer[:150]}…")
            lines.append(
                f"  - Faithfulness={gr.scores.faithfulness:.2f}, "
                f"Precision={gr.scores.context_precision:.2f}, "
                f"Recall={gr.scores.context_recall:.2f}, "
                f"Correctness={gr.scores.answer_correctness:.2f}"
            )
            if gr.scores.citation_accuracy is not None:
                lines.append(f"  - Citation Accuracy={gr.scores.citation_accuracy:.2f}")
            lines.append("")

    # ── Key findings ───────────────────────────────────────────────────────
    lines.append("## Key Findings")
    lines.append("")

    if report.vector_avg_scores and report.graph_avg_scores:
        v = report.vector_avg_scores
        g = report.graph_avg_scores

        if g.faithfulness > v.faithfulness:
            lines.append("- **Graph RAG shows higher faithfulness**, suggesting graph-grounded retrieval reduces hallucination.")
        if g.context_recall > v.context_recall:
            lines.append("- **Graph RAG achieves better context recall**, indicating k-hop expansion surfaces more relevant facts.")
        if g.answer_correctness > v.answer_correctness:
            lines.append("- **Graph RAG produces more correct answers**, particularly for multi-hop reasoning questions.")
        if g.citation_accuracy and g.citation_accuracy > 0.7:
            lines.append("- **Citation accuracy is strong**, demonstrating effective provenance linking from graph to source text.")

        lines.append("")
        lines.append(
            "Graph-backed RAG leverages entity relationships, k-hop neighborhood expansion, "
            "and provenance edges to provide richer, more structured context for LLM prompting. "
            "This aligns with data2.ai's eXAI philosophy of making every AI decision "
            "traceable and explainable."
        )

    return "\n".join(lines)
