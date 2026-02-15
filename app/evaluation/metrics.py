"""RAGAS-style evaluation metrics using LLM-as-judge.

Metrics:
  - Faithfulness: are the LLM claims supported by retrieved context?
  - Context Precision: is the retrieved context relevant?
  - Context Recall: are ground-truth facts covered by retrieved context?
  - Answer Correctness: semantic alignment with gold-standard answer
  - Citation Accuracy: do citations correctly map to supporting evidence? (graph-RAG only)
"""

from __future__ import annotations

import json
import logging

from app.models.schema import MetricScores
from app.rag.llm_client import extract_structured
from app.rag.embeddings import cosine_similarity, embed_text

logger = logging.getLogger(__name__)

# ── Tool schemas for LLM-as-judge ──────────────────────────────────────────────

JUDGE_TOOL = {
    "name": "evaluate_response",
    "description": "Evaluate a RAG system response against ground truth.",
    "input_schema": {
        "type": "object",
        "properties": {
            "faithfulness": {
                "type": "number",
                "description": (
                    "Score 0.0-1.0. What fraction of claims in the answer are "
                    "directly supported by the provided context? 1.0 = all claims supported."
                ),
            },
            "context_precision": {
                "type": "number",
                "description": (
                    "Score 0.0-1.0. What fraction of the retrieved context is relevant "
                    "to answering the question? 1.0 = all context is relevant."
                ),
            },
            "context_recall": {
                "type": "number",
                "description": (
                    "Score 0.0-1.0. What fraction of the facts in the expected answer "
                    "are covered by the retrieved context? 1.0 = all facts covered."
                ),
            },
            "answer_correctness": {
                "type": "number",
                "description": (
                    "Score 0.0-1.0. How correct is the generated answer compared to the "
                    "expected answer? Consider factual accuracy and completeness."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of scores.",
            },
        },
        "required": [
            "faithfulness", "context_precision", "context_recall",
            "answer_correctness", "reasoning",
        ],
    },
}

JUDGE_SYSTEM = """You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.
Your job is to objectively score the quality of a RAG response by comparing:
  - The generated answer against the expected (gold-standard) answer
  - The retrieved context against what's needed to answer correctly

Score each metric from 0.0 to 1.0 with rigorous, evidence-based reasoning.
Be strict: only give 1.0 if the criterion is fully met."""


def evaluate_response(
    question: str,
    answer: str,
    expected_answer: str,
    context_chunks: list[str],
    citations: list[dict] | None = None,
) -> MetricScores:
    """Evaluate a RAG response using LLM-as-judge + embedding similarity.

    Uses Claude to judge faithfulness, context precision/recall, and answer correctness.
    Uses embedding similarity as an additional signal for answer correctness.
    Citation accuracy is computed separately for graph-RAG responses.
    """
    context_text = "\n---\n".join(context_chunks) if context_chunks else "(no context)"

    user_message = (
        f"## Question\n{question}\n\n"
        f"## Generated Answer\n{answer}\n\n"
        f"## Expected Answer\n{expected_answer}\n\n"
        f"## Retrieved Context\n{context_text}\n\n"
        "Evaluate the generated answer against the expected answer and context."
    )

    try:
        result = extract_structured(
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
            tools=[JUDGE_TOOL],
            tool_choice={"type": "tool", "name": "evaluate_response"},
        )
    except Exception as exc:
        logger.error("LLM judge failed: %s", exc)
        return MetricScores(
            faithfulness=0.0,
            context_precision=0.0,
            context_recall=0.0,
            answer_correctness=0.0,
        )

    # Embedding-based answer similarity as additional signal
    emb_similarity = cosine_similarity(
        embed_text(answer), embed_text(expected_answer),
    )

    # Blend LLM judge score with embedding similarity (70/30 split)
    llm_correctness = _clamp(result.get("answer_correctness", 0.0))
    blended_correctness = 0.7 * llm_correctness + 0.3 * emb_similarity

    # Citation accuracy (if citations provided)
    citation_accuracy = None
    if citations:
        citation_accuracy = _compute_citation_accuracy(
            citations, expected_answer, context_chunks,
        )

    scores = MetricScores(
        faithfulness=_clamp(result.get("faithfulness", 0.0)),
        context_precision=_clamp(result.get("context_precision", 0.0)),
        context_recall=_clamp(result.get("context_recall", 0.0)),
        answer_correctness=_clamp(blended_correctness),
        citation_accuracy=citation_accuracy,
    )

    logger.debug("Evaluation scores: %s", scores)
    return scores


def _compute_citation_accuracy(
    citations: list[dict],
    expected_answer: str,
    context_chunks: list[str],
) -> float:
    """Compute what fraction of citations actually support the answer."""
    if not citations:
        return 0.0

    expected_lower = expected_answer.lower()
    supported = 0

    for cite in citations:
        source_text = cite.get("source_text", "").lower()
        entity_name = cite.get("entity_name", "").lower()

        # A citation is "accurate" if its source text or entity name
        # overlaps meaningfully with the expected answer
        if (entity_name and entity_name in expected_lower) or \
           any(word in source_text for word in expected_lower.split()
               if len(word) > 4):
            supported += 1

    return min(supported / len(citations), 1.0)


def _clamp(value: float) -> float:
    """Clamp a value between 0.0 and 1.0."""
    return max(0.0, min(1.0, float(value)))
