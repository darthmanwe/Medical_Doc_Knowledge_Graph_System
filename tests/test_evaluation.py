"""Tests for the evaluation module components."""

import pytest

from app.evaluation.questions import GOLD_STANDARD_QA
from app.evaluation.report import generate_report
from app.models.schema import (
    EvaluationRecord,
    EvaluationReport,
    MetricScores,
    QuestionCategory,
)


class TestGoldStandardQuestions:
    """Validate the question bank structure."""

    def test_minimum_question_count(self):
        assert len(GOLD_STANDARD_QA) >= 10, "Need at least 10 gold-standard questions"

    def test_all_categories_represented(self):
        categories = {qa.category for qa in GOLD_STANDARD_QA}
        assert QuestionCategory.SINGLE_HOP in categories
        assert QuestionCategory.MULTI_HOP in categories
        assert QuestionCategory.PROVENANCE in categories
        assert QuestionCategory.RELATIONSHIP in categories
        assert QuestionCategory.CROSS_REFERENCE in categories

    def test_questions_have_expected_answers(self):
        for qa in GOLD_STANDARD_QA:
            assert len(qa.question) > 10, f"Question too short: {qa.question}"
            assert len(qa.expected_answer) > 10, f"Answer too short for: {qa.question}"
            assert len(qa.expected_source_sections) >= 1

    def test_no_duplicate_questions(self):
        questions = [qa.question for qa in GOLD_STANDARD_QA]
        assert len(questions) == len(set(questions)), "Duplicate questions found"


class TestMetricScores:
    """Test MetricScores model validation."""

    def test_valid_scores(self):
        scores = MetricScores(
            faithfulness=0.85,
            context_precision=0.90,
            context_recall=0.75,
            answer_correctness=0.80,
            citation_accuracy=0.70,
        )
        assert scores.faithfulness == 0.85

    def test_scores_clamped_to_range(self):
        with pytest.raises(Exception):
            MetricScores(
                faithfulness=1.5,  # Over 1.0
                context_precision=0.9,
                context_recall=0.8,
                answer_correctness=0.7,
            )


class TestReportGeneration:
    """Test the report generator with mock data."""

    def test_generates_markdown(self):
        report = EvaluationReport(
            records=[
                EvaluationRecord(
                    question="Test question",
                    category=QuestionCategory.SINGLE_HOP,
                    strategy="vector",
                    answer="Test answer",
                    expected_answer="Expected",
                    scores=MetricScores(
                        faithfulness=0.8,
                        context_precision=0.7,
                        context_recall=0.6,
                        answer_correctness=0.9,
                    ),
                    retrieval_time_ms=100.0,
                    generation_time_ms=200.0,
                ),
                EvaluationRecord(
                    question="Test question",
                    category=QuestionCategory.SINGLE_HOP,
                    strategy="graph",
                    answer="Test graph answer",
                    expected_answer="Expected",
                    scores=MetricScores(
                        faithfulness=0.9,
                        context_precision=0.85,
                        context_recall=0.8,
                        answer_correctness=0.95,
                        citation_accuracy=0.8,
                    ),
                    retrieval_time_ms=150.0,
                    generation_time_ms=250.0,
                ),
            ],
            vector_avg_scores=MetricScores(
                faithfulness=0.8,
                context_precision=0.7,
                context_recall=0.6,
                answer_correctness=0.9,
            ),
            graph_avg_scores=MetricScores(
                faithfulness=0.9,
                context_precision=0.85,
                context_recall=0.8,
                answer_correctness=0.95,
                citation_accuracy=0.8,
            ),
        )

        markdown = generate_report(report)
        assert "# RAG Evaluation Report" in markdown
        assert "Vector RAG" in markdown
        assert "Graph RAG" in markdown
        assert "Key Findings" in markdown
