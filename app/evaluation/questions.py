"""Gold-standard test question bank for evaluation.

Each question includes: query text, expected answer, expected source sections,
and question category for stratified metric reporting.
"""

from app.models.schema import GoldStandardQA, QuestionCategory

GOLD_STANDARD_QA: list[GoldStandardQA] = [
    # ── Single-hop factual ─────────────────────────────────────────────────
    GoldStandardQA(
        question="What is the patient's blood pressure?",
        expected_answer="The patient's blood pressure is 152/88 mmHg.",
        expected_source_sections=["Objective"],
        category=QuestionCategory.SINGLE_HOP,
    ),
    GoldStandardQA(
        question="What is the patient's heart rate and oxygen saturation?",
        expected_answer="Heart rate is 78 bpm and oxygen saturation (SpO2) is 97%.",
        expected_source_sections=["Objective"],
        category=QuestionCategory.SINGLE_HOP,
    ),
    GoldStandardQA(
        question="What is the patient's age and sex?",
        expected_answer="The patient is a 62-year-old male.",
        expected_source_sections=["Subjective"],
        category=QuestionCategory.SINGLE_HOP,
    ),

    # ── Multi-hop reasoning ────────────────────────────────────────────────
    GoldStandardQA(
        question="What conditions could explain the patient's exertional symptoms given their risk factors?",
        expected_answer=(
            "Stable angina is the likely diagnosis given the exertional pattern "
            "(chest tightness 2-3x/week when walking uphill or carrying groceries, "
            "resolves with rest) combined with multiple risk factors: hypertension, "
            "age 62, and family history of myocardial infarction at age 58."
        ),
        expected_source_sections=["Subjective", "Assessment"],
        category=QuestionCategory.MULTI_HOP,
    ),
    GoldStandardQA(
        question="How does the patient's medication adherence relate to their blood pressure control?",
        expected_answer=(
            "The patient admits to occasionally missing blood pressure medications. "
            "This inconsistent adherence is directly related to the borderline "
            "hypertension control noted in the assessment, with a current reading of 152/88."
        ),
        expected_source_sections=["Subjective", "Assessment", "Objective"],
        category=QuestionCategory.MULTI_HOP,
    ),

    # ── Provenance tracing ─────────────────────────────────────────────────
    GoldStandardQA(
        question="What source section mentions the patient's medication adherence?",
        expected_answer=(
            "The Subjective section mentions that the patient admits occasionally "
            "missing BP meds."
        ),
        expected_source_sections=["Subjective"],
        category=QuestionCategory.PROVENANCE,
    ),
    GoldStandardQA(
        question="Which section of the note contains the family history information?",
        expected_answer=(
            "The Assessment section mentions family history: father had a myocardial "
            "infarction at age 58."
        ),
        expected_source_sections=["Assessment"],
        category=QuestionCategory.PROVENANCE,
    ),

    # ── Relationship queries ───────────────────────────────────────────────
    GoldStandardQA(
        question="What medications are prescribed or planned for the patient's conditions?",
        expected_answer=(
            "The patient is to continue antihypertensives for hypertension (with "
            "reinforced daily adherence) and was provided sublingual nitroglycerin "
            "for exertional chest discomfort. A fasting lipid panel was also ordered."
        ),
        expected_source_sections=["Plan"],
        category=QuestionCategory.RELATIONSHIP,
    ),
    GoldStandardQA(
        question="What procedures are pending or scheduled for the patient?",
        expected_answer=(
            "A treadmill myocardial perfusion scan was completed yesterday but "
            "the report is pending final review. A follow-up visit is scheduled "
            "in 1-2 weeks to review scan results and discuss possible cardiology referral."
        ),
        expected_source_sections=["Objective", "Plan"],
        category=QuestionCategory.RELATIONSHIP,
    ),

    # ── Cross-reference (demographics + SOAP) ─────────────────────────────
    GoldStandardQA(
        question="What is the patient's health card number and postal code?",
        expected_answer="Health card number is 9696178816 and postal code is K7L 3V8.",
        expected_source_sections=["Demographics"],
        category=QuestionCategory.CROSS_REFERENCE,
    ),
    GoldStandardQA(
        question="What is the full name and contact information of the patient?",
        expected_answer=(
            "Peter Julius Fern. Home phone: 613-6565-890, mobile: 647-666-8888, "
            "email: peter.fern@email.com, address: 45 Maple Ave, Toronto, ON K7L 3V8, Canada."
        ),
        expected_source_sections=["Demographics"],
        category=QuestionCategory.CROSS_REFERENCE,
    ),
]
