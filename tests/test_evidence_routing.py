from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from razorpay_ai.answer_pipeline import AnswerPipeline


class Chunk:
    def to_dict(self):
        return {"text": "Evidence", "score": 0.9, "source_id": "s", "title": "Payment Links", "url": "u", "product": "p", "topic": "t", "provenance_status": "verified", "chunk_id": "c"}


class Retriever:
    def __init__(self, chunks): self.chunks = chunks
    def retrieve(self, text): return self.chunks


class NLU:
    def predict(self, text):
        return {"domain": "RAZORPAY", "domain_confidence": 0.3, "intent_confidence": 0.1}


class RAG:
    def __init__(self, chunks):
        self.nlu_pipeline = NLU(); self.semantic_retriever = Retriever(chunks); self.tfidf_retriever = None


class Gemini:
    def generate(self, system, prompt): return "Grounded answer"


def test_verified_evidence_allows_low_confidence_razorpay_question_to_proceed():
    result = AnswerPipeline(RAG([Chunk()]), Gemini()).answer("How do I create a payment link?")
    assert result["status"] == "ANSWERED"


def test_low_confidence_razorpay_question_still_clarifies_without_evidence():
    result = AnswerPipeline(RAG([]), Gemini()).answer("How do I create a payment link?")
    assert result["status"] == "CLARIFY"
