from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from razorpay_ai.conversation import ConversationMemory


def test_follow_up_is_enriched_from_real_prior_source():
    memory = ConversationMemory()
    memory.record(
        "demo",
        "How do I create a Payment Link?",
        {"status": "ANSWERED", "sources": [{"title": "Create a Payment Link"}]},
    )
    query, applied = memory.contextualize("demo", "What is its expiry?")
    assert applied is True
    assert "Create a Payment Link" in query
    assert query.endswith("What is its expiry?")


def test_unrelated_or_unknown_conversation_is_not_expanded():
    memory = ConversationMemory()
    assert memory.contextualize("unknown", "What is its expiry?") == ("What is its expiry?", False)
    memory.record("demo", "What is a Payment Link?", {"status": "CLARIFY", "sources": []})
    assert memory.contextualize("demo", "How do I create a Payment Link?") == (
        "How do I create a Payment Link?",
        False,
    )
