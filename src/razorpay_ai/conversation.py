"""Small, process-local conversation memory for follow-up RAG questions.

This intentionally stores no data outside the running process.  It is a bounded
convenience layer, not a replacement for a durable chat store: callers may pass a
conversation id and receive a self-contained retrieval query when a short
pronoun-led follow-up needs the topic of the preceding supported turn.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import re


_FOLLOW_UP_PATTERN = re.compile(
    r"\b(it|its|they|them|their|that|those|this|these|one|previous|above)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConversationTurn:
    query: str
    status: str
    topic_hint: str | None


class ConversationMemory:
    """Bounded in-memory turns, isolated by a caller-supplied conversation id."""

    def __init__(self, max_turns: int = 6):
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self._turns: dict[str, deque[ConversationTurn]] = defaultdict(lambda: deque(maxlen=max_turns))

    def contextualize(self, conversation_id: str | None, query: str) -> tuple[str, bool]:
        """Return a retrieval query enriched only for an evident follow-up."""
        if not conversation_id or not _FOLLOW_UP_PATTERN.search(query):
            return query, False
        turns = self._turns.get(conversation_id)
        if not turns:
            return query, False
        prior = next((turn for turn in reversed(turns) if turn.topic_hint), None)
        if prior is None:
            return query, False
        return f"Razorpay context: {prior.topic_hint}. Follow-up question: {query}", True

    def record(self, conversation_id: str | None, query: str, result: dict) -> None:
        """Remember only turns with real retrieved documentation as topic evidence."""
        if not conversation_id:
            return
        sources = result.get("sources") or []
        topic_hint = None
        if sources:
            source = sources[0]
            topic_hint = source.get("title") or source.get("topic")
        self._turns[conversation_id].append(
            ConversationTurn(query=query, status=result["status"], topic_hint=topic_hint)
        )
