"""Low-frequency learning suggestions for the desktop pet."""

from __future__ import annotations

import html
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from deepmate.domain import Message, MessageRole
from deepmate.pet.state import PetProfile
from deepmate.providers import ModelConversationItem, ModelProvider, ModelRequest
from deepmate.tools.url_safety import validate_public_url

MAX_LEARNING_SOURCE_BYTES = 500_000


@dataclass(frozen=True, slots=True)
class LearningCandidate:
    """One external item that may be useful for the current work."""

    title: str
    url: str
    summary: str = ""
    source: str = ""

    def is_ready(self) -> bool:
        """Return whether this candidate can be shown to the user."""
        return bool(self.title.strip() and self.url.strip())


@dataclass(frozen=True, slots=True)
class LearningSuggestion:
    """One short learning-mode suggestion card."""

    title: str
    url: str
    summary: str
    reason: str = ""
    source: str = "fallback"


def fetch_learning_candidates(
    url: str,
    *,
    timeout: int = 10,
    limit: int = 20,
) -> tuple[LearningCandidate, ...]:
    """Fetch a whitelist source and extract simple link candidates."""
    validate_public_url(url)
    request = Request(url, headers={"User-Agent": "DeepmatePet/0.1"})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(MAX_LEARNING_SOURCE_BYTES + 1)
    if len(raw) > MAX_LEARNING_SOURCE_BYTES:
        raise ValueError(f"learning source exceeds {MAX_LEARNING_SOURCE_BYTES} bytes")
    text = raw.decode("utf-8", errors="replace")
    return _links_from_html(text, base_url=url, source=url, limit=limit)


def generate_learning_suggestion(
    candidates: Iterable[LearningCandidate],
    *,
    interest_tags: Iterable[str] = (),
    current_work_summary: str = "",
    profile: PetProfile | None = None,
    provider: ModelProvider | None = None,
    model: str = "",
) -> LearningSuggestion | None:
    """Pick one candidate and generate a short explanation."""
    ranked = rank_learning_candidates(candidates, interest_tags=interest_tags)
    if not ranked:
        return None
    candidate = ranked[0]
    fallback = _fallback_suggestion(candidate, interest_tags)
    if provider is None or not model.strip():
        return fallback
    try:
        response = provider.complete(
            _learning_request(
                ranked[:10],
                interest_tags=tuple(interest_tags),
                current_work_summary=current_work_summary,
                profile=profile,
                model=model,
            )
        )
    except Exception:
        return fallback
    text = " ".join(response.content.split())
    if not text:
        return fallback
    return LearningSuggestion(
        title=candidate.title,
        url=candidate.url,
        summary=text[:220],
        reason="model_selected",
        source="llm",
    )


def rank_learning_candidates(
    candidates: Iterable[LearningCandidate],
    *,
    interest_tags: Iterable[str] = (),
) -> tuple[LearningCandidate, ...]:
    """Return candidates sorted by simple local tag overlap."""
    tags = tuple(tag.strip().lower() for tag in interest_tags if tag.strip())
    ready = tuple(candidate for candidate in candidates if candidate.is_ready())
    if not tags:
        return ready
    scored = tuple((_candidate_score(candidate, tags), candidate) for candidate in ready)
    return tuple(
        candidate
        for score, candidate in sorted(scored, key=lambda item: item[0], reverse=True)
        if score > 0
    ) or ready[:1]


def interest_tags_from_texts(values: Iterable[str]) -> tuple[str, ...]:
    """Extract coarse project-interest tags without exposing raw text."""
    text = " ".join(value.lower() for value in values if value.strip())
    tag_map: Mapping[str, tuple[str, ...]] = {
        "ai-agent": ("agent", "multi-agent", "autonomous"),
        "coding-agent": ("coding agent", "code agent", "developer tool"),
        "context-compression": ("context", "compression", "compaction"),
        "mcp": ("mcp", "model context protocol"),
        "sandbox": ("sandbox", "permission", "approval"),
        "subagent": ("subagent", "delegate", "reflect"),
        "cost-control": ("cost", "cache", "token"),
        "desktop-companion": ("desktop pet", "companion", "buddy"),
    }
    return tuple(
        tag
        for tag, needles in tag_map.items()
        if any(needle in text for needle in needles)
    )


def _learning_request(
    candidates: tuple[LearningCandidate, ...],
    *,
    interest_tags: tuple[str, ...],
    current_work_summary: str,
    profile: PetProfile | None,
    model: str,
) -> ModelRequest:
    system = Message(
        role=MessageRole.SYSTEM,
        content=(
            "You choose one useful external item for a desktop pet learning card. "
            "Return one concise zh-CN explanation, max 120 Chinese characters. "
            "Do not quote long copyrighted text. Do not mention private memory."
        ),
    )
    body = {
        "pet": profile.species if profile else "",
        "style": profile.style if profile else "",
        "current_work_summary": current_work_summary[:300],
        "interest_tags": list(interest_tags),
        "candidates": [
            {
                "title": candidate.title[:160],
                "summary": candidate.summary[:220],
                "url": candidate.url,
            }
            for candidate in candidates
        ],
    }
    user = Message(role=MessageRole.USER, content=json.dumps(body, ensure_ascii=False))
    return ModelRequest(
        model=model.strip(),
        conversation=(
            ModelConversationItem.from_message(system),
            ModelConversationItem.from_message(user),
        ),
        options={"max_tokens": 120},
    )


def _fallback_suggestion(
    candidate: LearningCandidate,
    interest_tags: Iterable[str],
) -> LearningSuggestion:
    tags = ", ".join(tag for tag in interest_tags if tag)
    reason = f"Related to {tags}." if tags else "May be related to current work."
    summary = candidate.summary.strip() or candidate.title.strip()
    return LearningSuggestion(
        title=candidate.title.strip(),
        url=candidate.url.strip(),
        summary=summary[:220],
        reason=reason,
        source="fallback",
    )


def _candidate_score(candidate: LearningCandidate, tags: tuple[str, ...]) -> int:
    text = f"{candidate.title} {candidate.summary}".lower()
    return sum(1 for tag in tags if tag.replace("-", " ") in text or tag in text)


def _links_from_html(
    text: str,
    *,
    base_url: str,
    source: str,
    limit: int,
) -> tuple[LearningCandidate, ...]:
    seen: set[str] = set()
    results: list[LearningCandidate] = []
    for match in re.finditer(
        r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        href = html.unescape(match.group(1)).strip()
        title = _strip_tags(match.group(2))
        if not href or not title:
            continue
        url = urljoin(base_url, href)
        if url in seen:
            continue
        seen.add(url)
        results.append(LearningCandidate(title=title, url=url, source=source))
        if len(results) >= limit:
            break
    return tuple(results)


def _strip_tags(value: str) -> str:
    clean = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(clean).split())
