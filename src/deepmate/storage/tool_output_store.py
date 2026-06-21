"""Session-scoped raw tool output storage."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepmate.storage.atomic import atomic_write_json

TOOL_OUTPUT_REF_PREFIX = "out_"
_REF_RE = re.compile(r"^out_[0-9a-f]{12,32}$")


@dataclass(frozen=True, slots=True)
class ToolOutputRecord:
    """One raw tool output saved for session-local retrieval."""

    ref: str
    session_id: str
    profile: str
    created_at: str
    tool_name: str
    tool_source: str
    content_kind: str
    estimated_tokens: int
    sha256: str
    content: str


class ToolOutputStore:
    """Store raw tool outputs under one profile/session boundary."""

    def __init__(self, root: str | Path, profile: str, session_id: str) -> None:
        clean_profile = _clean_segment(profile, fallback="default")
        clean_session_id = _clean_segment(session_id, fallback="unknown-session")
        self._root = Path(root).resolve()
        self._profile = clean_profile
        self._session_id = clean_session_id
        self._session_dir = (
            self._root / "tool_outputs" / clean_profile / clean_session_id
        ).resolve()
        if not _is_relative_to(self._session_dir, self._root):
            raise ValueError("tool output store path escaped data root")

    @classmethod
    def in_data_dir(
        cls,
        data_dir: str | Path,
        profile: str,
        session_id: str,
    ) -> "ToolOutputStore":
        """Create a store rooted at Deepmate's runtime data directory."""
        return cls(data_dir, profile=profile, session_id=session_id)

    @property
    def profile(self) -> str:
        """Return the profile boundary for this store."""
        return self._profile

    @property
    def session_id(self) -> str:
        """Return the session boundary for this store."""
        return self._session_id

    def save(
        self,
        *,
        tool_name: str,
        tool_source: str,
        content_kind: str,
        content: str,
        estimated_tokens: int,
        request_id: str = "",
    ) -> ToolOutputRecord:
        """Save raw output text and return its retrieval record."""
        text = content if isinstance(content, str) else str(content)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        ref = _make_ref(
            self._session_id,
            tool_name,
            request_id,
            content_hash,
        )
        record = ToolOutputRecord(
            ref=ref,
            session_id=self._session_id,
            profile=self._profile,
            created_at=datetime.now(UTC).isoformat(),
            tool_name=tool_name.strip(),
            tool_source=tool_source.strip() or "unknown",
            content_kind=content_kind.strip() or "plain",
            estimated_tokens=max(0, int(estimated_tokens)),
            sha256=content_hash,
            content=text,
        )
        atomic_write_json(self._path_for_ref(ref), _record_payload(record))
        return record

    def load(self, ref: str) -> ToolOutputRecord | None:
        """Load a raw output by ref within this store's session boundary."""
        clean_ref = ref.strip()
        if not _REF_RE.match(clean_ref):
            return None
        path = self._path_for_ref(clean_ref)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        record = _record_from_payload(payload)
        if record.profile != self._profile or record.session_id != self._session_id:
            return None
        return record

    def has_records(self) -> bool:
        """Return whether this session has any saved raw tool outputs."""
        if not self._session_dir.exists():
            return False
        return any(path.is_file() for path in self._session_dir.glob("out_*.json"))

    def refs(self) -> tuple[str, ...]:
        """Return saved refs under this session boundary."""
        if not self._session_dir.exists():
            return ()
        refs: list[str] = []
        for path in self._session_dir.glob("out_*.json"):
            ref = path.stem
            if _REF_RE.match(ref):
                refs.append(ref)
        return tuple(sorted(refs))

    def prune_unreferenced(self, keep_refs: Iterable[str]) -> int:
        """Delete saved outputs not present in the provided current ref set."""
        keep = {ref.strip() for ref in keep_refs if _REF_RE.match(str(ref).strip())}
        deleted = 0
        for ref in self.refs():
            if ref in keep:
                continue
            path = self._path_for_ref(ref)
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            deleted += 1
        return deleted

    def _path_for_ref(self, ref: str) -> Path:
        if not _REF_RE.match(ref):
            raise ValueError(f"invalid tool output ref: {ref}")
        path = (self._session_dir / f"{ref}.json").resolve()
        if not _is_relative_to(path, self._session_dir):
            raise ValueError("tool output ref path escaped session directory")
        return path


def _make_ref(
    session_id: str,
    tool_name: str,
    request_id: str,
    content_hash: str,
) -> str:
    seed = "\0".join((session_id.strip(), tool_name.strip(), request_id.strip(), content_hash))
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"{TOOL_OUTPUT_REF_PREFIX}{digest[:16]}"


def tool_output_ref_value(value: str) -> str:
    """Return a valid tool output ref from a trace/transcript ref value."""
    clean = value.strip()
    if not clean.startswith("tool_output_ref="):
        return ""
    ref = clean.removeprefix("tool_output_ref=").strip()
    if not _REF_RE.match(ref):
        return ""
    return ref


def _record_payload(record: ToolOutputRecord) -> dict[str, object]:
    return {
        "ref": record.ref,
        "session_id": record.session_id,
        "profile": record.profile,
        "created_at": record.created_at,
        "tool_name": record.tool_name,
        "tool_source": record.tool_source,
        "content_kind": record.content_kind,
        "estimated_tokens": record.estimated_tokens,
        "sha256": record.sha256,
        "content": record.content,
    }


def _record_from_payload(payload: dict[str, Any]) -> ToolOutputRecord:
    return ToolOutputRecord(
        ref=_text(payload.get("ref")),
        session_id=_text(payload.get("session_id")),
        profile=_text(payload.get("profile")),
        created_at=_text(payload.get("created_at")),
        tool_name=_text(payload.get("tool_name")),
        tool_source=_text(payload.get("tool_source")),
        content_kind=_text(payload.get("content_kind")),
        estimated_tokens=_int(payload.get("estimated_tokens")),
        sha256=_text(payload.get("sha256")),
        content=_text(payload.get("content")),
    )


def _clean_segment(value: str, *, fallback: str) -> str:
    clean = str(value).strip()
    if not clean:
        clean = fallback
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", clean).strip(".-") or fallback


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
