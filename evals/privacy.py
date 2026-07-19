"""Privacy and secret scanning for committed assets and local run artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PrivacyFinding:
    kind: str
    source: str
    message: str

    def to_record(self) -> dict[str, str]:
        return {"kind": self.kind, "source": self.source, "message": self.message}


_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "api_key",
        re.compile(
            r"(?i)['\"]?\b(?:api[_ -]?key|access[_ -]?key|client[_ -]?secret)"
            r"\b['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9_./+\-=]{12,}"
        ),
        "credential-like API key assignment",
    ),
    (
        "provider_token",
        re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{16,}\b"),
        "provider-style token",
    ),
    (
        "bearer_token",
        re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"),
        "bearer token",
    ),
    (
        "password",
        re.compile(
            r"(?i)['\"]?\b(?:password|passwd|pwd)\b['\"]?\s*[:=]"
            r"\s*['\"]?[^\s'\"]{8,}"
        ),
        "password assignment",
    ),
    (
        "account",
        re.compile(
            r"(?i)['\"]?\b(?:account|username|user[_ -]?id|tenant[_ -]?id)"
            r"\b['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9_.@+-]{6,}"
        ),
        "account identifier",
    ),
    (
        "session_id",
        re.compile(
            r"(?i)['\"]?\bsession[_ -]?id\b['\"]?\s*[:=]"
            r"\s*['\"]?[A-Za-z0-9_-]{8,}"
        ),
        "session identifier",
    ),
    (
        "private_project",
        re.compile(
            r"(?i)['\"]?\b(?:private[_ -]?project|project[_ -]?name)"
            r"\b['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9_.-]{6,}"
        ),
        "private project identifier",
    ),
    (
        "connection_string",
        re.compile(
            r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://"
            r"[^\s/]+:[^\s@]+@"
        ),
        "connection string with embedded credentials",
    ),
    (
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "private key material",
    ),
    (
        "ipv4",
        re.compile(
            r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)"
            r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])"
        ),
        "literal IPv4 address",
    ),
    (
        "email",
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        "email address",
    ),
    (
        "user_path",
        re.compile(r"(?:/Users/[^/\s]+|/home/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+)"),
        "user-specific filesystem path",
    ),
    (
        "reasoning_scratchpad",
        re.compile(
            r"(?i)(?:<thinking>|</thinking>|assistant_reasoning\s*[:=]|"
            r"reasoning_scratchpad\s*[:=])"
        ),
        "private reasoning scratchpad",
    ),
)


def scan_text(
    text: str,
    *,
    source: str = "<memory>",
    canaries: Iterable[str] = (),
) -> tuple[PrivacyFinding, ...]:
    """Return redaction-safe findings without echoing the sensitive value."""
    findings: list[PrivacyFinding] = []
    for kind, pattern, message in _PATTERNS:
        if pattern.search(text):
            findings.append(PrivacyFinding(kind, source, message))
    for canary in canaries:
        if canary and canary in text:
            findings.append(
                PrivacyFinding("canary", source, "runtime privacy marker escaped isolation")
            )
    return tuple(findings)


def scan_value(
    value: object,
    *,
    source: str = "<value>",
    canaries: Iterable[str] = (),
    replacements: Mapping[str, str] | None = None,
) -> tuple[PrivacyFinding, ...]:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    for raw, label in (replacements or {}).items():
        if raw:
            text = text.replace(raw, label)
    return scan_text(text, source=source, canaries=canaries)


def scan_paths(
    paths: Iterable[Path],
    *,
    canaries: Iterable[str] = (),
) -> tuple[PrivacyFinding, ...]:
    findings: list[PrivacyFinding] = []
    candidates = sorted({item.absolute() for item in paths})
    if len(candidates) > 10_000:
        return (
            PrivacyFinding(
                "asset_budget",
                "<asset-set>",
                "privacy scan exceeded the file-count budget",
            ),
        )
    total_bytes = 0
    for path in candidates:
        if path.is_symlink():
            findings.append(
                PrivacyFinding("symlink", str(path), "scanned assets may not be symlinks")
            )
            continue
        if not path.is_file():
            continue
        try:
            total_bytes += path.stat().st_size
        except OSError:
            continue
        if total_bytes > 50_000_000:
            findings.append(
                PrivacyFinding(
                    "asset_budget",
                    "<asset-set>",
                    "privacy scan exceeded the byte budget",
                )
            )
            break
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        findings.extend(scan_text(content, source=str(path), canaries=canaries))
    return tuple(findings)


def sanitize_value(
    value: object,
    *,
    replacements: Mapping[str, str] | None = None,
    canaries: Iterable[str] = (),
) -> object:
    """Redact known trial roots and sensitive values before persistence."""
    replacement_map = {
        key: label
        for key, label in (replacements or {}).items()
        if key
    }
    canary_values = tuple(canary for canary in canaries if canary)

    def clean(item: object) -> object:
        if isinstance(item, str):
            text = item
            for raw, label in replacement_map.items():
                text = text.replace(raw, label)
            for canary in canary_values:
                text = text.replace(canary, "<redacted-private-marker>")
            for _kind, pattern, _message in _PATTERNS:
                text = pattern.sub("<redacted>", text)
            return text
        if isinstance(item, Mapping):
            return {str(key): clean(child) for key, child in item.items()}
        if isinstance(item, (list, tuple, set)):
            return [clean(child) for child in item]
        return item

    return clean(value)


def committed_asset_paths(evals_root: Path) -> tuple[Path, ...]:
    """Return every committed eval asset except the scanner's own pattern literals."""
    paths: list[Path] = []
    for path in evals_root.rglob("*"):
        if (
            not path.is_file()
            or path.name == "privacy.py"
            or "__pycache__" in path.parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        paths.append(path)
    return tuple(paths)
