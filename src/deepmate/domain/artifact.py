"""Artifact domain objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Reference to a produced artifact, not the artifact content itself."""

    uri: str
    title: str = ""

    def is_ready(self) -> bool:
        """Return whether the artifact has a usable location."""
        return bool(self.uri.strip())
