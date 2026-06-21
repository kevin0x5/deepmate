"""Profile domain objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProfileRef:
    """Reference to a Deepmate profile directory."""

    name: str
    uri: str
    global_uri: str = ""
    project_uri: str = ""

    def is_ready(self) -> bool:
        """Return whether the profile can be resolved by a loader."""
        return bool(self.name.strip() and self.uri.strip())
