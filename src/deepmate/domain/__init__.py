"""Domain objects shared by Deepmate modules."""

from deepmate.domain.approval import ApprovalRequest
from deepmate.domain.artifact import ArtifactRef
from deepmate.domain.capability import CapabilityKind, CapabilityRef
from deepmate.domain.errors import ErrorInfo
from deepmate.domain.event import RuntimeEvent
from deepmate.domain.memory import MemoryEntry, MemorySource
from deepmate.domain.message import Message, MessageRole
from deepmate.domain.profile import ProfileRef

__all__ = [
    "ApprovalRequest",
    "ArtifactRef",
    "CapabilityKind",
    "CapabilityRef",
    "ErrorInfo",
    "MemoryEntry",
    "MemorySource",
    "Message",
    "MessageRole",
    "ProfileRef",
    "RuntimeEvent",
]
