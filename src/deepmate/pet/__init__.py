"""Desktop pet companion state and local host helpers."""

from deepmate.pet.copy import PetCopyResult, generate_pet_copy
from deepmate.pet.events import (
    PetAction,
    PetEvent,
    PetSeverity,
    PetVisualState,
    event_for_turn_finished,
    event_for_turn_progress,
    event_for_turn_started,
    event_for_turn_waiting,
)
from deepmate.pet.electron_host import run_pet_host
from deepmate.pet.learning import (
    LearningCandidate,
    LearningSuggestion,
    generate_learning_suggestion,
)
from deepmate.pet.policy import PetDisplayDecision, PetDisplayPolicy
from deepmate.pet.state import (
    PetProfile,
    PetStateStore,
    PetUserAction,
    default_pet_profile,
)

__all__ = [
    "LearningCandidate",
    "LearningSuggestion",
    "PetAction",
    "PetCopyResult",
    "PetDisplayDecision",
    "PetDisplayPolicy",
    "PetEvent",
    "PetProfile",
    "PetSeverity",
    "PetStateStore",
    "PetUserAction",
    "PetVisualState",
    "default_pet_profile",
    "event_for_turn_finished",
    "event_for_turn_progress",
    "event_for_turn_started",
    "event_for_turn_waiting",
    "generate_learning_suggestion",
    "generate_pet_copy",
    "run_pet_host",
]
