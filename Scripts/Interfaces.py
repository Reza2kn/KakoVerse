from abc import ABC, abstractmethod
from typing import Dict, Any

class EmpathyVerifier(ABC):
    @abstractmethod
    def __call__(self, context: Dict[str, Any], reply: str) -> float:
        """Return 0..1 empathy score using your trained RM (from rubric pairs)."""
        pass

class SafetyVerifier(ABC):
    @abstractmethod
    def __call__(self, reply: str) -> bool:
        """Return True if any hard rule is violated."""
        pass

class PresenceVerifier(ABC):
    @abstractmethod
    def __call__(self, history: list[str]) -> float:
        """0..1 ratio of supportive vs referral tokens over last 3 turns."""
        pass

class StabilizationVerifier(ABC):
    @abstractmethod
    def __call__(self, prev_user: str, curr_user: str) -> float:
        """0..1 proxy of de-escalation (self-reports, prosody, lexical calm)."""
        pass
