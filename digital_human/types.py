from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class SPECIAL:
    """Base human attributes (1-10). Deep, stable, hard to change."""

    strength: int
    perception: int
    endurance: int
    charisma: int
    intelligence: int
    agility: int
    luck: int

    def __post_init__(self) -> None:
        for attr in (
            "strength", "perception", "endurance", "charisma",
            "intelligence", "agility", "luck",
        ):
            val = getattr(self, attr)
            if not isinstance(val, int) or not (1 <= val <= 10):
                raise ValueError(
                    f"SPECIAL.{attr} must be int 1-10, got {val!r}"
                )


@dataclass(frozen=True)
class JJDIDTIEBUCKLE:
    """Professional overlay (1-10). Trainable, more malleable than SPECIAL."""

    justice: int
    judgment: int
    dependability: int
    initiative: int
    decisiveness: int
    tact: int
    integrity: int
    enthusiasm: int
    bearing: int
    unselfishness: int
    courage: int
    knowledge: int
    loyalty: int
    endurance: int

    def __post_init__(self) -> None:
        for attr in (
            "justice", "judgment", "dependability", "initiative",
            "decisiveness", "tact", "integrity", "enthusiasm",
            "bearing", "unselfishness", "courage", "knowledge",
            "loyalty", "endurance",
        ):
            val = getattr(self, attr)
            if not isinstance(val, int) or not (1 <= val <= 10):
                raise ValueError(
                    f"JJDIDTIEBUCKLE.{attr} must be int 1-10, got {val!r}"
                )


@dataclass
class Drives:
    """What the human wants. Scored 0-10. Can shift over time."""

    compensation: float = 5.0
    mission: float = 5.0
    recognition: float = 5.0
    security: float = 5.0
    growth: float = 5.0
    comfort: float = 5.0
    social_status: float = 5.0
    autonomy: float = 5.0
    priority_order: list[str] = field(default_factory=lambda: [
        "compensation", "mission", "recognition", "security",
        "growth", "comfort", "social_status", "autonomy",
    ])

    DRIVE_NAMES: frozenset[str] = field(
        default=frozenset({
            "compensation", "mission", "recognition", "security",
            "growth", "comfort", "social_status", "autonomy",
        }),
        init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        for name in self.DRIVE_NAMES:
            val = getattr(self, name)
            if not (0.0 <= val <= 10.0):
                raise ValueError(f"Drive {name} must be 0-10, got {val}")
        if set(self.priority_order) != self.DRIVE_NAMES:
            raise ValueError(
                "priority_order must contain exactly all 8 drive names"
            )

    def dominant(self) -> str:
        best_name = self.priority_order[0]
        best_val = getattr(self, best_name)
        for name in self.priority_order[1:]:
            val = getattr(self, name)
            if val > best_val:
                best_val = val
                best_name = name
        return best_name

    def get(self, name: str) -> float:
        if name not in self.DRIVE_NAMES:
            raise KeyError(f"Unknown drive: {name}")
        return getattr(self, name)

    def top_drives(self, n: int = 3) -> list[tuple[str, float]]:
        scored = [(name, getattr(self, name)) for name in self.priority_order]
        scored.sort(key=lambda x: -x[1])
        return scored[:n]


@dataclass
class Belief:
    value: str
    confidence: float
    formed_on_day: int
    last_updated_day: int


@dataclass
class Concern:
    topic: str
    weight: float
    affects_work: bool


@dataclass
class InnerWorld:
    beliefs: dict[str, Belief] = field(default_factory=dict)
    feelings: dict[str, float] = field(default_factory=dict)
    personal_concerns: list[Concern] = field(default_factory=list)

    def update_feeling(self, key: str, delta: float) -> None:
        current = self.feelings.get(key, 0.0)
        self.feelings[key] = max(-1.0, min(1.0, current + delta))

    def set_belief(
        self, topic: str, value: str, confidence: float, sim_day: int,
    ) -> None:
        existing = self.beliefs.get(topic)
        if existing is not None:
            existing.value = value
            existing.confidence = confidence
            existing.last_updated_day = sim_day
        else:
            self.beliefs[topic] = Belief(value, confidence, sim_day, sim_day)


@dataclass
class Episode:
    sim_day: int
    event_type: str
    content: str
    emotional_valence: float
    people_involved: list[str] = field(default_factory=list)
    lesson: str | None = None
    tags: list[str] = field(default_factory=list)
    raw_impulse: str | None = None


@dataclass
class ExchangeEntry:
    cycle: int
    a_sent_system: str
    a_sent_trait: str
    a_sent_raw: int
    a_sent_filtered: int
    b_sent_system: str
    b_sent_trait: str
    b_sent_raw: int
    b_sent_filtered: int
    combined_score: int = 0

    def __post_init__(self):
        self.combined_score = self.a_sent_filtered + self.b_sent_filtered


@dataclass
class ConversationRecord:
    sim_day: int
    reason: str
    cycles: int
    exchange_log: list[ExchangeEntry] = field(default_factory=list)
    relationship_delta: float = 0.0
    total_score: float = 0.0


@dataclass
class Memory:
    episodes: deque[Episode] = field(
        default_factory=lambda: deque(maxlen=500)
    )
    long_term: list[Episode] = field(default_factory=list)
    conversation_logs: dict[str, list[ConversationRecord]] = field(
        default_factory=dict
    )

    LONG_TERM_MAX: int = field(default=50, init=False, repr=False)
    HIGH_IMPACT_THRESHOLD: float = field(default=0.6, init=False, repr=False)

    def record(self, episode: Episode) -> None:
        self.episodes.append(episode)
        if (
            abs(episode.emotional_valence) >= self.HIGH_IMPACT_THRESHOLD
            and len(self.long_term) < self.LONG_TERM_MAX
        ):
            self.long_term.append(episode)

    def recent(self, n: int = 10) -> list[Episode]:
        items = list(self.episodes)
        return items[-n:] if len(items) >= n else items

    def search(self, tag: str) -> list[Episode]:
        results = [e for e in self.episodes if tag in e.tags]
        results.extend(e for e in self.long_term if tag in e.tags and e not in results)
        return results

    def log_conversation(self, other_id: str, record: ConversationRecord) -> None:
        if other_id not in self.conversation_logs:
            self.conversation_logs[other_id] = []
        self.conversation_logs[other_id].append(record)


@dataclass
class Relationship:
    target_id: str
    trust: float = 0.5
    respect: float = 0.5
    cooperation: float = 0.5
    familiarity: float = 0.0
    warmth: float = 0.5
    authority: float = 0.0
    gratitude: float = 0.0
    opinion: str = ""
    conversation_history_score: float = 0.0
    last_interaction_day: int = 0
    behavioral_adjustments: dict[str, float] = field(default_factory=dict)


class CopingStyle(Enum):
    PUSH_THROUGH = "push_through"
    WITHDRAW = "withdraw"
    EXTERNALIZE = "externalize"
    SEEK_HELP = "seek_help"


@dataclass
class StressState:
    score: float = 0.0
    trajectory: str = "stable"
    sources: dict[str, float] = field(default_factory=dict)
    coping_style: CopingStyle = CopingStyle.PUSH_THROUGH
    threshold: float = 70.0
    recovery_rate: float = 2.0

    def is_critical(self) -> bool:
        return self.score >= self.threshold

    def clamp(self) -> None:
        self.score = max(0.0, min(100.0, self.score))


@dataclass
class Task:
    task_id: str
    description: str
    complexity: float = 1.0
    importance: float = 5.0
    influence: float = 0.0
    deadline_day: int | None = None
    assigned_day: int = 0
    requirements: dict[str, int] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    human_id: str
    sim_day: int
    quality: float
    speed: float
    errors: int
    completed: bool
    trait_match: float
    details: str


@dataclass
class WorkContext:
    role: str = ""
    responsibilities: list[str] = field(default_factory=list)
    hours_per_week: float = 40.0
    compensation: dict[str, Any] = field(default_factory=dict)
    task_queue: deque[Task] = field(default_factory=deque)
    capacity_threshold: int = 5
    skills: list[str] = field(default_factory=list)
    tenure_days: int = 0
    performance_history: list[float] = field(default_factory=list)
    authority_level: int = 0
    reports_to: str | None = None


class ActionType(Enum):
    PROCESS_TASK = "process_task"
    REST = "rest"
    PLAN = "plan"
    ADJUST_EFFORT = "adjust_effort"
    DELEGATE = "delegate"
    ESCALATE = "escalate"
    REQUEST_HELP = "request_help"
    CONFRONT = "confront"
    OFFER_HELP = "offer_help"
    SOCIALIZE = "socialize"
    CUT_CORNERS = "cut_corners"
    REDUCE_EFFORT = "reduce_effort"
    TAKE_INITIATIVE = "take_initiative"
    PUSH_THROUGH = "push_through"


SPECIAL_TRAIT_NAMES: frozenset[str] = frozenset({
    "strength", "perception", "endurance", "charisma",
    "intelligence", "agility", "luck",
})

SOCIAL_ACTIONS: frozenset[ActionType] = frozenset({
    ActionType.DELEGATE,
    ActionType.ESCALATE,
    ActionType.REQUEST_HELP,
    ActionType.CONFRONT,
    ActionType.OFFER_HELP,
    ActionType.SOCIALIZE,
})


@dataclass
class Option:
    action_type: ActionType
    score: float = 0.0
    reasoning: str = ""
    target_id: str | None = None
    task_id: str | None = None


@dataclass
class Action:
    action_type: ActionType
    reasoning: str
    target_id: str | None = None
    task_id: str | None = None
    sim_day: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    enabled: bool = False
    model: str = "qwen2.5:7b-instruct"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.9
    max_tokens: int = 150
