"""Conversation protocol — score exchange between two Digital Humans.

Spec §4. SPECIAL scores come through direct. JJDIDTIEBUCKLE scores
get +3 filter (capped at 10, overflow to receiver's running total).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

from digital_human.types import (
    Belief,
    ConversationRecord,
    Episode,
    ExchangeEntry,
    Relationship,
)

if TYPE_CHECKING:
    from digital_human.human import DigitalHuman


REASON_CYCLES: dict[str, tuple[int, int]] = {
    "task_handoff": (1, 2),
    "help_request": (2, 4),
    "monday_sync": (5, 8),
    "team_meeting": (5, 8),
    "deep_collaboration": (8, 15),
    "social": (2, 5),
    "confrontation": (3, 6),
    "escalation": (2, 4),
    "offer_help": (2, 4),
    "delegation": (1, 3),
}

TASK_TRAITS_JJD = ["knowledge", "dependability", "initiative", "judgment", "decisiveness"]
TASK_TRAITS_SPECIAL = ["intelligence", "agility"]

SOCIAL_TRAITS_JJD = ["tact", "enthusiasm", "unselfishness", "loyalty"]
SOCIAL_TRAITS_SPECIAL = ["charisma", "perception"]

CONFLICT_TRAITS_JJD = ["courage", "justice", "bearing", "integrity"]
CONFLICT_TRAITS_SPECIAL = ["strength", "endurance"]

REASON_TRAIT_POOLS: dict[str, tuple[list[str], list[str]]] = {
    "task_handoff": (TASK_TRAITS_JJD, TASK_TRAITS_SPECIAL),
    "help_request": (TASK_TRAITS_JJD + ["unselfishness", "tact"], TASK_TRAITS_SPECIAL + ["charisma"]),
    "monday_sync": (TASK_TRAITS_JJD + SOCIAL_TRAITS_JJD, TASK_TRAITS_SPECIAL + SOCIAL_TRAITS_SPECIAL),
    "team_meeting": (TASK_TRAITS_JJD + SOCIAL_TRAITS_JJD, TASK_TRAITS_SPECIAL + SOCIAL_TRAITS_SPECIAL),
    "deep_collaboration": (TASK_TRAITS_JJD + ["enthusiasm", "loyalty"], TASK_TRAITS_SPECIAL),
    "social": (SOCIAL_TRAITS_JJD, SOCIAL_TRAITS_SPECIAL),
    "confrontation": (CONFLICT_TRAITS_JJD, CONFLICT_TRAITS_SPECIAL),
    "escalation": (CONFLICT_TRAITS_JJD + ["judgment", "knowledge"], CONFLICT_TRAITS_SPECIAL + ["intelligence"]),
    "offer_help": (["unselfishness", "knowledge", "enthusiasm", "tact"], ["charisma", "intelligence"]),
    "delegation": (TASK_TRAITS_JJD + ["tact", "bearing"], TASK_TRAITS_SPECIAL + ["charisma"]),
}

DEFAULT_TRAIT_POOLS = (
    TASK_TRAITS_JJD + SOCIAL_TRAITS_JJD,
    TASK_TRAITS_SPECIAL + SOCIAL_TRAITS_SPECIAL,
)


@dataclass
class ConversationResult:
    record_a: ConversationRecord
    record_b: ConversationRecord
    relationship_delta_a: float
    relationship_delta_b: float
    overflow_a: int
    overflow_b: int
    tasks_transferred: int


class Conversation:
    """Score exchange protocol between two humans.

    run() is pure — does not mutate either human.
    apply_conversation_result() is the mutation boundary.
    """

    def __init__(
        self,
        human_a: DigitalHuman,
        human_b: DigitalHuman,
        reason: str,
        sim_day: int,
        rng: random.Random | None = None,
    ) -> None:
        self.human_a = human_a
        self.human_b = human_b
        self.reason = reason
        self.sim_day = sim_day
        # Instance RNG (falls back to the module RNG for direct use).
        self._rng = rng if rng is not None else random

        lo, hi = REASON_CYCLES.get(reason, (2, 4))
        self.expected_cycles = self._rng.randint(lo, hi)

        jjd_pool, special_pool = REASON_TRAIT_POOLS.get(reason, DEFAULT_TRAIT_POOLS)
        self.jjd_pool = list(dict.fromkeys(jjd_pool))
        self.special_pool = list(dict.fromkeys(special_pool))

        self.exchange_log: list[ExchangeEntry] = []
        self.overflow_a = 0
        self.overflow_b = 0

    def run(self) -> ConversationResult:
        for cycle in range(1, self.expected_cycles + 1):
            entry = self._exchange_cycle(cycle)
            self.exchange_log.append(entry)

        delta_a = self._compute_relationship_delta(perspective="a")
        delta_b = self._compute_relationship_delta(perspective="b")

        record_a = ConversationRecord(
            sim_day=self.sim_day,
            reason=self.reason,
            cycles=self.expected_cycles,
            exchange_log=list(self.exchange_log),
            relationship_delta=delta_a,
            total_score=sum(e.combined_score for e in self.exchange_log),
        )
        record_b = ConversationRecord(
            sim_day=self.sim_day,
            reason=self.reason,
            cycles=self.expected_cycles,
            exchange_log=list(self.exchange_log),
            relationship_delta=delta_b,
            total_score=sum(e.combined_score for e in self.exchange_log),
        )

        tasks_transferred = 0
        if self.reason in ("task_handoff", "delegation"):
            tasks_transferred = min(
                len(self.human_a.work_context.task_queue),
                max(1, self.expected_cycles),
            )

        return ConversationResult(
            record_a=record_a,
            record_b=record_b,
            relationship_delta_a=delta_a,
            relationship_delta_b=delta_b,
            overflow_a=self.overflow_a,
            overflow_b=self.overflow_b,
            tasks_transferred=tasks_transferred,
        )

    def _exchange_cycle(self, cycle: int) -> ExchangeEntry:
        a_system, a_trait = self._select_trait(self.human_a, cycle)
        b_system, b_trait = self._select_trait(self.human_b, cycle)

        a_raw = self._get_raw_score(self.human_a, a_system, a_trait)
        b_raw = self._get_raw_score(self.human_b, b_system, b_trait)

        a_filtered, a_overflow = self._apply_filter(a_system, a_raw)
        b_filtered, b_overflow = self._apply_filter(b_system, b_raw)

        self.overflow_b += a_overflow
        self.overflow_a += b_overflow

        return ExchangeEntry(
            cycle=cycle,
            a_sent_system=a_system,
            a_sent_trait=a_trait,
            a_sent_raw=a_raw,
            a_sent_filtered=a_filtered,
            b_sent_system=b_system,
            b_sent_trait=b_trait,
            b_sent_raw=b_raw,
            b_sent_filtered=b_filtered,
            combined_score=a_filtered + b_filtered,
        )

    def _select_trait(self, human: DigitalHuman, cycle: int) -> tuple[str, str]:
        if cycle % 2 == 1 and self.jjd_pool:
            system = "jjdidtiebuckle"
            pool = self.jjd_pool
            weights = [getattr(human.jjdidtiebuckle, t, 5) for t in pool]
        elif self.special_pool:
            system = "special"
            pool = self.special_pool
            weights = [getattr(human.special, t, 5) for t in pool]
        else:
            if self.jjd_pool:
                system = "jjdidtiebuckle"
                pool = self.jjd_pool
                weights = [getattr(human.jjdidtiebuckle, t, 5) for t in pool]
            else:
                return ("special", "charisma")

        trait = self._rng.choices(pool, weights=weights, k=1)[0]
        return (system, trait)

    @staticmethod
    def _get_raw_score(human: DigitalHuman, system: str, trait: str) -> int:
        if system == "special":
            return getattr(human.special, trait)
        else:
            return getattr(human.jjdidtiebuckle, trait)

    @staticmethod
    def _apply_filter(system: str, raw: int) -> tuple[int, int]:
        if system == "special":
            return (raw, 0)
        boosted = raw + 3
        if boosted > 10:
            return (10, boosted - 10)
        return (boosted, 0)

    def _compute_relationship_delta(self, perspective: str) -> float:
        if not self.exchange_log:
            return 0.0

        total_contribution = 0.0
        for entry in self.exchange_log:
            if perspective == "a":
                received_score = entry.b_sent_filtered
                sent_score = entry.a_sent_filtered
            else:
                received_score = entry.a_sent_filtered
                sent_score = entry.b_sent_filtered

            diff = abs(received_score - sent_score)
            similarity = 1.0 - (diff / 10.0)
            combined = (received_score + sent_score) / 20.0
            total_contribution += similarity * combined

        avg_contribution = total_contribution / len(self.exchange_log)
        delta = avg_contribution * 0.15

        if perspective == "a":
            delta += self.overflow_a * 0.01
        else:
            delta += self.overflow_b * 0.01

        return delta


# --- Mutation boundary ---

def apply_conversation_result(
    human_a: DigitalHuman,
    human_b: DigitalHuman,
    result: ConversationResult,
    sim_day: int,
) -> None:
    human_a.memory.log_conversation(human_b.human_id, result.record_a)
    human_b.memory.log_conversation(human_a.human_id, result.record_b)

    _update_relationship(human_a, human_b.human_id, result.relationship_delta_a, result.record_a, sim_day)
    _update_relationship(human_b, human_a.human_id, result.relationship_delta_b, result.record_b, sim_day)

    if result.tasks_transferred > 0:
        for _ in range(result.tasks_transferred):
            if human_a.work_context.task_queue:
                task = human_a.work_context.task_queue.popleft()
                human_b.work_context.task_queue.append(task)

    valence_a = _conversation_valence(result.record_a)
    valence_b = _conversation_valence(result.record_b)

    human_a.memory.record(Episode(
        sim_day=sim_day,
        event_type="conversation",
        content=f"Had a {result.record_a.reason} conversation with {human_b.human_id}",
        emotional_valence=valence_a,
        people_involved=[human_b.human_id],
        tags=["conversation", result.record_a.reason],
    ))
    human_b.memory.record(Episode(
        sim_day=sim_day,
        event_type="conversation",
        content=f"Had a {result.record_b.reason} conversation with {human_a.human_id}",
        emotional_valence=valence_b,
        people_involved=[human_a.human_id],
        tags=["conversation", result.record_b.reason],
    ))

    _update_inner_world_from_conversation(human_a, human_b.human_id, result.record_a, valence_a, sim_day)
    _update_inner_world_from_conversation(human_b, human_a.human_id, result.record_b, valence_b, sim_day)


def _update_relationship(
    human: DigitalHuman,
    other_id: str,
    delta: float,
    record: ConversationRecord,
    sim_day: int,
) -> None:
    if other_id not in human.relationships:
        human.relationships[other_id] = Relationship(target_id=other_id)

    rel = human.relationships[other_id]
    rel.conversation_history_score += record.total_score
    rel.last_interaction_day = sim_day
    rel.familiarity = min(1.0, rel.familiarity + 0.05)

    CAP = 0.05

    reason = record.reason
    if reason in ("task_handoff", "delegation", "deep_collaboration"):
        rel.trust = min(1.0, rel.trust + min(CAP, delta * 0.6))
        rel.cooperation = min(1.0, rel.cooperation + min(CAP, delta * 0.8))
        rel.respect = min(1.0, rel.respect + min(CAP, delta * 0.4))
    elif reason == "social":
        rel.warmth = min(1.0, rel.warmth + min(CAP, delta * 0.8))
        rel.familiarity = min(1.0, rel.familiarity + min(CAP, delta * 0.3))
        rel.trust = min(1.0, rel.trust + min(CAP, delta * 0.3))
    elif reason == "confrontation":
        rel.respect = min(1.0, rel.respect + min(CAP, delta * 0.5))
        rel.warmth = max(0.0, rel.warmth - 0.02)
    elif reason in ("help_request", "offer_help"):
        rel.gratitude = min(1.0, rel.gratitude + min(CAP, delta * 0.6))
        rel.trust = min(1.0, rel.trust + min(CAP, delta * 0.5))
        rel.warmth = min(1.0, rel.warmth + min(CAP, delta * 0.4))
    elif reason in ("monday_sync", "team_meeting"):
        rel.cooperation = min(1.0, rel.cooperation + min(CAP, delta * 0.5))
        rel.familiarity = min(1.0, rel.familiarity + min(CAP, delta * 0.2))
    elif reason == "escalation":
        rel.authority = max(-1.0, min(1.0, rel.authority + 0.05))
        rel.trust = min(1.0, rel.trust + min(CAP, delta * 0.3))
    else:
        rel.trust = min(1.0, rel.trust + min(CAP, delta * 0.3))
        rel.warmth = min(1.0, rel.warmth + min(CAP, delta * 0.3))
        rel.cooperation = min(1.0, rel.cooperation + min(CAP, delta * 0.3))


def _conversation_valence(record: ConversationRecord) -> float:
    if not record.exchange_log:
        return 0.0
    avg_combined = record.total_score / max(1, record.cycles)
    normalized = avg_combined / 20.0
    valence = (normalized - 0.5) * 1.0
    if record.reason == "confrontation":
        valence -= 0.2
    if record.reason == "social":
        valence += 0.1
    return max(-1.0, min(1.0, valence))


def _update_inner_world_from_conversation(
    human: DigitalHuman,
    other_id: str,
    record: ConversationRecord,
    valence: float,
    sim_day: int,
) -> None:
    if valence > 0.1:
        belief_value = "positive"
    elif valence < -0.1:
        belief_value = "negative"
    else:
        belief_value = "neutral"

    confidence = min(1.0, abs(valence) + 0.3)
    human.inner_world.beliefs[f"opinion_of_{other_id}"] = Belief(
        value=belief_value,
        confidence=confidence,
        formed_on_day=sim_day,
        last_updated_day=sim_day,
    )

    human.inner_world.update_feeling("sense_of_belonging", valence * 0.05)
    if record.reason in ("help_request", "offer_help"):
        human.inner_world.update_feeling("job_satisfaction", valence * 0.03)
    if record.reason == "social":
        human.inner_world.update_feeling("motivation", valence * 0.02)


# --- Relationship recomputation from logs (spec §4.5) ---

_REASON_DIMENSION_WEIGHTS: dict[str, dict[str, float]] = {
    "task_handoff":        {"trust": 0.3, "cooperation": 0.5, "respect": 0.2},
    "delegation":          {"trust": 0.3, "cooperation": 0.4, "respect": 0.3},
    "deep_collaboration":  {"trust": 0.3, "cooperation": 0.3, "respect": 0.2, "warmth": 0.2},
    "help_request":        {"trust": 0.3, "gratitude": 0.4, "warmth": 0.3},
    "offer_help":          {"trust": 0.2, "gratitude": 0.3, "warmth": 0.3, "respect": 0.2},
    "social":              {"warmth": 0.5, "trust": 0.2, "familiarity": 0.3},
    "monday_sync":         {"cooperation": 0.4, "familiarity": 0.3, "trust": 0.3},
    "team_meeting":        {"cooperation": 0.4, "familiarity": 0.3, "trust": 0.3},
    "confrontation":       {"respect": 0.5, "trust": 0.2, "warmth": -0.3},
    "escalation":          {"trust": 0.3, "respect": 0.3, "cooperation": 0.4},
}

_DEFAULT_DIMENSION_WEIGHTS: dict[str, float] = {
    "trust": 0.3, "warmth": 0.3, "cooperation": 0.2, "respect": 0.2,
}


def _exchange_quality(entry: ExchangeEntry, perspective: str) -> tuple[float, float]:
    if perspective == "a":
        received = entry.b_sent_filtered
        sent = entry.a_sent_filtered
    else:
        received = entry.a_sent_filtered
        sent = entry.b_sent_filtered

    diff = abs(received - sent)
    similarity = 1.0 - (diff / 10.0)
    magnitude = (received + sent) / 20.0
    return similarity, magnitude


class RecomputedRelationship(TypedDict):
    trust: float
    respect: float
    warmth: float
    cooperation: float
    familiarity: float
    gratitude: float
    closeness: float
    bond_type: str
    conversation_history_score: float


def recompute_from_logs(
    logs: list[ConversationRecord],
    perspective: str = "a",
) -> RecomputedRelationship:
    if not logs:
        return {
            "trust": 0.5, "respect": 0.5, "cooperation": 0.5,
            "familiarity": 0.0, "warmth": 0.5, "gratitude": 0.0,
            "closeness": 0.0, "bond_type": "stranger",
            "conversation_history_score": 0.0,
        }

    dim_totals: dict[str, float] = {
        "trust": 0.0, "respect": 0.0, "cooperation": 0.0,
        "familiarity": 0.0, "warmth": 0.0, "gratitude": 0.0,
    }
    dim_counts: dict[str, float] = {k: 0.0 for k in dim_totals}

    total_history_score = 0.0
    total_similarity = 0.0
    total_magnitude = 0.0
    total_exchanges = 0

    for record in logs:
        weights = _REASON_DIMENSION_WEIGHTS.get(record.reason, _DEFAULT_DIMENSION_WEIGHTS)
        total_history_score += record.total_score

        for entry in record.exchange_log:
            sim, mag = _exchange_quality(entry, perspective)
            total_similarity += sim
            total_magnitude += mag
            total_exchanges += 1

            quality = sim * mag
            for dim, weight in weights.items():
                if weight > 0:
                    dim_totals[dim] += quality * weight
                    dim_counts[dim] += abs(weight)
                elif weight < 0:
                    dim_totals[dim] -= quality * abs(weight) * 0.3
                    dim_counts[dim] += abs(weight)

    result: dict[str, float] = {}
    for dim in dim_totals:
        if dim_counts[dim] > 0:
            raw = dim_totals[dim] / dim_counts[dim]
            scaled = min(0.4, raw * 1.2)
            default = 0.0 if dim in ("familiarity", "gratitude") else 0.5
            n_conversations = len(logs)
            blend_weight = min(1.0, n_conversations / 20.0)
            result[dim] = default * (1 - blend_weight) + (default + scaled) * blend_weight
            result[dim] = max(0.0, min(1.0, result[dim]))
        else:
            result[dim] = 0.5 if dim not in ("familiarity", "gratitude") else 0.0

    interaction_familiarity = min(1.0, len(logs) * 0.05)
    result["familiarity"] = max(result["familiarity"], interaction_familiarity)

    avg_similarity = 0.0
    avg_magnitude = 0.0
    if total_exchanges > 0:
        avg_similarity = total_similarity / total_exchanges
        avg_magnitude = total_magnitude / total_exchanges
    result["closeness"] = avg_similarity * avg_magnitude

    return {
        "trust": result["trust"],
        "respect": result["respect"],
        "warmth": result["warmth"],
        "cooperation": result["cooperation"],
        "familiarity": result["familiarity"],
        "gratitude": result["gratitude"],
        "closeness": result["closeness"],
        "bond_type": _classify_bond(avg_similarity, avg_magnitude, len(logs)),
        "conversation_history_score": total_history_score,
    }


def _classify_bond(avg_similarity: float, avg_magnitude: float, n_conversations: int) -> str:
    if n_conversations < 2:
        return "stranger"
    if avg_similarity < 0.5:
        return "mismatched"
    if avg_magnitude > 0.6:
        return "strong"
    elif avg_magnitude > 0.35:
        return "developing"
    else:
        return "low_energy_stable"


def apply_recomputed_relationship(
    human: DigitalHuman,
    other_id: str,
    recomputed: RecomputedRelationship,
    sim_day: int,
) -> None:
    if other_id not in human.relationships:
        human.relationships[other_id] = Relationship(target_id=other_id)

    rel = human.relationships[other_id]

    # Single application path for behavioral adjustments: The World
    # only ACCUMULATES the (decaying) ledger day-to-day; the ledger
    # is applied exactly once here, on top of the log-derived values.
    # Previously the deltas were applied live daily AND the full
    # ledger re-applied at every recompute, which double-counted and
    # pegged dimensions (e.g. respect) at 1.0 permanently.
    rel.trust = recomputed["trust"]
    rel.respect = recomputed["respect"]
    rel.cooperation = recomputed["cooperation"]
    rel.familiarity = recomputed["familiarity"]
    rel.warmth = recomputed["warmth"]
    rel.gratitude = recomputed["gratitude"]
    rel.conversation_history_score = recomputed["conversation_history_score"]
    rel.last_interaction_day = sim_day

    for dim, adj in rel.behavioral_adjustments.items():
        if hasattr(rel, dim):
            current = getattr(rel, dim)
            setattr(rel, dim, max(0.0, min(1.0, current + adj)))

    rel.opinion = _generate_opinion(rel, recomputed["bond_type"])


def _generate_opinion(rel: Relationship, bond_type: str) -> str:
    parts: list[str] = []

    if bond_type == "stranger":
        return "Don't know them well enough to say."

    if bond_type == "mismatched":
        parts.append("We don't quite click.")
    elif bond_type == "strong":
        parts.append("We work well together.")
    elif bond_type == "developing":
        parts.append("Getting to know them - seems decent.")
    elif bond_type == "low_energy_stable":
        parts.append("Fine. Nothing special, nothing bad.")

    if rel.trust > 0.7:
        parts.append("I trust them.")
    elif rel.trust < 0.3:
        parts.append("Not sure I can rely on them.")

    if rel.respect > 0.7:
        parts.append("Capable person.")
    elif rel.respect < 0.3:
        parts.append("Question their competence sometimes.")

    if rel.warmth > 0.7:
        parts.append("Genuinely like them.")

    if rel.gratitude > 0.5:
        parts.append("They've helped me out.")

    return " ".join(parts) if parts else ""
