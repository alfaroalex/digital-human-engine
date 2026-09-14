"""Direct tests for the conversation protocol (conversation.py).

Covers the +3 JJDIDTIEBUCKLE filter with overflow, relationship delta
math, recompute_from_logs, bond classification, and the
apply_conversation_result mutation boundary.
"""

from __future__ import annotations

import random

import pytest

from digital_human.conversation import (
    Conversation,
    _classify_bond,
    apply_conversation_result,
    recompute_from_logs,
)
from digital_human.human import DigitalHuman
from digital_human.types import (
    JJDIDTIEBUCKLE,
    SPECIAL,
    ConversationRecord,
    Drives,
    ExchangeEntry,
    Task,
    WorkContext,
)


def _make_human(human_id: str) -> DigitalHuman:
    return DigitalHuman(
        human_id=human_id,
        name=human_id,
        special=SPECIAL(5, 5, 5, 5, 5, 5, 5),
        jjdidtiebuckle=JJDIDTIEBUCKLE(
            justice=5, judgment=5, dependability=5, initiative=5,
            decisiveness=5, tact=5, integrity=5, enthusiasm=5,
            bearing=5, unselfishness=5, courage=5, knowledge=5,
            loyalty=5, endurance=5,
        ),
        drives=Drives(),
        work_context=WorkContext(role="Tester", capacity_threshold=5),
    )


def _entry(a: int = 8, b: int = 8, cycle: int = 1) -> ExchangeEntry:
    return ExchangeEntry(
        cycle=cycle,
        a_sent_system="special", a_sent_trait="charisma",
        a_sent_raw=a, a_sent_filtered=a,
        b_sent_system="special", b_sent_trait="charisma",
        b_sent_raw=b, b_sent_filtered=b,
    )


def _record(
    reason: str = "social",
    entries: list[ExchangeEntry] | None = None,
    day: int = 1,
) -> ConversationRecord:
    entries = entries if entries is not None else [_entry()]
    return ConversationRecord(
        sim_day=day,
        reason=reason,
        cycles=len(entries),
        exchange_log=entries,
        total_score=sum(e.combined_score for e in entries),
    )


class TestApplyFilter:
    """Spec §4: SPECIAL goes through direct; JJDIDTIEBUCKLE gets +3,
    capped at 10 with the excess returned as overflow."""

    def test_jjd_boost_no_overflow(self):
        assert Conversation._apply_filter("jjdidtiebuckle", 5) == (8, 0)
        assert Conversation._apply_filter("jjdidtiebuckle", 7) == (10, 0)

    def test_jjd_boost_with_overflow(self):
        assert Conversation._apply_filter("jjdidtiebuckle", 8) == (10, 1)
        assert Conversation._apply_filter("jjdidtiebuckle", 10) == (10, 3)

    def test_special_passes_through_unfiltered(self):
        assert Conversation._apply_filter("special", 8) == (8, 0)
        assert Conversation._apply_filter("special", 10) == (10, 0)
        assert Conversation._apply_filter("special", 1) == (1, 0)


class TestRelationshipDelta:
    def _conv(self) -> Conversation:
        return Conversation(
            _make_human("a"), _make_human("b"), "social",
            sim_day=1, rng=random.Random(0),
        )

    def test_matched_high_scores(self):
        """Both send 8: similarity = 1 - 0/10 = 1.0, combined =
        16/20 = 0.8 -> contribution 0.8 -> delta = 0.8 * 0.15 = 0.12."""
        conv = self._conv()
        conv.exchange_log = [_entry(a=8, b=8)]
        assert conv._compute_relationship_delta("a") == pytest.approx(0.12)
        assert conv._compute_relationship_delta("b") == pytest.approx(0.12)

    def test_mismatched_scores_contribute_less(self):
        """a sends 10, b sends 2: similarity = 1 - 8/10 = 0.2,
        combined = 12/20 = 0.6 -> contribution 0.12 -> delta 0.018."""
        conv = self._conv()
        conv.exchange_log = [_entry(a=10, b=2)]
        assert conv._compute_relationship_delta("a") == pytest.approx(0.018)

    def test_overflow_adds_per_point(self):
        conv = self._conv()
        conv.exchange_log = [_entry(a=8, b=8)]
        conv.overflow_a = 2
        assert conv._compute_relationship_delta("a") == pytest.approx(0.14)
        assert conv._compute_relationship_delta("b") == pytest.approx(0.12)

    def test_empty_log_is_zero(self):
        conv = self._conv()
        conv.exchange_log = []
        assert conv._compute_relationship_delta("a") == 0.0

    def test_run_produces_expected_cycle_count(self):
        conv = self._conv()
        result = conv.run()
        assert len(result.record_a.exchange_log) == conv.expected_cycles
        for e in result.record_a.exchange_log:
            assert e.combined_score == e.a_sent_filtered + e.b_sent_filtered
            assert 1 <= e.a_sent_filtered <= 10
            assert 1 <= e.b_sent_filtered <= 10


class TestRecomputeFromLogs:
    def test_empty_logs_return_defaults(self):
        result = recompute_from_logs([])
        assert result["trust"] == 0.5
        assert result["respect"] == 0.5
        assert result["cooperation"] == 0.5
        assert result["warmth"] == 0.5
        assert result["familiarity"] == 0.0
        assert result["gratitude"] == 0.0
        assert result["closeness"] == 0.0
        assert result["bond_type"] == "stranger"

    def test_single_log_blends_toward_default(self):
        """blend_weight = 1/20 — one conversation barely moves the
        dimensions off their defaults."""
        result = recompute_from_logs([_record("social", [_entry(8, 8)])])
        # quality = 1.0 * 0.8 = 0.8; scaled = min(0.4, 0.96) = 0.4
        # trust = 0.5*(1 - 1/20) + (0.5 + 0.4)*(1/20) = 0.52
        assert result["trust"] == pytest.approx(0.52)
        assert result["warmth"] == pytest.approx(0.52)

    def test_many_logs_full_blend(self):
        """20+ logs -> blend weight 1.0 -> default + scaled cap."""
        logs = [_record("social", [_entry(8, 8)], day=d) for d in range(20)]
        result = recompute_from_logs(logs)
        assert result["trust"] == pytest.approx(0.9)
        assert result["warmth"] == pytest.approx(0.9)
        # familiarity floor from interaction count: 20 * 0.05 = 1.0
        assert result["familiarity"] == pytest.approx(1.0)
        # closeness = avg_similarity * avg_magnitude = 1.0 * 0.8
        assert result["closeness"] == pytest.approx(0.8)
        assert result["bond_type"] == "strong"
        assert result["conversation_history_score"] == pytest.approx(20 * 16)

    def test_confrontation_negative_warmth_weight(self):
        """Confrontation logs raise respect but pull warmth below
        its 0.5 default."""
        logs = [
            _record("confrontation", [_entry(8, 8)], day=d) for d in range(20)
        ]
        result = recompute_from_logs(logs)
        assert result["respect"] > 0.5
        assert result["warmth"] < 0.5

    def test_untouched_dimension_keeps_default(self):
        """Social logs never touch gratitude (no weight) — it stays
        at its 0.0 default."""
        logs = [_record("social", [_entry(8, 8)], day=d) for d in range(20)]
        result = recompute_from_logs(logs)
        assert result["gratitude"] == 0.0


class TestClassifyBond:
    def test_stranger_below_two_conversations(self):
        assert _classify_bond(1.0, 0.9, 0) == "stranger"
        assert _classify_bond(1.0, 0.9, 1) == "stranger"

    def test_mismatched_low_similarity(self):
        assert _classify_bond(0.3, 0.9, 5) == "mismatched"

    def test_strong_high_magnitude(self):
        assert _classify_bond(0.8, 0.7, 5) == "strong"

    def test_developing_mid_magnitude(self):
        assert _classify_bond(0.8, 0.5, 5) == "developing"

    def test_low_energy_stable_low_magnitude(self):
        assert _classify_bond(0.8, 0.2, 5) == "low_energy_stable"


class TestApplyConversationResult:
    def test_both_humans_get_logs_and_relationships(self):
        a, b = _make_human("a"), _make_human("b")
        conv = Conversation(a, b, "social", sim_day=3, rng=random.Random(1))
        result = conv.run()
        apply_conversation_result(a, b, result, sim_day=3)

        assert len(a.memory.conversation_logs["b"]) == 1
        assert len(b.memory.conversation_logs["a"]) == 1

        rel_ab = a.relationships["b"]
        rel_ba = b.relationships["a"]
        assert rel_ab.last_interaction_day == 3
        assert rel_ba.last_interaction_day == 3
        assert rel_ab.familiarity > 0.0
        # social conversations move warmth and trust upward
        assert rel_ab.warmth >= 0.5
        assert rel_ab.trust >= 0.5

        conv_episodes_a = [
            e for e in a.memory.episodes if e.event_type == "conversation"
        ]
        conv_episodes_b = [
            e for e in b.memory.episodes if e.event_type == "conversation"
        ]
        assert len(conv_episodes_a) == 1
        assert len(conv_episodes_b) == 1
        assert conv_episodes_a[0].people_involved == ["b"]

        # opinion belief formed about the other person
        assert "opinion_of_b" in a.inner_world.beliefs

    def test_delegation_transfers_tasks(self):
        a, b = _make_human("a"), _make_human("b")
        for i in range(5):
            a.work_context.task_queue.append(
                Task(task_id=f"t{i}", description=f"task {i}")
            )

        conv = Conversation(a, b, "delegation", sim_day=1, rng=random.Random(2))
        result = conv.run()
        apply_conversation_result(a, b, result, sim_day=1)

        assert result.tasks_transferred >= 1
        assert len(a.work_context.task_queue) == 5 - result.tasks_transferred
        assert len(b.work_context.task_queue) == result.tasks_transferred
        # transferred tasks are the actual Task objects from a's queue
        assert b.work_context.task_queue[0].task_id == "t0"

    def test_social_reason_does_not_transfer_tasks(self):
        a, b = _make_human("a"), _make_human("b")
        a.work_context.task_queue.append(Task(task_id="t0", description="x"))

        conv = Conversation(a, b, "social", sim_day=1, rng=random.Random(3))
        result = conv.run()
        apply_conversation_result(a, b, result, sim_day=1)

        assert result.tasks_transferred == 0
        assert len(a.work_context.task_queue) == 1
        assert len(b.work_context.task_queue) == 0
