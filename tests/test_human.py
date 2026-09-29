"""Tests for the Digital Human class (Step 1).

Key behavioral assertions:
- Human A (high performer) never cuts corners
- Human B (coaster) can cut corners, scores reduce_effort higher
- Human C (social butterfly) socializes more
- Stress rises under overload, recovers with rest
- Memory accumulates
- Trait-gated option availability
- Drive-based scoring differences

Note: act() returns a decision but does NOT execute it. Task dequeuing
is The World's job. These tests verify decision-making, not execution.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import yaml

from digital_human.human import DigitalHuman, Stimulus
from digital_human.types import (
    JJDIDTIEBUCKLE,
    SPECIAL,
    ActionType,
    CopingStyle,
    Drives,
    Episode,
    Memory,
    Option,
    Task,
)


@pytest.fixture
def profiles() -> dict:
    config_path = Path(__file__).parent.parent / "configs" / "test_profiles.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def human_a(profiles) -> DigitalHuman:
    return DigitalHuman.from_config(profiles["humans"][0])


@pytest.fixture
def human_b(profiles) -> DigitalHuman:
    return DigitalHuman.from_config(profiles["humans"][1])


@pytest.fixture
def human_c(profiles) -> DigitalHuman:
    return DigitalHuman.from_config(profiles["humans"][2])


def make_tasks(n: int, day: int) -> list[Task]:
    return [
        Task(
            task_id=f"task_{day}_{i}",
            description=f"Task {i} on day {day}",
            complexity=random.uniform(1.0, 5.0),
            importance=random.uniform(3.0, 8.0),
        )
        for i in range(n)
    ]


def make_stimulus(day: int, n_tasks: int = 2) -> Stimulus:
    return Stimulus(
        sim_day=day,
        new_tasks=make_tasks(n_tasks, day),
        environmental_signals=["normal_day"],
        observable_others=[
            {"id": "coworker_1", "stress": 30.0, "workload": 3},
        ],
    )


class TestInstantiation:
    def test_load_all_three_humans(self, human_a, human_b, human_c):
        assert human_a.human_id == "human_a"
        assert human_b.human_id == "human_b"
        assert human_c.human_id == "human_c"
        assert human_a.name == "Jordan"
        assert human_b.name == "Casey"
        assert human_c.name == "Riley"

    def test_special_scores_valid(self, human_a):
        s = human_a.special
        assert s.strength == 8
        assert s.perception == 9
        assert s.intelligence == 9
        for attr in ("strength", "perception", "endurance", "charisma",
                     "intelligence", "agility", "luck"):
            assert 1 <= getattr(s, attr) <= 10

    def test_jjdidtiebuckle_scores_valid(self, human_a):
        j = human_a.jjdidtiebuckle
        assert j.integrity == 9
        assert j.initiative == 8
        for attr in ("justice", "judgment", "dependability", "initiative",
                     "decisiveness", "tact", "integrity", "enthusiasm",
                     "bearing", "unselfishness", "courage", "knowledge",
                     "loyalty", "endurance"):
            assert 1 <= getattr(j, attr) <= 10

    def test_drives_loaded(self, human_a, human_b):
        assert human_a.drives.mission == 9.0
        assert human_a.drives.comfort == 2.0
        assert human_b.drives.comfort == 9.0
        assert human_b.drives.mission == 2.0

    def test_drive_priority_order(self, human_a, human_b):
        assert human_a.drives.priority_order[0] == "mission"
        assert human_b.drives.priority_order[0] == "comfort"

    def test_inner_world_loaded(self, human_a):
        assert human_a.inner_world.feelings["job_satisfaction"] == 0.4

    def test_work_context_loaded(self, human_a, human_b):
        assert human_a.work_context.role == "Senior Engineer"
        assert human_b.work_context.capacity_threshold == 4

    def test_person_first(self):
        human = DigitalHuman(
            human_id="bare",
            name="Bare Human",
            special=SPECIAL(5, 5, 5, 5, 5, 5, 5),
            jjdidtiebuckle=JJDIDTIEBUCKLE(5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5),
            drives=Drives(),
        )
        assert human.work_context.role == ""

    def test_special_validation_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="SPECIAL.strength"):
            SPECIAL(0, 5, 5, 5, 5, 5, 5)
        with pytest.raises(ValueError, match="SPECIAL.luck"):
            SPECIAL(5, 5, 5, 5, 5, 5, 11)

    def test_special_validates_1_to_10(self):
        SPECIAL(1, 1, 1, 1, 1, 1, 1)
        SPECIAL(10, 10, 10, 10, 10, 10, 10)
        with pytest.raises(ValueError):
            SPECIAL(0, 5, 5, 5, 5, 5, 5)
        with pytest.raises(ValueError):
            SPECIAL(5, 5, 5, 5, 5, 5, 11)

    def test_jjdidtiebuckle_validation_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="JJDIDTIEBUCKLE.justice"):
            JJDIDTIEBUCKLE(0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)
        with pytest.raises(ValueError, match="JJDIDTIEBUCKLE.endurance"):
            JJDIDTIEBUCKLE(5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 11)

    def test_drives_validation_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="Drive compensation"):
            Drives(compensation=-1.0)


class TestDerivedAttributes:
    def test_coping_style_derived(self, human_a, human_b, human_c):
        assert isinstance(human_a.stress.coping_style, CopingStyle)
        assert isinstance(human_b.stress.coping_style, CopingStyle)
        assert isinstance(human_c.stress.coping_style, CopingStyle)

    def test_high_performer_stress_threshold_higher(self, human_a, human_b):
        assert human_a.stress.threshold > human_b.stress.threshold

    def test_recovery_rate_varies(self, human_a, human_b):
        assert human_a.stress.recovery_rate > human_b.stress.recovery_rate

    def test_processing_capacity_varies(self, human_a, human_b):
        assert human_a.processing_capacity > human_b.processing_capacity

    def test_visualization_capacity_varies(self, human_a, human_b):
        assert human_a.visualization_capacity > human_b.visualization_capacity

    def test_conversation_readiness_social_butterfly_highest(self, human_a, human_b, human_c):
        assert human_c.conversation_readiness > human_a.conversation_readiness
        assert human_c.conversation_readiness > human_b.conversation_readiness


class TestDailyCycle:
    def test_single_cycle_produces_action(self, human_a):
        stimulus = make_stimulus(day=1, n_tasks=2)
        action = human_a.daily_cycle(stimulus)
        assert action.action_type is not None
        assert action.sim_day == 1

    def test_perceive_adds_tasks_to_queue(self, human_a):
        initial_count = len(human_a.work_context.task_queue)
        stimulus = make_stimulus(day=1, n_tasks=3)
        human_a.perceive(stimulus)
        assert len(human_a.work_context.task_queue) == initial_count + 3

    def test_understand_generates_options(self, human_a):
        stimulus = make_stimulus(day=1, n_tasks=2)
        human_a.perceive(stimulus)
        options = human_a.understand()
        assert len(options) > 0
        assert all(isinstance(o, Option) for o in options)

    def test_understand_requires_perceive(self, human_a):
        with pytest.raises(RuntimeError, match="perceive"):
            human_a.understand()

    def test_act_requires_understand(self, human_a):
        with pytest.raises(RuntimeError, match="understand"):
            human_a.act()

    def test_act_does_not_dequeue_tasks(self, human_a):
        """act() returns a decision but does NOT execute it.
        Task dequeuing is The World's job."""
        stimulus = make_stimulus(day=1, n_tasks=3)
        queue_before = len(human_a.work_context.task_queue)
        human_a.perceive(stimulus)
        queue_after_perceive = len(human_a.work_context.task_queue)
        human_a.understand()
        human_a.act()
        queue_after_act = len(human_a.work_context.task_queue)
        # perceive adds tasks, but act does NOT remove them
        assert queue_after_perceive == queue_before + 3
        assert queue_after_act == queue_after_perceive

    def test_100_cycles_run_without_error(self, human_a):
        random.seed(42)
        for day in range(1, 101):
            n_tasks = random.randint(1, 4)
            stimulus = make_stimulus(day=day, n_tasks=n_tasks)
            action = human_a.daily_cycle(stimulus)
            assert action.action_type is not None

    def test_100_cycles_all_three_humans(self, human_a, human_b, human_c):
        random.seed(42)
        for human in (human_a, human_b, human_c):
            for day in range(1, 101):
                n_tasks = random.randint(1, 3)
                stimulus = make_stimulus(day=day, n_tasks=n_tasks)
                action = human.daily_cycle(stimulus)
                assert action.action_type is not None


class TestBehavioralDifferences:
    def _run_cycles(
        self, human: DigitalHuman, days: int, tasks_per_day: int, seed: int = 42,
    ) -> list[ActionType]:
        random.seed(seed)
        actions = []
        for day in range(1, days + 1):
            stimulus = make_stimulus(day=day, n_tasks=tasks_per_day)
            action = human.daily_cycle(stimulus)
            actions.append(action.action_type)
        return actions

    def test_high_performer_vs_coaster_different_decisions(self, human_a, human_b):
        actions_a = self._run_cycles(human_a, 50, 3, seed=42)
        actions_b = self._run_cycles(human_b, 50, 3, seed=42)
        assert actions_a != actions_b

    def test_human_a_never_cuts_corners(self, human_a):
        """Human A (integrity=9) should NEVER generate cut_corners."""
        actions = self._run_cycles(human_a, 100, 5, seed=42)
        assert ActionType.CUT_CORNERS not in actions

    def test_human_b_can_cut_corners(self, human_b):
        """Human B (integrity=3) CAN generate cut_corners when overloaded."""
        random.seed(42)
        actions = []
        for day in range(1, 201):
            stimulus = make_stimulus(day=day, n_tasks=6)
            action = human_b.daily_cycle(stimulus)
            actions.append(action.action_type)

        assert ActionType.CUT_CORNERS in actions, (
            "Human B (integrity=3) should generate cut_corners under heavy load"
        )

    def test_human_b_scores_reduce_effort_higher(self, human_b, human_a):
        """Human B (comfort=9, dependability=3) should score reduce_effort
        higher than Human A (comfort=2, dependability=9)."""
        stimulus = make_stimulus(day=1, n_tasks=3)

        human_a.perceive(stimulus)
        options_a = human_a.understand()
        human_a.act()

        stimulus_b = make_stimulus(day=1, n_tasks=3)
        human_b.perceive(stimulus_b)
        options_b = human_b.understand()
        human_b.act()

        reduce_a = [o for o in options_a if o.action_type == ActionType.REDUCE_EFFORT]
        reduce_b = [o for o in options_b if o.action_type == ActionType.REDUCE_EFFORT]

        assert len(reduce_b) > 0, "Human B should generate reduce_effort option"

        if reduce_a:
            assert reduce_b[0].score > reduce_a[0].score

    def test_social_butterfly_socializes_more(self, human_a, human_c):
        """Human C (charisma=5, enthusiasm=9) should socialize more than Human A."""
        actions_a = self._run_cycles(human_a, 100, 2, seed=42)
        actions_c = self._run_cycles(human_c, 100, 2, seed=42)

        socialize_a = actions_a.count(ActionType.SOCIALIZE)
        socialize_c = actions_c.count(ActionType.SOCIALIZE)
        assert socialize_c > socialize_a, (
            f"Social butterfly should socialize more: C={socialize_c}, A={socialize_a}"
        )

    def test_high_initiative_takes_initiative(self, human_a, human_b):
        """Human A (initiative=8) should take initiative. Human B (initiative=2) should not."""
        actions_a = self._run_cycles(human_a, 100, 2, seed=42)
        actions_b = self._run_cycles(human_b, 100, 2, seed=42)

        init_a = actions_a.count(ActionType.TAKE_INITIATIVE)
        init_b = actions_b.count(ActionType.TAKE_INITIATIVE)
        assert init_a > init_b, (
            f"High-initiative human should take more initiative: A={init_a}, B={init_b}"
        )


class TestStress:
    def test_stress_rises_under_sustained_overload(self, human_a):
        random.seed(42)
        initial_stress = human_a.stress.score

        for day in range(1, 31):
            stimulus = make_stimulus(day=day, n_tasks=8)
            human_a.daily_cycle(stimulus)

        assert human_a.stress.score > initial_stress, (
            f"Stress should rise under overload: initial={initial_stress}, "
            f"final={human_a.stress.score}"
        )

    def test_stress_recovers_with_rest(self, human_a):
        random.seed(42)

        for day in range(1, 21):
            stimulus = make_stimulus(day=day, n_tasks=8)
            human_a.daily_cycle(stimulus)

        stressed_score = human_a.stress.score

        # Simulate The World having processed accumulated tasks
        human_a.work_context.task_queue.clear()

        for day in range(21, 41):
            stimulus = make_stimulus(day=day, n_tasks=0)
            human_a.daily_cycle(stimulus)

        assert human_a.stress.score < stressed_score, (
            f"Stress should recover: stressed={stressed_score}, "
            f"recovered={human_a.stress.score}"
        )

    def test_coaster_breaks_sooner(self, human_a, human_b):
        random.seed(42)

        for day in range(1, 51):
            stim_a = make_stimulus(day=day, n_tasks=8)
            stim_b = make_stimulus(day=day, n_tasks=8)
            human_a.daily_cycle(stim_a)
            human_b.daily_cycle(stim_b)

        ratio_a = human_a.stress.score / human_a.stress.threshold
        ratio_b = human_b.stress.score / human_b.stress.threshold
        assert ratio_b > ratio_a, (
            f"Coaster should be proportionally closer to breaking: "
            f"B={human_b.stress.score:.1f}/{human_b.stress.threshold:.1f} ({ratio_b:.2%}), "
            f"A={human_a.stress.score:.1f}/{human_a.stress.threshold:.1f} ({ratio_a:.2%})"
        )


class TestMemory:
    def test_memory_accumulates(self, human_a):
        random.seed(42)
        for day in range(1, 51):
            stimulus = make_stimulus(day=day, n_tasks=2)
            human_a.daily_cycle(stimulus)

        assert len(human_a.memory.episodes) >= 50, (
            f"Should have at least 50 episodes after 50 cycles, "
            f"got {len(human_a.memory.episodes)}"
        )

    def test_memory_episodes_have_correct_structure(self, human_a):
        stimulus = make_stimulus(day=1, n_tasks=2)
        human_a.daily_cycle(stimulus)

        assert len(human_a.memory.episodes) > 0
        ep = human_a.memory.episodes[0]
        assert ep.sim_day == 1
        assert ep.event_type == "decision"
        assert isinstance(ep.emotional_valence, float)
        assert -1.0 <= ep.emotional_valence <= 1.0
        assert len(ep.tags) > 0

    def test_high_impact_memories_promoted_to_long_term(self, human_a):
        random.seed(42)
        for day in range(1, 101):
            stimulus = make_stimulus(day=day, n_tasks=random.randint(1, 8))
            human_a.daily_cycle(stimulus)

        high_impact = Episode(
            sim_day=200,
            event_type="observation",
            content="Something significant happened",
            emotional_valence=0.9,
            tags=["significant"],
        )
        human_a.memory.record(high_impact)
        assert high_impact in human_a.memory.long_term

    def test_memory_bounded_at_500(self):
        mem = Memory()
        for i in range(600):
            mem.record(Episode(
                sim_day=i, event_type="test", content=f"ep {i}",
                emotional_valence=0.0,
            ))
        assert len(mem.episodes) == 500

    def test_memory_search_by_tag(self, human_a):
        random.seed(42)
        for day in range(1, 11):
            stimulus = make_stimulus(day=day, n_tasks=2)
            human_a.daily_cycle(stimulus)

        results = human_a.memory.search("process_task")
        assert isinstance(results, list)


class TestInnerWorld:
    def test_feelings_shift_over_cycles(self, human_a):
        initial_satisfaction = human_a.inner_world.feelings.get("job_satisfaction", 0.0)
        random.seed(42)

        for day in range(1, 51):
            stimulus = make_stimulus(day=day, n_tasks=2)
            human_a.daily_cycle(stimulus)

        final_satisfaction = human_a.inner_world.feelings.get("job_satisfaction", 0.0)
        assert initial_satisfaction != final_satisfaction, (
            "Job satisfaction should shift after 50 cycles"
        )

    def test_feelings_bounded(self, human_a):
        random.seed(42)
        for day in range(1, 201):
            stimulus = make_stimulus(day=day, n_tasks=random.randint(0, 10))
            human_a.daily_cycle(stimulus)

        for key, val in human_a.inner_world.feelings.items():
            assert -1.0 <= val <= 1.0, f"Feeling {key}={val} out of bounds"


class TestPerceptionSampling:
    def test_perception4_notices_quality_review_within_50_days(self, human_b):
        """Low perception is noisy sampling, not prefix truncation:
        a perception-4 human keeps 1 of 2 signals at random each day,
        so over 50 days a trailing 'quality_review' signal must be
        noticed at least once — and missed at least once."""
        random.seed(42)
        seen = 0
        for day in range(1, 51):
            stim = Stimulus(
                sim_day=day,
                new_tasks=[],
                environmental_signals=["normal_day", "quality_review"],
                observable_others=[],
            )
            perception = human_b.perceive(stim)
            if "quality_review" in perception.environmental_signals:
                seen += 1
                assert human_b.stress.sources.get("social_pressure") == 15.0
        assert seen >= 1, (
            "perception-4 human never noticed quality_review in 50 days "
            "— signals are still being prefix-truncated"
        )
        assert seen < 50, "perception-4 human should also miss it sometimes"


class TestStressThreshold:
    def test_stress_clamped_to_range(self, human_b):
        random.seed(42)
        for day in range(1, 201):
            n_tasks = random.choice([0, 0, 0, 10, 10, 10])
            stimulus = make_stimulus(day=day, n_tasks=n_tasks)
            human_b.daily_cycle(stimulus)
            assert 0.0 <= human_b.stress.score <= 100.0, (
                f"Stress out of bounds on day {day}: {human_b.stress.score}"
            )
