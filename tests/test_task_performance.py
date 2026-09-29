"""Tests for trait-gated tasks with performance measurement."""


import pytest

from digital_human.human import DigitalHuman
from digital_human.types import (
    JJDIDTIEBUCKLE,
    SPECIAL,
    ActionType,
    Drives,
    Task,
    WorkContext,
)
from digital_human.world import World


def _make_human(
    human_id: str,
    special: SPECIAL,
    jjdidtiebuckle: JJDIDTIEBUCKLE | None = None,
    drives: Drives | None = None,
) -> DigitalHuman:
    return DigitalHuman(
        human_id=human_id,
        name=human_id,
        special=special,
        jjdidtiebuckle=jjdidtiebuckle or JJDIDTIEBUCKLE(
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        ),
        drives=drives or Drives(),
        work_context=WorkContext(capacity_threshold=5),
    )


@pytest.fixture
def world_100():
    world = World.from_config("configs/test_profiles.yaml", seed=42)
    world.run(days=100)
    return world


class TestTaskRequirementsHighMatch:
    def test_agent_exceeds_all_requirements(self):
        human = _make_human(
            "smart",
            SPECIAL(9, 9, 9, 9, 9, 9, 9),
            JJDIDTIEBUCKLE(8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8),
        )
        task = Task(
            task_id="easy_1",
            description="routine filing",
            requirements={"intelligence": 4, "knowledge": 3},
        )
        world = World(humans=[human], seed=42)
        world.sim_day = 1
        result = world._compute_task_result(human, task, ActionType.PROCESS_TASK)

        assert result.completed is True
        assert result.quality > 0.8
        assert result.speed >= 1.0
        assert result.trait_match == 1.0
        assert result.errors == 0


class TestTaskRequirementsSlightGap:
    def test_agent_slightly_below_requirements(self):
        human = _make_human(
            "average",
            SPECIAL(5, 5, 5, 5, 5, 5, 5),
            JJDIDTIEBUCKLE(5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5),
        )
        task = Task(
            task_id="hard_1",
            description="code review",
            requirements={"intelligence": 7, "knowledge": 7, "perception": 6},
        )
        world = World(humans=[human], seed=42)
        world.sim_day = 1
        result = world._compute_task_result(human, task, ActionType.PROCESS_TASK)

        assert result.completed is True
        assert result.quality < 0.7
        assert result.speed < 1.0
        assert result.trait_match < 1.0


class TestTaskRequirementsLargeGap:
    def test_agent_far_below_fails_task(self):
        human = _make_human(
            "weak",
            SPECIAL(1, 1, 1, 1, 1, 1, 1),
            JJDIDTIEBUCKLE(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        )
        task = Task(
            task_id="impossible_1",
            description="strategic planning",
            requirements={"intelligence": 7, "judgment": 7, "initiative": 6},
        )
        world = World(humans=[human], seed=42)
        world.sim_day = 1
        result = world._compute_task_result(human, task, ActionType.PROCESS_TASK)

        assert result.completed is False
        assert "could not handle" in result.details


class TestTaskNoRequirementsFallback:
    def test_empty_requirements_uses_basic_calc(self):
        human = _make_human("basic", SPECIAL(5, 5, 5, 5, 5, 5, 5))
        task = Task(
            task_id="basic_1",
            description="generic task",
        )
        world = World(humans=[human], seed=42)
        world.sim_day = 1
        result = world._compute_task_result(human, task, ActionType.PROCESS_TASK)

        assert result.completed is True
        assert result.speed == 1.0
        assert result.trait_match == 1.0
        assert result.details == "standard task"
        assert 0.1 <= result.quality <= 1.0


class TestTaskResultRecordedInDataset:
    def test_results_stored(self, world_100):
        ds = world_100.dataset
        assert len(ds.task_results) > 0
        for hid in ("human_a", "human_b", "human_c"):
            perf = ds.get_human_task_performance(hid)
            assert len(perf) > 0


class TestTaskFailureCreatesStress:
    def test_failure_adds_stress_source(self):
        human = _make_human(
            "weak",
            SPECIAL(1, 1, 1, 1, 1, 1, 1),
            JJDIDTIEBUCKLE(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        )
        task = Task(
            task_id="fail_1",
            description="strategic planning",
            requirements={"intelligence": 7, "judgment": 7, "initiative": 6},
        )
        world = World(humans=[human], seed=42)
        world.sim_day = 1
        result = world._compute_task_result(human, task, ActionType.PROCESS_TASK)
        world._apply_task_feedback(human, task, result)

        assert human.stress.sources.get("task_failure", 0) > 0
        assert human.inner_world.feelings.get("motivation", 0) < 0
        failure_episodes = [
            e for e in human.memory.episodes
            if e.event_type == "task_failure"
        ]
        assert len(failure_episodes) > 0


class TestTaskSuccessBoostsSatisfaction:
    def test_high_quality_important_task_boosts(self):
        human = _make_human(
            "star",
            SPECIAL(9, 9, 9, 9, 9, 9, 9),
            JJDIDTIEBUCKLE(9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9),
        )
        initial_satisfaction = human.inner_world.feelings.get("job_satisfaction", 0.0)
        initial_motivation = human.inner_world.feelings.get("motivation", 0.0)

        task = Task(
            task_id="big_1",
            description="strategic planning",
            importance=8.5,
            requirements={"intelligence": 5, "judgment": 5},
        )
        world = World(humans=[human], seed=42)
        world.sim_day = 1
        result = world._compute_task_result(human, task, ActionType.PROCESS_TASK)
        assert result.quality > 0.8
        world._apply_task_feedback(human, task, result)

        assert human.inner_world.feelings.get("job_satisfaction", 0) > initial_satisfaction
        assert human.inner_world.feelings.get("motivation", 0) > initial_motivation


class TestJordanExcelsAtStrategicTasks:
    def test_jordan_strategic_high_quality(self):
        world = World.from_config("configs/test_profiles.yaml", seed=42)
        jordan = world.humans["human_a"]
        task = Task(
            task_id="strat_1",
            description="strategic planning",
            requirements={"intelligence": 6, "judgment": 6, "initiative": 5},
        )
        world.sim_day = 1
        result = world._compute_task_result(jordan, task, ActionType.PROCESS_TASK)

        assert result.completed is True
        assert result.quality > 0.7
        assert result.speed >= 1.0
        assert "exceeded" in result.details


class TestCaseyFailsAtComplexTasks:
    def test_casey_complex_low_quality(self):
        world = World.from_config("configs/test_profiles.yaml", seed=42)
        casey = world.humans["human_b"]
        task = Task(
            task_id="complex_1",
            description="code review",
            requirements={"intelligence": 8, "knowledge": 8, "perception": 7},
        )
        world.sim_day = 1
        result = world._compute_task_result(casey, task, ActionType.PROCESS_TASK)

        assert result.quality < 0.5
        assert result.speed < 1.0
        assert result.trait_match < 1.0
        assert "struggled" in result.details
