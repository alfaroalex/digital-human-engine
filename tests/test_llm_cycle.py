"""Tests for LLM-driven Perceive + Understand and Act tagging layer.

All tests use mocked LLM responses — no live ollama required.
"""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from digital_human.human import DigitalHuman, Stimulus
from digital_human.llm import parse_decision
from digital_human.types import (
    JJDIDTIEBUCKLE,
    SPECIAL,
    ActionType,
    Drives,
    LLMConfig,
    Task,
    WorkContext,
)
from digital_human.world import World


def _make_human(llm_enabled: bool = False) -> DigitalHuman:
    return DigitalHuman(
        human_id="test_h",
        name="Tester",
        special=SPECIAL(5, 7, 5, 8, 6, 5, 5),
        jjdidtiebuckle=JJDIDTIEBUCKLE(
            justice=5, judgment=6, dependability=7, initiative=5,
            decisiveness=5, tact=3, integrity=5, enthusiasm=5,
            bearing=6, unselfishness=6, courage=7, knowledge=6,
            loyalty=5, endurance=5,
        ),
        drives=Drives(mission=7.0, comfort=4.0),
        work_context=WorkContext(
            role="Engineer",
            responsibilities=["code_review", "design"],
            capacity_threshold=5,
            authority_level=2,
            reports_to="boss",
        ),
        llm_config=LLMConfig(enabled=llm_enabled),
    )


def _make_stimulus(day: int = 1, n_tasks: int = 2) -> Stimulus:
    return Stimulus(
        sim_day=day,
        new_tasks=[
            Task(
                task_id=f"t_{day}_{i}",
                description="code review",
                requirements={"intelligence": 6, "knowledge": 5},
            )
            for i in range(n_tasks)
        ],
        environmental_signals=["normal_day"],
        observable_others=[
            {"id": "coworker_1", "stress": 20.0, "workload": 3,
             "apparent_regard": 0.6, "authority_level": 1},
        ],
    )


MOCK_PERCEPTION = "Another pile of code reviews landed. Coworker_1 seems calm enough."

MOCK_DECISION_TEXT = (
    "WANT: Take a break and scroll my phone\n"
    "REASON: I'm tired of reviewing code all day\n"
    "DECIDE: Process the code reviews since they're due\n"
    "ACTION: process_task\n"
    "TARGET: none\n"
    "CONFLICT: Giving up rest even though I need it"
)

MOCK_DECISION_CONFRONT = (
    "WANT: Tell coworker_1 off for sloppy code\n"
    "REASON: Fed up with reviewing garbage\n"
    "DECIDE: Confront them professionally\n"
    "ACTION: confront\n"
    "TARGET: coworker_1\n"
    "CONFLICT: Risking the relationship"
)


class TestLLMPerceive:
    @patch("digital_human.llm.generate_perception", return_value=MOCK_PERCEPTION)
    def test_called_when_enabled(self, mock_gen):
        h = _make_human(llm_enabled=True)
        stim = _make_stimulus()
        h._llm_perceive(stim)
        assert h._llm_perception == MOCK_PERCEPTION
        mock_gen.assert_called_once()

    def test_skipped_when_disabled(self):
        h = _make_human(llm_enabled=False)
        stim = _make_stimulus()
        h._llm_perceive(stim)
        assert h._llm_perception is None


class TestLLMUnderstand:
    @patch("digital_human.llm.generate_decision")
    def test_produces_valid_action(self, mock_gen):
        mock_gen.return_value = parse_decision(MOCK_DECISION_TEXT)
        h = _make_human(llm_enabled=True)
        h._llm_perception = MOCK_PERCEPTION
        stim = _make_stimulus()
        h.perceive(stim)

        decision = h._llm_understand()
        assert decision is not None
        assert decision["action"] == "process_task"
        assert decision["target"] is None
        assert decision["conflict"] is not None

    def test_parses_structured_response(self):
        result = parse_decision(MOCK_DECISION_TEXT)
        assert result is not None
        assert result["action"] == "process_task"
        assert result["want"] == "Take a break and scroll my phone"
        assert result["reason"] == "I'm tired of reviewing code all day"
        assert result["decide"] == "Process the code reviews since they're due"
        assert result["target"] is None
        assert result["conflict"] == "Giving up rest even though I need it"

    def test_parse_returns_none_on_garbage(self):
        assert parse_decision("hello world") is None
        assert parse_decision("ACTION: not_a_valid_action") is None
        assert parse_decision("") is None

    @patch("digital_human.llm.generate_decision", return_value=None)
    def test_falls_back_on_parse_failure(self, mock_gen):
        h = _make_human(llm_enabled=True)
        h._llm_perception = MOCK_PERCEPTION
        stim = _make_stimulus()
        h.perceive(stim)
        decision = h._llm_understand()
        assert decision is None


class TestActTags:
    def _run_and_get_action(self, n_tasks=2):
        h = _make_human()
        stim = _make_stimulus(n_tasks=n_tasks)
        return h.daily_cycle(stim), h

    def test_execution_quality_tags(self):
        action, h = self._run_and_get_action()
        assert "tact" in action.metadata
        assert "composure" in action.metadata
        assert "competence" in action.metadata
        assert "reliability" in action.metadata
        assert 0.0 <= action.metadata["tact"] <= 1.0
        assert 0.0 <= action.metadata["competence"] <= 1.0

    def test_reception_prediction_on_confront(self):
        h = _make_human()
        h.jjdidtiebuckle = JJDIDTIEBUCKLE(
            justice=5, judgment=5, dependability=5, initiative=5,
            decisiveness=5, tact=2, integrity=5, enthusiasm=5,
            bearing=5, unselfishness=5, courage=8, knowledge=5,
            loyalty=5, endurance=5,
        )
        decision = {
            "action": "confront",
            "want": "tell them off",
            "reason": "fed up",
            "decide": "confront them",
            "target": "coworker_1",
            "conflict": None,
        }
        stim = _make_stimulus()
        h.perceive(stim)
        action = h.act(llm_decision=decision)
        assert action.metadata.get("reception_prediction") == "likely_hostile_reception"

    def test_relationship_impact_tags(self):
        action, _ = self._run_and_get_action()
        at = action.action_type
        if at in (ActionType.PROCESS_TASK, ActionType.PUSH_THROUGH,
                  ActionType.CONFRONT, ActionType.OFFER_HELP,
                  ActionType.CUT_CORNERS, ActionType.DELEGATE,
                  ActionType.TAKE_INITIATIVE):
            assert "relationship_impacts" in action.metadata

    def test_stress_impact_tags(self):
        action, _ = self._run_and_get_action()
        assert "suppression_cost" in action.metadata
        assert "effort_cost" in action.metadata
        assert "relief_potential" in action.metadata
        assert isinstance(action.metadata["relief_potential"], list)

    def test_performance_metrics_on_task_action(self):
        random.seed(42)
        h = _make_human()
        stim = _make_stimulus(n_tasks=3)
        action = h.daily_cycle(stim)
        if action.action_type in (ActionType.PROCESS_TASK,
                                   ActionType.CUT_CORNERS,
                                   ActionType.PUSH_THROUGH):
            assert "trait_match" in action.metadata
            assert "predicted_quality" in action.metadata
            assert "error_probability" in action.metadata


class TestLLMTaskGeneration:
    @patch("digital_human.llm.generate_tasks", return_value=[
        {"description": "design API", "requirements": {"intelligence": 7},
         "complexity": 0.8, "importance": 7.0},
    ])
    def test_produces_valid_tasks(self, mock_gen):
        w = World.from_config(
            "configs/test_profiles.yaml", seed=42,
            llm_config=LLMConfig(enabled=True),
        )
        role = w.human_list[0].work_context.role
        templates = w._get_templates_for_role(role)
        assert len(templates) > 0
        assert all("description" in t and "requirements" in t for t in templates)

    def test_falls_back_to_hardcoded_templates(self):
        w = World.from_config("configs/test_profiles.yaml", seed=42)
        templates = w._get_templates_for_role("nonexistent_role")
        assert len(templates) == 10


class TestFullCycleWithLLM:
    @patch("digital_human.llm.generate_decision")
    @patch("digital_human.llm.generate_perception", return_value=MOCK_PERCEPTION)
    def test_full_cycle_with_llm_mock(self, mock_perceive, mock_decide):
        mock_decide.return_value = parse_decision(MOCK_DECISION_TEXT)
        h = _make_human(llm_enabled=True)
        stim = _make_stimulus()
        action = h.daily_cycle(stim)

        assert action.action_type == ActionType.PROCESS_TASK
        assert action.metadata.get("llm_driven") is True
        assert action.metadata.get("conflict") is not None
        assert "tact" in action.metadata

    def test_full_cycle_without_llm_matches_existing(self):
        random.seed(42)
        h = _make_human(llm_enabled=False)
        stim = _make_stimulus()
        action = h.daily_cycle(stim)

        assert action.action_type is not None
        assert action.metadata.get("llm_driven") is None
        assert "tact" in action.metadata
        assert "composure" in action.metadata
