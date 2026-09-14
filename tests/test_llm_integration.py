"""Tests for LLM integration — all mock-based, no ollama required."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from digital_human.human import DigitalHuman, Stimulus
from digital_human.types import (
    JJDIDTIEBUCKLE,
    SPECIAL,
    ActionType,
    Drives,
    Episode,
    LLMConfig,
    WorkContext,
)
from digital_human.world import World


def _make_human(llm_config: LLMConfig | None = None) -> DigitalHuman:
    return DigitalHuman(
        human_id="test_h",
        name="Tester",
        special=SPECIAL(5, 5, 5, 5, 5, 5, 5),
        jjdidtiebuckle=JJDIDTIEBUCKLE(
            justice=5, judgment=5, dependability=5, initiative=5,
            decisiveness=5, tact=5, integrity=5, enthusiasm=5,
            bearing=5, unselfishness=5, courage=5, knowledge=5,
            loyalty=5, endurance=5,
        ),
        drives=Drives(),
        work_context=WorkContext(role="Analyst"),
        llm_config=llm_config,
    )


def test_llm_disabled_no_calls():
    h = _make_human(LLMConfig(enabled=False))
    for day in range(1, 11):
        stim = Stimulus(sim_day=day)
        h.daily_cycle(stim)
        h.generate_internal_dialogue(day)
    assert len(h.memory_nodes) == 0


@patch("digital_human.llm.generate_memory_node", return_value="Today felt routine.")
def test_memory_node_stored_correctly(mock_gen):
    h = _make_human(LLMConfig(enabled=True))
    stim = Stimulus(sim_day=1)
    h.daily_cycle(stim)
    h.generate_internal_dialogue(1)

    assert len(h.memory_nodes) == 1
    assert h.memory_nodes[0] == (1, "Today felt routine.")

    dialogue_episodes = [
        e for e in h.memory.episodes if e.event_type == "internal_dialogue"
    ]
    assert len(dialogue_episodes) == 1
    assert dialogue_episodes[0].content == "Today felt routine."
    assert dialogue_episodes[0].sim_day == 1
    assert "internal_dialogue" in dialogue_episodes[0].tags


def test_serialize_for_llm_produces_valid_dict():
    h = _make_human()
    stim = Stimulus(sim_day=1)
    h.daily_cycle(stim)

    state = h._serialize_for_llm()
    expected_keys = {
        "name", "role", "traits_summary", "top_drives",
        "stress_score", "stress_trajectory", "top_stress_source",
        "feelings", "raw_impulse", "filtered_action",
        "task_queue_size", "task_descriptions", "relationships",
        "recent_nodes",
    }
    assert expected_keys == set(state.keys())
    assert state["name"] == "Tester"
    assert state["role"] == "Analyst"
    assert isinstance(state["stress_score"], float)
    assert isinstance(state["task_queue_size"], int)


@patch("digital_human.llm.generate_memory_node", side_effect=Exception("connection refused"))
def test_graceful_degradation(mock_gen):
    h = _make_human(LLMConfig(enabled=True))
    for day in range(1, 6):
        stim = Stimulus(sim_day=day)
        h.daily_cycle(stim)
        h.generate_internal_dialogue(day)

    assert len(h.memory_nodes) == 0
    assert len(h.memory.episodes) > 0


def test_llm_config_loads_from_yaml():
    config = {
        "humans": [{
            "id": "h1",
            "name": "YamlPerson",
            "special": dict(strength=3, perception=3, endurance=3,
                            charisma=3, intelligence=3, agility=3, luck=3),
            "jjdidtiebuckle": dict(
                justice=5, judgment=5, dependability=5, initiative=5,
                decisiveness=5, tact=5, integrity=5, enthusiasm=5,
                bearing=5, unselfishness=5, courage=5, knowledge=5,
                loyalty=5, endurance=5,
            ),
            "drives": dict(
                compensation=5, mission=5, recognition=5, security=5,
                growth=5, comfort=5, social_status=5, autonomy=5,
                priority_order=[
                    "compensation", "mission", "recognition", "security",
                    "growth", "comfort", "social_status", "autonomy",
                ],
            ),
        }],
        "llm": {
            "enabled": True,
            "model": "llama3:8b",
            "base_url": "http://myhost:11434",
        },
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
    ) as f:
        yaml.dump(config, f)
        path = f.name

    try:
        w = World.from_config(path, seed=1)
        assert w.llm_config is not None
        assert w.llm_config.enabled is True
        assert w.llm_config.model == "llama3:8b"
        assert w.llm_config.base_url == "http://myhost:11434"
        h = w.human_list[0]
        assert h.llm_config is not None
        assert h.llm_config.enabled is True
    finally:
        os.unlink(path)


@patch("digital_human.llm.generate_memory_node", return_value="Today felt routine.")
def test_world_calls_internal_dialogue(mock_gen):
    w = World.from_config(
        "configs/test_profiles.yaml", seed=42,
        llm_config=LLMConfig(enabled=True),
    )
    w.run(days=3)

    for human in w.human_list:
        assert len(human.memory_nodes) == 3
        for day, text in human.memory_nodes:
            assert text == "Today felt routine."


def _make_bridge_human(memory_nodes: list[tuple[int, str]]) -> DigitalHuman:
    """comfort=9 passes the REDUCE_EFFORT gate; intelligence=10 zeroes
    the scoring noise so scores are deterministic and comparable."""
    h = DigitalHuman(
        human_id="bridge_h",
        name="Bridge",
        special=SPECIAL(5, 5, 5, 5, 10, 5, 5),
        jjdidtiebuckle=JJDIDTIEBUCKLE(
            justice=5, judgment=5, dependability=5, initiative=5,
            decisiveness=5, tact=5, integrity=5, enthusiasm=5,
            bearing=5, unselfishness=5, courage=5, knowledge=5,
            loyalty=5, endurance=5,
        ),
        drives=Drives(comfort=9.0),
        work_context=WorkContext(role="Analyst"),
    )
    h.memory_nodes = memory_nodes
    return h


def test_keyword_bridge_quit():
    """Quit-flavored memory nodes add exactly +2.0 to REDUCE_EFFORT
    and REST relative to an identical human without them."""
    def scores(h: DigitalHuman) -> dict[ActionType, float]:
        h.perceive(Stimulus(sim_day=3))
        return {o.action_type: o.score for o in h.understand()}

    base = scores(_make_bridge_human([]))
    boosted = scores(_make_bridge_human(
        [(1, "I'm thinking about quitting"), (2, "Maybe leaving is best")]
    ))

    assert ActionType.REDUCE_EFFORT in base, (
        "comfort=9 human must generate the reduce_effort option"
    )
    assert boosted[ActionType.REDUCE_EFFORT] == pytest.approx(
        base[ActionType.REDUCE_EFFORT] + 2.0
    )
    assert boosted[ActionType.REST] == pytest.approx(
        base[ActionType.REST] + 2.0
    )


def test_keyword_bridge_requires_word_boundary():
    """'quite'/'mosquito' must NOT trigger the quit bridge."""
    def scores(h: DigitalHuman) -> dict[ActionType, float]:
        h.perceive(Stimulus(sim_day=3))
        return {o.action_type: o.score for o in h.understand()}

    base = scores(_make_bridge_human([]))
    near_miss = scores(_make_bridge_human(
        [(1, "I'm quite happy, no mosquito problems here")]
    ))

    assert near_miss[ActionType.REDUCE_EFFORT] == pytest.approx(
        base[ActionType.REDUCE_EFFORT]
    )
    assert near_miss[ActionType.REST] == pytest.approx(base[ActionType.REST])


def test_keyword_bridge_confront():
    h = DigitalHuman(
        human_id="test_h",
        name="Tester",
        special=SPECIAL(5, 5, 5, 5, 5, 5, 5),
        jjdidtiebuckle=JJDIDTIEBUCKLE(
            justice=5, judgment=5, dependability=5, initiative=5,
            decisiveness=5, tact=5, integrity=5, enthusiasm=5,
            bearing=5, unselfishness=5, courage=7, knowledge=5,
            loyalty=5, endurance=5,
        ),
        drives=Drives(),
    )
    h.memory_nodes = [(1, "I'm fed up with this")]
    stim = Stimulus(
        sim_day=2,
        observable_others=[{"id": "other", "stress": 10, "workload": 2,
                            "apparent_regard": 0.5, "authority_level": 0}],
    )
    h.perceive(stim)
    options = h.understand()
    confront_opts = [o for o in options if o.action_type == ActionType.CONFRONT]
    assert len(confront_opts) > 0
    assert confront_opts[0].score > 0
