"""Tests for authority awareness and impulse/filter system."""

import random
from pathlib import Path

import pytest
import yaml

from digital_human.human import DigitalHuman, Stimulus
from digital_human.types import (
    JJDIDTIEBUCKLE,
    SPECIAL,
    ActionType,
    Drives,
    Task,
    WorkContext,
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


def make_stimulus(day: int, n_tasks: int = 2, others=None) -> Stimulus:
    if others is None:
        others = [
            {"id": "coworker_1", "stress": 30.0, "workload": 3,
             "apparent_regard": 0.5, "authority_level": 1},
        ]
    return Stimulus(
        sim_day=day,
        new_tasks=[
            Task(task_id=f"task_{day}_{i}", description=f"Task {i}", complexity=1.0)
            for i in range(n_tasks)
        ],
        environmental_signals=["normal_day"],
        observable_others=others,
    )


class TestAuthorityEscalation:
    def test_escalate_targets_reports_to(self, human_b):
        """Casey (reports_to=human_a) should escalate to Jordan."""
        random.seed(42)
        human_b.stress.score = 60.0
        others = [
            {"id": "human_a", "stress": 10.0, "workload": 2,
             "apparent_regard": 0.5, "authority_level": 3},
            {"id": "human_c", "stress": 20.0, "workload": 3,
             "apparent_regard": 0.5, "authority_level": 2},
        ]
        stim = make_stimulus(day=1, n_tasks=8, others=others)
        human_b.perceive(stim)
        options = human_b.understand()

        escalate_opts = [o for o in options if o.action_type == ActionType.ESCALATE]
        assert len(escalate_opts) > 0, "Casey should have escalate option when stressed with superior visible"
        assert escalate_opts[0].target_id == "human_a", (
            f"Escalation target should be reports_to (human_a), got {escalate_opts[0].target_id}"
        )

    def test_no_escalate_without_superior(self, human_a):
        """Jordan (no reports_to) should never generate escalate."""
        random.seed(42)
        human_a.stress.score = 60.0
        others = [
            {"id": "human_b", "stress": 30.0, "workload": 3,
             "apparent_regard": 0.5, "authority_level": 1},
        ]
        stim = make_stimulus(day=1, n_tasks=8, others=others)
        human_a.perceive(stim)
        options = human_a.understand()

        escalate_opts = [o for o in options if o.action_type == ActionType.ESCALATE]
        assert len(escalate_opts) == 0, (
            "Jordan (no superior) should never generate escalate option"
        )

    def test_delegate_down_not_up(self, human_b):
        """Casey (authority=1) should not delegate to Jordan (authority=3)."""
        random.seed(42)
        others = [
            {"id": "human_a", "stress": 10.0, "workload": 2,
             "apparent_regard": 0.5, "authority_level": 3},
        ]
        stim = make_stimulus(day=1, n_tasks=5, others=others)
        human_b.perceive(stim)
        options = human_b.understand()

        delegate_opts = [o for o in options if o.action_type == ActionType.DELEGATE]
        assert len(delegate_opts) == 0, (
            "Casey should not be able to delegate to Jordan (higher authority)"
        )


class TestAuthorityConfrontScore:
    def test_authority_gap_affects_confront_score(self, human_b):
        """Confronting someone with higher authority should be harder."""
        random.seed(42)
        human_b.relationships["human_a"] = __import__(
            "digital_human.types", fromlist=["Relationship"]
        ).Relationship(target_id="human_a", respect=0.1)
        # human_b has courage=2, won't normally confront

        human_high_courage = DigitalHuman(
            human_id="test_courage",
            name="Test",
            special=SPECIAL(5, 5, 5, 5, 5, 5, 5),
            jjdidtiebuckle=JJDIDTIEBUCKLE(8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 5, 5, 5),
            drives=Drives(),
            work_context=WorkContext(authority_level=1, reports_to="boss"),
        )
        from digital_human.types import Relationship
        human_high_courage.relationships["peer"] = Relationship(
            target_id="peer", respect=0.1,
        )
        human_high_courage.relationships["boss"] = Relationship(
            target_id="boss", respect=0.1,
        )

        others_peer = [
            {"id": "peer", "stress": 10.0, "workload": 2,
             "apparent_regard": 0.2, "authority_level": 1},
        ]
        others_boss = [
            {"id": "boss", "stress": 10.0, "workload": 2,
             "apparent_regard": 0.2, "authority_level": 3},
        ]

        stim_peer = make_stimulus(day=1, n_tasks=2, others=others_peer)
        human_high_courage.perceive(stim_peer)
        opts_peer = human_high_courage.understand()
        confront_peer = [o for o in opts_peer if o.action_type == ActionType.CONFRONT and o.target_id == "peer"]
        human_high_courage.act()

        stim_boss = make_stimulus(day=2, n_tasks=2, others=others_boss)
        human_high_courage.perceive(stim_boss)
        opts_boss = human_high_courage.understand()
        confront_boss = [o for o in opts_boss if o.action_type == ActionType.CONFRONT and o.target_id == "boss"]

        assert confront_peer and confront_boss, "Should have confront options for both"
        assert confront_peer[0].score > confront_boss[0].score, (
            f"Confronting peer should score higher than boss: "
            f"peer={confront_peer[0].score:.2f}, boss={confront_boss[0].score:.2f}"
        )


class TestRawImpulse:
    def test_raw_impulse_generated(self):
        """High stress + high comfort drive MUST produce an extreme
        impulse: stress_ratio = 70/57.5 > 0.8 with comfort=9 forces
        'quit' on any non-confront option, and the confront branch
        (ratio > 0.7) forces 'aggressive_confrontation'."""
        human = DigitalHuman(
            human_id="stressed",
            name="Stressed",
            special=SPECIAL(4, 4, 2, 4, 4, 4, 4),
            jjdidtiebuckle=JJDIDTIEBUCKLE(4, 4, 3, 2, 3, 3, 3, 3, 2, 3, 6, 3, 3, 3),
            drives=Drives(comfort=9.0),
            work_context=WorkContext(authority_level=1, reports_to="boss"),
        )
        human.stress.score = 70.0
        human.stress.threshold = 57.5

        stim = make_stimulus(day=1, n_tasks=6, others=[
            {"id": "coworker", "stress": 10.0, "workload": 2,
             "apparent_regard": 0.5, "authority_level": 1},
        ])
        human.perceive(stim)
        human.understand()

        chosen = human._current_options[0]
        raw = human._generate_raw_impulse(chosen)
        assert raw in DigitalHuman.EXTREME_IMPULSES, (
            f"stress_ratio > 0.8 with comfort=9 must yield an extreme "
            f"impulse, got '{raw}' for chosen {chosen.action_type}"
        )

    def test_low_stress_impulse_matches_chosen(self):
        """Below half the stress threshold, the impulse IS the chosen
        action — no extreme impulses."""
        human = DigitalHuman(
            human_id="calm",
            name="Calm",
            special=SPECIAL(4, 4, 2, 4, 4, 4, 4),
            jjdidtiebuckle=JJDIDTIEBUCKLE(4, 4, 3, 2, 3, 3, 3, 3, 2, 3, 6, 3, 3, 3),
            drives=Drives(comfort=9.0),
        )
        human.stress.score = 10.0
        human.stress.threshold = 57.5

        stim = make_stimulus(day=1, n_tasks=2)
        human.perceive(stim)
        human.understand()
        chosen = human._current_options[0]
        raw = human._generate_raw_impulse(chosen)
        assert raw == chosen.action_type.value

    def test_high_stress_comfort_seeker_gets_quit(self):
        """High-comfort person under extreme stress should get 'quit' impulse."""
        human = DigitalHuman(
            human_id="quitter",
            name="Quitter",
            special=SPECIAL(4, 4, 2, 4, 4, 4, 4),
            jjdidtiebuckle=JJDIDTIEBUCKLE(4, 4, 3, 2, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3),
            drives=Drives(comfort=9.0),
        )
        human.stress.score = 50.0
        human.stress.threshold = 57.5

        from digital_human.types import Option
        chosen = Option(action_type=ActionType.REST)
        raw = human._generate_raw_impulse(chosen)
        assert raw == "quit", f"Expected 'quit' impulse, got '{raw}'"


class TestFilter:
    def test_filter_suppresses_extreme(self):
        """High bearing+tact+judgment should convert extreme to professional."""
        human = DigitalHuman(
            human_id="composed",
            name="Composed",
            special=SPECIAL(5, 5, 8, 5, 5, 5, 5),
            jjdidtiebuckle=JJDIDTIEBUCKLE(5, 8, 5, 5, 5, 7, 5, 5, 7, 5, 5, 5, 5, 5),
            drives=Drives(),
        )
        from digital_human.types import Option
        chosen = Option(action_type=ActionType.CONFRONT)
        action_type, metadata = human._apply_filter("aggressive_confrontation", chosen)
        assert action_type == ActionType.CONFRONT
        assert metadata.get("filter_result") == "suppressed"
        assert "tone" not in metadata or metadata["tone"] != "extreme"

    def test_filter_fails_low_bearing(self):
        """Low bearing+tact+judgment should fail to suppress."""
        human = DigitalHuman(
            human_id="unfiltered",
            name="Unfiltered",
            special=SPECIAL(5, 5, 2, 5, 5, 5, 5),
            jjdidtiebuckle=JJDIDTIEBUCKLE(5, 2, 5, 5, 5, 2, 5, 5, 2, 5, 5, 5, 5, 5),
            drives=Drives(),
        )
        human.stress.score = 60.0
        human.stress.threshold = 57.5
        from digital_human.types import Option
        chosen = Option(action_type=ActionType.CONFRONT)
        action_type, metadata = human._apply_filter("aggressive_confrontation", chosen)
        assert action_type == ActionType.CONFRONT
        assert metadata.get("filter_result") == "failed"
        assert metadata.get("tone") == "extreme"


class TestSuppressionStress:
    def test_suppression_gap_creates_stress(self):
        """Large gap between impulse and action MUST add suppression
        stress. stress_ratio = 62/70 > 0.8 with comfort=9 guarantees
        an extreme impulse; bearing+tact+judgment = 23 minus a small
        endurance penalty stays > 15, so the filter suppresses it."""
        human = DigitalHuman(
            human_id="suppressor",
            name="Suppressor",
            special=SPECIAL(5, 5, 8, 5, 9, 5, 5),
            jjdidtiebuckle=JJDIDTIEBUCKLE(5, 8, 5, 5, 5, 7, 5, 5, 8, 5, 7, 5, 5, 5),
            drives=Drives(comfort=9.0),
            work_context=WorkContext(authority_level=1, reports_to="boss"),
        )
        human.stress.score = 62.0
        human.stress.threshold = 70.0

        stim = make_stimulus(day=1, n_tasks=6, others=[
            {"id": "coworker", "stress": 10.0, "workload": 2,
             "apparent_regard": 0.5, "authority_level": 1},
        ])
        human.perceive(stim)
        human.understand()
        human.act()

        episodes = list(human.memory.episodes)
        # Precondition asserted: an extreme impulse must have fired.
        assert episodes[-1].raw_impulse in DigitalHuman.EXTREME_IMPULSES, (
            f"Expected an extreme impulse at stress ratio > 0.8, "
            f"got {episodes[-1].raw_impulse!r}"
        )
        assert human.stress.sources.get("suppression", 0) > 0, (
            "Suppression stress should be > 0 after filtering an extreme impulse"
        )


class TestRawImpulseInMemory:
    def test_raw_impulse_logged_in_memory(self):
        """Memory should contain both the action and raw impulse."""
        human = DigitalHuman(
            human_id="logger",
            name="Logger",
            special=SPECIAL(5, 5, 5, 5, 5, 5, 5),
            jjdidtiebuckle=JJDIDTIEBUCKLE(5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5),
            drives=Drives(),
        )
        stim = make_stimulus(day=1, n_tasks=2)
        human.daily_cycle(stim)

        episodes = list(human.memory.episodes)
        assert len(episodes) > 0
        last = episodes[-1]
        assert hasattr(last, "raw_impulse"), "Episode should have raw_impulse field"
