"""Tests for The World (Step 4).

Spec §7 Step 4: 3 humans in The World for 1000 days. Verify dataset
contains decision traces, conversation logs, relationship evolution,
stress trajectories. Verify emergent behavioral differences.
"""

import random
from collections import Counter

import pytest

from digital_human.human import DigitalHuman
from digital_human.world import (
    World,
    Dataset,
    DailySnapshot,
    ConversationEvent,
    RelationshipSnapshot,
    TaskGenerationConfig,
    InteractionRule,
)
from digital_human.types import ActionType


@pytest.fixture(scope="module")
def world_1000():
    world = World.from_config("configs/test_profiles.yaml", seed=42)
    world.run(days=1000)
    return world


@pytest.fixture
def world_short():
    world = World.from_config("configs/test_profiles.yaml", seed=42)
    world.run(days=50)
    return world


class TestWorldSetup:
    def test_loads_three_humans(self):
        world = World.from_config("configs/test_profiles.yaml", seed=42)
        assert len(world.humans) == 3
        assert "human_a" in world.humans
        assert "human_b" in world.humans
        assert "human_c" in world.humans

    def test_humans_have_correct_names(self):
        world = World.from_config("configs/test_profiles.yaml", seed=42)
        assert world.humans["human_a"].name == "Jordan"
        assert world.humans["human_b"].name == "Casey"
        assert world.humans["human_c"].name == "Riley"

    def test_default_interaction_rules_generated(self):
        world = World.from_config("configs/test_profiles.yaml", seed=42)
        assert len(world.interaction_rules) == 6

    def test_sim_day_advances(self, world_short):
        assert world_short.sim_day == 50

    def test_tick_advances_one_day(self):
        world = World.from_config("configs/test_profiles.yaml", seed=42)
        assert world.sim_day == 0
        world.tick()
        assert world.sim_day == 1
        world.tick()
        assert world.sim_day == 2


class TestDatasetContents:
    def test_decision_traces_present(self, world_1000):
        ds = world_1000.dataset
        assert len(ds.daily_snapshots) == 3000

    def test_conversation_logs_present(self, world_1000):
        ds = world_1000.dataset
        assert len(ds.conversation_events) > 0
        assert len(ds.conversation_events) > 200

    def test_relationship_evolution_present(self, world_1000):
        ds = world_1000.dataset
        assert len(ds.relationship_snapshots) > 0
        assert len(ds.relationship_snapshots) > 100

    def test_stress_trajectories_queryable(self, world_1000):
        ds = world_1000.dataset
        for hid in ("human_a", "human_b", "human_c"):
            trajectory = ds.get_human_trajectory(hid, "stress_score")
            assert len(trajectory) == 1000
            for day, score in trajectory:
                assert 0 <= score <= 100, f"{hid} day {day}: stress={score}"

    def test_action_counts_tracked(self, world_1000):
        ds = world_1000.dataset
        for hid in ("human_a", "human_b", "human_c"):
            counts = ds.action_counts[hid]
            total = sum(counts.values())
            assert total == 1000, f"{hid} should have 1000 actions, got {total}"

    def test_conversations_between_queryable(self, world_1000):
        ds = world_1000.dataset
        convs_ab = ds.get_conversations_between("human_a", "human_b")
        assert len(convs_ab) > 0

    def test_relationship_trajectory_queryable(self, world_1000):
        ds = world_1000.dataset
        traj = ds.get_relationship_trajectory("human_a", "human_b")
        assert len(traj) > 0
        days = [r.sim_day for r in traj]
        assert min(days) <= 20
        assert max(days) >= 990


class TestEmergentDifferences:
    def test_different_action_distributions(self, world_1000):
        ds = world_1000.dataset
        counts_a = ds.action_counts["human_a"]
        counts_b = ds.action_counts["human_b"]
        counts_c = ds.action_counts["human_c"]

        assert counts_a.get("cut_corners", 0) == 0, (
            f"Jordan should never cut corners: {counts_a.get('cut_corners', 0)}"
        )

        assert counts_a != counts_b, "Jordan and Casey should have different action distributions"

        socialize_c = counts_c.get("socialize", 0)
        socialize_a = counts_a.get("socialize", 0)
        assert socialize_c > socialize_a, (
            f"Riley should socialize more: Riley={socialize_c}, Jordan={socialize_a}"
        )

    def test_different_stress_patterns(self, world_1000):
        ds = world_1000.dataset
        stress_a = ds.get_human_trajectory("human_a", "stress_score")
        stress_b = ds.get_human_trajectory("human_b", "stress_score")

        avg_a = sum(s for _, s in stress_a) / len(stress_a)
        avg_b = sum(s for _, s in stress_b) / len(stress_b)

        assert avg_a != avg_b, (
            f"Stress averages should differ: Jordan={avg_a:.1f}, Casey={avg_b:.1f}"
        )

    def test_relationships_form_between_all_pairs(self, world_1000):
        for hid in ("human_a", "human_b", "human_c"):
            human = world_1000.humans[hid]
            other_ids = {"human_a", "human_b", "human_c"} - {hid}
            for other_id in other_ids:
                assert other_id in human.relationships, (
                    f"{hid} should have relationship with {other_id}"
                )
                rel = human.relationships[other_id]
                assert rel.familiarity > 0.5, (
                    f"{hid} -> {other_id} familiarity should be high after 1000 days: "
                    f"{rel.familiarity:.3f}"
                )

    def test_relationship_differentiation(self, world_1000):
        human_a = world_1000.humans["human_a"]

        rel_ab = human_a.relationships["human_b"]
        rel_ac = human_a.relationships["human_c"]

        diffs = [
            abs(rel_ab.trust - rel_ac.trust),
            abs(rel_ab.warmth - rel_ac.warmth),
            abs(rel_ab.cooperation - rel_ac.cooperation),
            abs(rel_ab.respect - rel_ac.respect),
        ]
        max_diff = max(diffs)
        assert max_diff > 0.01, (
            f"A's relationships with B and C should differentiate: "
            f"diffs={[f'{d:.3f}' for d in diffs]}"
        )

    def test_memories_accumulate(self, world_1000):
        for hid in ("human_a", "human_b", "human_c"):
            human = world_1000.humans[hid]
            assert len(human.memory.episodes) > 400, (
                f"{hid} should have many memories: {len(human.memory.episodes)}"
            )

    def test_conversation_logs_accumulate(self, world_1000):
        for hid in ("human_a", "human_b", "human_c"):
            human = world_1000.humans[hid]
            other_ids = {"human_a", "human_b", "human_c"} - {hid}
            for other_id in other_ids:
                logs = human.memory.conversation_logs.get(other_id, [])
                assert len(logs) > 50, (
                    f"{hid} should have many conversations with {other_id}: "
                    f"{len(logs)}"
                )

    def test_opinions_generated(self, world_1000):
        for hid in ("human_a", "human_b", "human_c"):
            human = world_1000.humans[hid]
            for other_id in {"human_a", "human_b", "human_c"} - {hid}:
                rel = human.relationships.get(other_id)
                if rel:
                    assert rel.opinion != "", (
                        f"{hid} should have opinion about {other_id}"
                    )


class TestTaskGeneration:
    def test_tasks_generated_each_day(self, world_short):
        total_performance = sum(
            len(world_short.humans[hid].work_context.performance_history)
            for hid in ("human_a", "human_b", "human_c")
        )
        assert total_performance > 0, "At least some tasks should have been processed"

    def test_conversation_reasons_are_valid(self, world_short):
        valid_reasons = {
            "task_handoff", "help_request", "monday_sync", "team_meeting",
            "deep_collaboration", "social", "confrontation", "escalation",
            "offer_help", "delegation",
        }
        for event in world_short.dataset.conversation_events:
            assert event.reason in valid_reasons, f"Invalid reason: {event.reason}"


class TestEdgeCases:
    def test_single_human_world(self):
        import yaml
        with open("configs/test_profiles.yaml") as f:
            configs = yaml.safe_load(f)
        human = DigitalHuman.from_config(configs["humans"][0])

        world = World(humans=[human], seed=42)
        world.run(days=100)
        assert world.sim_day == 100
        assert len(world.dataset.daily_snapshots) == 100

    def test_custom_task_config(self):
        import yaml
        with open("configs/test_profiles.yaml") as f:
            configs = yaml.safe_load(f)
        humans_list = [DigitalHuman.from_config(h) for h in configs["humans"]]

        config = TaskGenerationConfig(
            base_tasks_per_day=10,
            variance=0,
        )
        world = World(humans=humans_list, task_config=config, seed=42)
        world.run(days=10)

        total_perf = sum(
            len(world.humans[hid].work_context.performance_history)
            for hid in ("human_a", "human_b", "human_c")
        )
        assert total_perf > 30, (
            f"With 10 tasks/day over 10 days, substantial tasks should be processed: {total_perf}"
        )

    def test_custom_interaction_rules(self):
        import yaml
        with open("configs/test_profiles.yaml") as f:
            configs = yaml.safe_load(f)
        humans_list = [DigitalHuman.from_config(h) for h in configs["humans"]]

        rules = [
            InteractionRule("human_a", "human_b", "deep_collaboration", frequency=3),
        ]
        world = World(humans=humans_list, interaction_rules=rules, seed=42)
        world.run(days=30)

        convs = world.dataset.get_conversations_between("human_a", "human_b")
        assert len(convs) >= 8

    def test_zero_days(self):
        world = World.from_config("configs/test_profiles.yaml", seed=42)
        ds = world.run(days=0)
        assert world.sim_day == 0
        assert len(ds.daily_snapshots) == 0
