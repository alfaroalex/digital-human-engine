"""Tests for behavioral tuning — action variety, relationship differentiation,
delegation cost, behavioral consequences, warmth erosion, social stress,
confrontation triggering, and confrontation effects."""


import pytest

from digital_human.human import DigitalHuman
from digital_human.types import Action, ActionType, Relationship
from digital_human.world import World


@pytest.fixture(scope="module")
def world_1000():
    world = World.from_config("configs/test_profiles.yaml", seed=42)
    world.run(days=1000)
    return world


@pytest.fixture(scope="module")
def world_500():
    world = World.from_config("configs/test_profiles.yaml", seed=42)
    world.run(days=500)
    return world


class TestActionVariety:
    def test_no_action_exceeds_70_percent(self, world_1000):
        ds = world_1000.dataset
        for hid in ("human_a", "human_b", "human_c"):
            counts = ds.action_counts[hid]
            total = sum(counts.values())
            for action_name, count in counts.items():
                pct = count / total * 100
                assert pct <= 70, (
                    f"{hid}: {action_name} = {pct:.1f}% (> 70%)"
                )

    def test_at_least_4_distinct_actions_per_human(self, world_1000):
        ds = world_1000.dataset
        for hid in ("human_a", "human_b", "human_c"):
            counts = ds.action_counts[hid]
            distinct = len([k for k, v in counts.items() if v > 0])
            assert distinct >= 4, (
                f"{hid}: only {distinct} distinct actions: {dict(counts)}"
            )


class TestRelationshipDifferentiation:
    def test_relationships_not_all_one(self, world_1000):
        for hid in ("human_a", "human_b", "human_c"):
            human = world_1000.humans[hid]
            for other_id, rel in human.relationships.items():
                dims = [rel.trust, rel.respect, rel.cooperation, rel.warmth]
                assert not all(d >= 0.99 for d in dims), (
                    f"{hid} -> {other_id}: all dimensions near 1.0: "
                    f"trust={rel.trust:.3f} respect={rel.respect:.3f} "
                    f"cooperation={rel.cooperation:.3f} warmth={rel.warmth:.3f}"
                )

    def test_behavioral_adjustments_accumulated(self, world_1000):
        has_adjustments = False
        for hid in ("human_a", "human_b", "human_c"):
            human = world_1000.humans[hid]
            for _other_id, rel in human.relationships.items():
                if rel.behavioral_adjustments:
                    has_adjustments = True
                    break
            if has_adjustments:
                break
        assert has_adjustments, "At least one relationship should have behavioral adjustments"


def _load_humans() -> list[DigitalHuman]:
    import yaml
    with open("configs/test_profiles.yaml") as f:
        configs = yaml.safe_load(f)
    return [DigitalHuman.from_config(h) for h in configs["humans"]]


class TestDelegationCost:
    def test_delegation_consequence_hits_resolved_target(self):
        """Direct mechanism: a DELEGATE action accumulates a negative
        cooperation adjustment on the resolved target's ledger — for
        both sides of the relationship."""
        world = World(humans=_load_humans(), seed=42)

        action = Action(
            action_type=ActionType.DELEGATE,
            reasoning="test",
            target_id="human_b",
            sim_day=1,
        )
        world._apply_behavioral_consequences({"human_a": action})

        rel_ba = world.humans["human_b"].relationships["human_a"]
        assert rel_ba.behavioral_adjustments["cooperation"] < 0
        rel_ab = world.humans["human_a"].relationships["human_b"]
        assert rel_ab.behavioral_adjustments["cooperation"] < 0

    def test_delegation_becomes_harder(self):
        world = World(humans=_load_humans(), seed=42)
        world.run(days=200)

        delegations_a = sum(
            1 for s in world.dataset.daily_snapshots
            if s.human_id == "human_a" and s.action_taken == "delegate"
        )
        # Precondition asserted, not guarded — this test must not
        # pass vacuously.
        assert delegations_a > 5, (
            f"Expected Jordan to delegate >5 times in 200 days "
            f"(seed=42), got {delegations_a}"
        )

        rel_ba = world.humans["human_b"].relationships.get("human_a")
        assert rel_ba is not None
        coop_adj = rel_ba.behavioral_adjustments.get("cooperation", 0.0)
        assert coop_adj < 0, (
            f"B's cooperation adjustment toward A should be negative "
            f"after {delegations_a} delegations: {coop_adj}"
        )


class TestBehavioralConsequences:
    def test_cut_corners_consequence_direct(self):
        """Direct mechanism: one CUT_CORNERS action accumulates
        negative trust/respect/warmth adjustments on every observer
        the cutter has a relationship with."""
        world = World(humans=_load_humans(), seed=42)
        hb = world.humans["human_b"]
        hb.relationships["human_a"] = Relationship(target_id="human_a")
        hb.relationships["human_c"] = Relationship(target_id="human_c")

        action = Action(
            action_type=ActionType.CUT_CORNERS, reasoning="test", sim_day=1,
        )
        world._apply_behavioral_consequences({"human_b": action})

        for other_id in ("human_a", "human_c"):
            rel = world.humans[other_id].relationships["human_b"]
            assert rel.behavioral_adjustments["trust"] < 0
            assert rel.behavioral_adjustments["respect"] < 0
            assert rel.behavioral_adjustments["warmth"] < 0

    def test_cut_corners_damages_reputation(self):
        world = World(humans=_load_humans(), seed=42)
        world.run(days=500)

        cuts = world.dataset.action_counts["human_b"].get("cut_corners", 0)
        # Precondition asserted, not guarded.
        assert cuts > 10, (
            f"Expected Casey to cut corners >10 times in 500 days "
            f"(seed=42), got {cuts}"
        )

        for other_id in ("human_a", "human_c"):
            rel = world.humans[other_id].relationships.get("human_b")
            assert rel is not None, f"{other_id} should know human_b"
            trust_adj = rel.behavioral_adjustments.get("trust", 0.0)
            respect_adj = rel.behavioral_adjustments.get("respect", 0.0)
            assert trust_adj < 0 or respect_adj < 0, (
                f"{other_id}'s trust/respect adjustments toward B "
                f"should be negative after {cuts} cut_corners: "
                f"trust_adj={trust_adj:.4f}, respect_adj={respect_adj:.4f}"
            )


class TestWarmthErosion:
    def test_warmth_erodes_with_low_respect(self, world_1000):
        ha = world_1000.humans["human_a"]
        rel_ab = ha.relationships.get("human_b")
        assert rel_ab is not None
        assert rel_ab.warmth < 0.7, (
            f"Jordan's warmth toward Casey should erode below 0.7 "
            f"due to corner-cutting: {rel_ab.warmth:.3f}"
        )


class TestSocialStress:
    def test_social_stress_from_poor_reputation(self, world_1000):
        stress_traj = world_1000.dataset.get_human_trajectory("human_b", "stress_score")
        late_stress = [s for day, s in stress_traj if day > 200]
        avg_late_stress = sum(late_stress) / len(late_stress) if late_stress else 0
        assert avg_late_stress > 15, (
            f"Casey's average late-game stress should be > 15 from social pressure: "
            f"{avg_late_stress:.1f}"
        )


class TestTargetConsistency:
    def test_confrontation_stress_hits_conversation_partner(self):
        """Bug #1 regression: the target is resolved exactly once.
        The person confronted in conversation is the person who gets
        the 'confronted' stress — never a different random pick."""
        world = World(humans=_load_humans(), seed=42)
        world.sim_day = 1

        action = Action(
            action_type=ActionType.CONFRONT, reasoning="test", sim_day=1,
        )
        actions = {"human_a": action}

        world._route_social_actions(actions)
        partner = action.target_id
        assert partner in ("human_b", "human_c"), (
            "routing must resolve and write back the target"
        )
        conv_events = [
            e for e in world.dataset.conversation_events if e.sim_day == 1
        ]
        assert conv_events and conv_events[0].receiver_id == partner

        world._apply_behavioral_consequences(actions)
        assert world.humans[partner].stress.sources.get("confronted") == 5.0
        other = ({"human_b", "human_c"} - {partner}).pop()
        assert "confronted" not in world.humans[other].stress.sources


class TestConfrontation:
    def test_confrontation_triggers_from_low_respect(self, world_1000):
        ds = world_1000.dataset
        confronts_a = ds.action_counts["human_a"].get("confront", 0)
        assert confronts_a > 0, (
            f"Jordan (justice=8, courage=7) should confront Casey at least once: "
            f"confront count={confronts_a}"
        )

    def test_confrontation_reduces_corner_cutting_temporarily(self, world_500):
        ds = world_500.dataset

        confrontation_days = [
            s.sim_day for s in ds.daily_snapshots
            if s.human_id == "human_a" and s.action_taken == "confront"
        ]

        if not confrontation_days:
            pytest.skip("No confrontations occurred in 500-day sim")

        all_cc_days = {
            s.sim_day for s in ds.daily_snapshots
            if s.human_id == "human_b" and s.action_taken == "cut_corners"
        }

        total_cc = len(all_cc_days)
        total_days = 500
        overall_rate = total_cc / total_days

        post_confront_days = 0
        post_confront_cc = 0
        for cday in confrontation_days:
            for offset in range(1, 8):
                check_day = cday + offset
                if check_day <= total_days:
                    post_confront_days += 1
                    if check_day in all_cc_days:
                        post_confront_cc += 1

        if post_confront_days >= 7:
            post_rate = post_confront_cc / post_confront_days
            assert post_rate < overall_rate, (
                f"Casey should cut fewer corners after confrontation: "
                f"post_rate={post_rate:.2f}, overall={overall_rate:.2f}"
            )
