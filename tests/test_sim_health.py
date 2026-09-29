"""Long-horizon simulation-health invariants (1000 days, seed=1).

Each test guards a model invariant that should hold over long runs.
They are marked strict-xfail against the currently known bugs B1–B4;
remove each marker when its bug is fixed.
"""

import copy

import pytest

from digital_human.types import ActionType
from digital_human.world import World

RELATIONSHIP_DIMS = ("trust", "respect", "warmth", "cooperation")


@pytest.fixture(scope="module")
def sim_health():
    world = World.from_config("configs/test_profiles.yaml", seed=1)

    delegations = []
    recomputes = []

    original_route = world._route_social_actions

    def route_spy(actions):
        original_route(actions)
        for actor_id, action in actions.items():
            if action.action_type == ActionType.DELEGATE:
                target_id = action.target_id
                actor_auth = world.humans[actor_id].work_context.authority_level
                target_auth = (
                    world.humans[target_id].work_context.authority_level
                    if target_id in world.humans
                    else None
                )
                delegations.append(
                    (world.sim_day, actor_id, target_id, actor_auth, target_auth)
                )

    original_recompute = world._recompute_relationships

    def recompute_spy():
        before = {
            (human.human_id, other_id): {dim: getattr(rel, dim) for dim in RELATIONSHIP_DIMS}
            for human in world.human_list
            for other_id, rel in human.relationships.items()
        }
        before = copy.deepcopy(before)
        original_recompute()
        for human in world.human_list:
            for other_id, rel in human.relationships.items():
                for dim in RELATIONSHIP_DIMS:
                    recomputes.append(
                        (
                            world.sim_day,
                            human.human_id,
                            other_id,
                            dim,
                            before[(human.human_id, other_id)][dim],
                            getattr(rel, dim),
                        )
                    )

    world._route_social_actions = route_spy
    world._recompute_relationships = recompute_spy

    world.run(days=1000)
    return world, delegations, recomputes


@pytest.mark.xfail(strict=True, reason="B1: relationship dimensions saturate at 1.0")
def test_relationship_dimensions_do_not_saturate(sim_health):
    world, _, _ = sim_health
    values = [
        getattr(snap, dim)
        for snap in world.dataset.relationship_snapshots
        if snap.sim_day >= 100
        for dim in RELATIONSHIP_DIMS
    ]
    saturated = sum(1 for v in values if v >= 0.99)
    share = saturated / len(values)
    assert share <= 0.10, f"saturated share {share:.4f} ({saturated}/{len(values)})"


@pytest.mark.xfail(strict=True, reason="B1: recompute jumps relationship dims by > 0.05")
def test_recompute_does_not_jump_relationships(sim_health):
    _, _, recomputes = sim_health
    assert recomputes, "no recomputes recorded"
    worst = max(recomputes, key=lambda r: abs(r[5] - r[4]))
    assert abs(worst[5] - worst[4]) <= 0.05, (
        f"max recompute jump {abs(worst[5] - worst[4]):.4f} "
        f"({worst[1]}->{worst[2]} {worst[3]} day {worst[0]}: {worst[4]:.4f} -> {worst[5]:.4f})"
    )


@pytest.mark.xfail(strict=True, reason="B2: bond_type does not reflect relationship dims")
def test_bond_type_reflects_relationship(sim_health):
    world, _, _ = sim_health
    final_day = max(s.sim_day for s in world.dataset.relationship_snapshots)
    final = [s for s in world.dataset.relationship_snapshots if s.sim_day == final_day]
    bond_types = {s.bond_type for s in final}
    assert len(bond_types) >= 2, f"only {len(bond_types)} bond_type(s): {bond_types}"
    violating = [(s.from_id, s.to_id, s.respect) for s in final
                 if s.respect < 0.3 and s.bond_type == "strong"]
    assert not violating, f"{len(violating)} strong-bond pairs with respect < 0.3: {violating}"


@pytest.mark.xfail(strict=True, reason="B3: delegation targets equal/higher authority")
def test_delegation_respects_authority(sim_health):
    _, delegations, _ = sim_health
    assert delegations, "no delegations recorded"
    violations = [d for d in delegations if d[4] is not None and d[4] >= d[3]]
    assert not violations, (
        f"{len(violations)}/{len(delegations)} delegations to target with "
        f"authority >= actor: {violations[:5]}"
    )


@pytest.mark.xfail(strict=True, reason="B4: long-term memory never populated")
def test_long_term_memory_is_used(sim_health):
    world, _, _ = sim_health
    counts = {h.human_id: len(h.memory.long_term) for h in world.human_list}
    assert all(n > 0 for n in counts.values()), f"long_term counts: {counts}"
    late = [
        e.sim_day
        for h in world.human_list
        for e in h.memory.long_term
        if e.sim_day > 500
    ]
    assert late, f"no long_term episode with sim_day > 500; counts {counts}"
