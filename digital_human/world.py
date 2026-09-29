"""The World — simulation environment for Digital Humans.

Spec §5. The World provides stimuli, runs the clock, routes events,
and records everything. It does NOT compute stress, manage relationships,
or make decisions for humans.

The World DOES execute actions — when a human returns an Action, The World
processes its effects (task dequeuing, conversation initiation, etc.).
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import yaml

from digital_human.conversation import (
    Conversation,
    apply_conversation_result,
    apply_recomputed_relationship,
    recompute_from_logs,
)
from digital_human.human import DigitalHuman, Stimulus
from digital_human.types import (
    SOCIAL_ACTIONS,
    SPECIAL_TRAIT_NAMES,
    ActionType,
    Episode,
    LLMConfig,
    Relationship,
    Task,
    TaskResult,
)

TASK_TEMPLATES = [
    {"description": "data analysis",
     "requirements": {"intelligence": 6, "knowledge": 5}},
    {"description": "client call",
     "requirements": {"charisma": 5, "tact": 6}},
    {"description": "code review",
     "requirements": {"intelligence": 7, "knowledge": 7, "perception": 6}},
    {"description": "team presentation",
     "requirements": {"charisma": 6, "enthusiasm": 5, "bearing": 5}},
    {"description": "inventory check",
     "requirements": {"dependability": 5, "endurance": 4}},
    {"description": "conflict resolution",
     "requirements": {"tact": 7, "judgment": 6, "courage": 5}},
    {"description": "strategic planning",
     "requirements": {"intelligence": 7, "judgment": 7, "initiative": 6}},
    {"description": "routine filing",
     "requirements": {"dependability": 4}},
    {"description": "mentoring session",
     "requirements": {"knowledge": 7, "tact": 6, "unselfishness": 5}},
    {"description": "deadline crunch",
     "requirements": {"endurance": 6, "decisiveness": 5, "bearing": 5}},
]


def _get_trait_score(human: DigitalHuman, trait_name: str) -> int:
    if trait_name in SPECIAL_TRAIT_NAMES:
        return getattr(human.special, trait_name)
    return getattr(human.jjdidtiebuckle, trait_name, 5)


@dataclass
class DailySnapshot:
    sim_day: int
    human_id: str
    action_taken: str
    stress_score: float
    stress_sources: dict[str, float]
    task_queue_size: int
    tasks_completed: int
    feelings: dict[str, float]
    beliefs_count: int
    memory_count: int
    performance_history_len: int


@dataclass
class ConversationEvent:
    sim_day: int
    initiator_id: str
    receiver_id: str
    reason: str
    cycles: int
    total_score: float
    tasks_transferred: int


@dataclass
class RelationshipSnapshot:
    sim_day: int
    from_id: str
    to_id: str
    trust: float
    respect: float
    cooperation: float
    familiarity: float
    warmth: float
    gratitude: float
    bond_type: str
    conversation_history_score: float


@dataclass
class Dataset:
    daily_snapshots: list[DailySnapshot] = field(default_factory=list)
    conversation_events: list[ConversationEvent] = field(default_factory=list)
    relationship_snapshots: list[RelationshipSnapshot] = field(default_factory=list)
    action_counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    task_results: list[TaskResult] = field(default_factory=list)

    def get_human_trajectory(
        self, human_id: str, metric: str = "stress_score",
    ) -> list[tuple[int, float]]:
        return [
            (s.sim_day, getattr(s, metric))
            for s in self.daily_snapshots
            if s.human_id == human_id and hasattr(s, metric)
        ]

    def get_conversations_between(self, id_a: str, id_b: str) -> list[ConversationEvent]:
        return [
            c for c in self.conversation_events
            if (c.initiator_id == id_a and c.receiver_id == id_b)
            or (c.initiator_id == id_b and c.receiver_id == id_a)
        ]

    def get_relationship_trajectory(self, from_id: str, to_id: str) -> list[RelationshipSnapshot]:
        return [
            r for r in self.relationship_snapshots
            if r.from_id == from_id and r.to_id == to_id
        ]

    def get_human_task_performance(self, human_id: str) -> list[TaskResult]:
        return [r for r in self.task_results if r.human_id == human_id]


@dataclass
class TaskGenerationConfig:
    base_tasks_per_day: int = 3
    variance: int = 2
    complexity_range: tuple[float, float] = (0.3, 1.0)
    per_human_load: dict[str, int] = field(default_factory=dict)


@dataclass
class InteractionRule:
    human_a_id: str
    human_b_id: str
    reason: str
    frequency: int = 1


class World:
    """The simulation environment.

    Runs the clock, delivers stimuli, executes actions returned by
    humans, routes conversations, and collects all data.
    """

    def __init__(
        self,
        humans: list[DigitalHuman],
        task_config: TaskGenerationConfig | None = None,
        interaction_rules: list[InteractionRule] | None = None,
        seed: int | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self.humans = {h.human_id: h for h in humans}
        self.human_list = humans
        self.task_config = task_config or TaskGenerationConfig()
        self.interaction_rules = interaction_rules or []
        self.dataset = Dataset()
        self.sim_day = 0
        self.error_counts: dict[str, int] = defaultdict(int)
        self._quality_review_flags: set[str] = set()
        self.llm_config = llm_config
        self._task_template_cache: dict[str, list[dict[str, Any]]] = {}

        # Per-instance RNG — two Worlds in one process must not
        # corrupt each other's determinism via the module-global RNG.
        self.rng = random.Random(seed)
        for human in self.human_list:
            human._rng = self.rng

        if not self.interaction_rules:
            self.interaction_rules = self._default_interaction_rules()

        if self.llm_config and self.llm_config.enabled:
            self._generate_role_templates()

    def run(self, days: int) -> Dataset:
        for _ in range(days):
            self.tick()
        return self.dataset

    def tick(self) -> None:
        self.sim_day += 1

        stimuli = self._generate_stimuli()

        actions: dict[str, Any] = {}
        for human in self.human_list:
            stimulus = stimuli[human.human_id]
            action = human.daily_cycle(stimulus)
            actions[human.human_id] = action

            # The World executes the action's effects
            self._execute_action(human, action)

            self._record_daily_snapshot(human, action)

        self._route_social_actions(actions)
        self._apply_behavioral_consequences(actions)
        self._decay_behavioral_adjustments()

        self._generate_internal_dialogues(actions)

        if self.sim_day % 30 == 0:
            self._recompute_relationships()
            self._quality_review_flags.clear()
            for hid, count in self.error_counts.items():
                if count > 5:
                    self._quality_review_flags.add(hid)
            self.error_counts.clear()

        if self.sim_day % 10 == 0:
            self._snapshot_relationships()

    # --- Action execution (The World's job, not the human's) ---

    def _execute_action(self, human: DigitalHuman, action: Any) -> None:
        """Execute the effects of a human's chosen action.

        The human decided what to do. The World makes it happen:
        dequeuing tasks, recording performance, etc.
        """
        for source in ("task_failure", "task_struggle", "confronted"):
            if source in human.stress.sources:
                human.stress.sources[source] *= 0.7

        task_actions = {
            ActionType.PROCESS_TASK,
            ActionType.CUT_CORNERS,
            ActionType.PUSH_THROUGH,
        }

        if action.action_type not in task_actions:
            return

        capacity = human.work_context.capacity_threshold
        queue_len = len(human.work_context.task_queue)

        if action.action_type in (ActionType.PUSH_THROUGH, ActionType.CUT_CORNERS):
            tasks_to_process = min(queue_len, capacity + 2)
        else:
            tasks_to_process = min(queue_len, capacity)

        for _ in range(tasks_to_process):
            if not human.work_context.task_queue:
                break
            task = human.work_context.task_queue.popleft()

            result = self._compute_task_result(human, task, action.action_type)
            self.dataset.task_results.append(result)
            human.work_context.performance_history.append(result.quality)
            self._apply_task_feedback(human, task, result)

            if action.action_type == ActionType.CUT_CORNERS:
                error_chance = (10 - human.jjdidtiebuckle.knowledge) / 10.0 * 0.4
                if self.rng.random() < error_chance:
                    self.error_counts[human.human_id] += 1

            action.task_id = task.task_id

    def _compute_task_result(
        self,
        human: DigitalHuman,
        task: Task,
        action_type: ActionType,
    ) -> TaskResult:
        if not task.requirements:
            base_quality = (
                human.special.intelligence / 10.0 * 0.4
                + human.jjdidtiebuckle.knowledge / 10.0 * 0.3
                + human.jjdidtiebuckle.dependability / 10.0 * 0.3
            )
            stress_penalty = human.stress.score / 200.0
            quality = base_quality - stress_penalty
            if action_type == ActionType.CUT_CORNERS:
                quality *= 0.6
            quality = max(0.1, min(1.0, quality))
            return TaskResult(
                task_id=task.task_id,
                human_id=human.human_id,
                sim_day=self.sim_day,
                quality=quality,
                speed=1.0,
                errors=0,
                completed=True,
                trait_match=1.0,
                details="standard task",
            )

        qualities: list[float] = []
        speeds: list[float] = []
        error_count = 0
        completed = True
        trait_matches: list[float] = []
        details_parts: list[str] = []

        for trait_name, required in task.requirements.items():
            score = _get_trait_score(human, trait_name)
            gap = score - required

            if gap >= 3:
                qualities.append(0.95)
                speeds.append(1.3)
                details_parts.append(f"exceeded requirements in {trait_name}")
            elif gap >= 1:
                qualities.append(0.8)
                speeds.append(1.0)
            elif gap == 0:
                qualities.append(0.6)
                speeds.append(1.0)
            elif gap >= -2:
                qualities.append(0.4)
                speeds.append(0.75)
                details_parts.append(f"struggled with {trait_name}")
            elif gap >= -4:
                qualities.append(0.2)
                speeds.append(0.5)
                details_parts.append(f"struggled with {trait_name}")
            else:
                completed = False
                qualities.append(0.0)
                speeds.append(0.3)
                details_parts.append(f"could not handle {trait_name}")

            if gap <= -2 and self.rng.random() < 0.5:
                error_count += 1

            trait_matches.append(min(1.0, score / max(1, required)))

        quality = sum(qualities) / len(qualities)
        speed = sum(speeds) / len(speeds)
        trait_match = sum(trait_matches) / len(trait_matches)

        stress_penalty = human.stress.score / 200.0
        quality = max(0.0, min(1.0, quality - stress_penalty))

        if action_type == ActionType.CUT_CORNERS:
            quality *= 0.6

        return TaskResult(
            task_id=task.task_id,
            human_id=human.human_id,
            sim_day=self.sim_day,
            quality=quality,
            speed=speed,
            errors=error_count,
            completed=completed,
            trait_match=trait_match,
            details=", ".join(details_parts) if details_parts else "adequate performance",
        )

    def _apply_task_feedback(
        self, human: DigitalHuman, task: Task, result: TaskResult,
    ) -> None:
        if result.quality > 0.8 and task.importance > 7:
            human.inner_world.update_feeling("job_satisfaction", 0.02)
            human.inner_world.update_feeling("motivation", 0.01)

        if not result.completed:
            human.stress.sources["task_failure"] = min(
                30.0, human.stress.sources.get("task_failure", 0.0) + 10.0,
            )
            human.inner_world.update_feeling("motivation", -0.03)
            human.memory.record(Episode(
                sim_day=self.sim_day,
                event_type="task_failure",
                content=(
                    f"I couldn't handle that {task.description}. "
                    f"{result.details}"
                ),
                emotional_valence=-0.7,
                tags=["task_failure", "work"],
            ))
        elif result.speed < 0.7:
            human.stress.sources["task_struggle"] = min(
                15.0, human.stress.sources.get("task_struggle", 0.0) + 3.0,
            )
        elif result.quality > 0.8:
            human.memory.record(Episode(
                sim_day=self.sim_day,
                event_type="task_success",
                content=(
                    f"I crushed that {task.description} today. "
                    f"{result.details}"
                ),
                emotional_valence=0.5,
                tags=["task_success", "work"],
            ))

    # --- Config loading ---

    @classmethod
    def from_config(
        cls,
        config_path: str,
        seed: int | None = None,
        llm_config: LLMConfig | None = None,
    ) -> World:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        if llm_config is None and "llm" in config:
            llm_section = config["llm"]
            llm_config = LLMConfig(
                enabled=llm_section.get("enabled", False),
                model=llm_section.get("model", "qwen2.5:7b-instruct"),
                base_url=llm_section.get("base_url", "http://localhost:11434"),
                temperature=llm_section.get("temperature", 0.8),
                max_tokens=llm_section.get("max_tokens", 150),
            )

        humans = []
        human_configs = config.get("humans", [])
        if isinstance(human_configs, dict):
            for hid, hcfg in human_configs.items():
                hcfg.setdefault("id", hid)
                humans.append(DigitalHuman.from_config(hcfg, llm_config=llm_config))
        else:
            for hcfg in human_configs:
                humans.append(DigitalHuman.from_config(hcfg, llm_config=llm_config))

        task_config = None
        if "task_generation" in config:
            tc = config["task_generation"]
            task_config = TaskGenerationConfig(
                base_tasks_per_day=tc.get("base_tasks_per_day", 3),
                variance=tc.get("variance", 2),
                per_human_load=tc.get("per_human_load", {}),
            )

        interaction_rules = []
        for rule in config.get("interaction_rules", []):
            interaction_rules.append(InteractionRule(
                human_a_id=rule["human_a"],
                human_b_id=rule["human_b"],
                reason=rule["reason"],
                frequency=rule.get("frequency", 1),
            ))

        return cls(
            humans=humans,
            task_config=task_config,
            interaction_rules=interaction_rules if interaction_rules else None,
            seed=seed,
            llm_config=llm_config,
        )

    # --- Task template generation ---

    def _generate_role_templates(self) -> None:
        import logging

        from digital_human.llm import generate_tasks

        if self.llm_config is None:
            return

        for human in self.human_list:
            role = human.work_context.role
            if role in self._task_template_cache:
                continue
            try:
                templates = generate_tasks(
                    self.llm_config,
                    role=role,
                    responsibilities=human.work_context.responsibilities,
                    n_tasks=20,
                )
                if templates:
                    self._task_template_cache[role] = templates
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "LLM task gen failed for role %s: %s", role, e,
                )

    def _get_templates_for_role(self, role: str) -> list[dict[str, Any]]:
        return self._task_template_cache.get(role, TASK_TEMPLATES)

    # --- Stimulus generation ---

    def _generate_stimuli(self) -> dict[str, Stimulus]:
        stimuli = {}
        tc = self.task_config

        for human in self.human_list:
            base = tc.per_human_load.get(human.human_id, tc.base_tasks_per_day)
            n_tasks = max(0, base + self.rng.randint(-tc.variance, tc.variance))

            tasks = []
            templates = self._get_templates_for_role(human.work_context.role)
            for i in range(n_tasks):
                template = self.rng.choice(templates)
                complexity = self.rng.uniform(*tc.complexity_range)
                scaled_reqs = {
                    k: min(10, max(1, round(v * complexity)))
                    for k, v in template["requirements"].items()
                }
                tasks.append(Task(
                    task_id=f"d{self.sim_day}_{human.human_id}_{i}",
                    description=template["description"],
                    complexity=complexity,
                    importance=self.rng.uniform(3.0, 9.0),
                    requirements=scaled_reqs,
                ))

            others = []
            for other in self.human_list:
                if other.human_id != human.human_id:
                    apparent = 0.5
                    rel = human.relationships.get(other.human_id)
                    if rel:
                        apparent = (rel.trust + rel.respect + rel.warmth) / 3.0
                    others.append({
                        "id": other.human_id,
                        "stress": other.stress.score,
                        "workload": len(other.work_context.task_queue),
                        "apparent_regard": apparent,
                        "authority_level": other.work_context.authority_level,
                    })

            signals = ["normal_day"]
            if self.sim_day % 7 in (6, 0):
                signals = ["weekend"]
            if self.sim_day % 30 == 1:
                signals.append("month_start")
            if human.human_id in self._quality_review_flags:
                signals.append("quality_review")

            stimuli[human.human_id] = Stimulus(
                sim_day=self.sim_day,
                new_tasks=tasks,
                environmental_signals=signals,
                observable_others=others,
            )

        return stimuli

    # --- Event routing ---

    def _route_social_actions(self, actions: dict[str, Any]) -> None:
        conversations_today: list[tuple[str, str, str]] = []

        for human_id, action in actions.items():
            if action.action_type in SOCIAL_ACTIONS:
                target_id = action.target_id
                if not target_id or target_id not in self.humans:
                    target_id = self._pick_interaction_target(human_id)
                # Resolve the target exactly once: write it back so
                # _apply_behavioral_consequences hits the same person.
                action.target_id = target_id
                if target_id:
                    reason = self._action_to_reason(action.action_type)
                    conversations_today.append((human_id, target_id, reason))

        for rule in self.interaction_rules:
            if self.sim_day % rule.frequency == 0 and (
                rule.human_a_id in self.humans and rule.human_b_id in self.humans
            ):
                conversations_today.append(
                    (rule.human_a_id, rule.human_b_id, rule.reason)
                )

        seen_pairs: set[tuple[str, str]] = set()
        for initiator_id, receiver_id, reason in conversations_today:
            pair = (
                (initiator_id, receiver_id)
                if initiator_id <= receiver_id
                else (receiver_id, initiator_id)
            )
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            initiator = self.humans[initiator_id]
            receiver = self.humans[receiver_id]

            conv = Conversation(initiator, receiver, reason, self.sim_day, rng=self.rng)
            result = conv.run()
            apply_conversation_result(initiator, receiver, result, self.sim_day)

            self.dataset.conversation_events.append(ConversationEvent(
                sim_day=self.sim_day,
                initiator_id=initiator_id,
                receiver_id=receiver_id,
                reason=reason,
                cycles=conv.expected_cycles,
                total_score=result.record_a.total_score,
                tasks_transferred=result.tasks_transferred,
            ))

    def _pick_interaction_target(self, human_id: str) -> str | None:
        human = self.humans[human_id]
        others = [h for h in self.human_list if h.human_id != human_id]
        if not others:
            return None

        weights = []
        for other in others:
            rel = human.relationships.get(other.human_id)
            if rel:
                weights.append(0.5 + rel.familiarity)
            else:
                weights.append(0.5)

        return self.rng.choices(
            [o.human_id for o in others], weights=weights, k=1
        )[0]

    @staticmethod
    def _action_to_reason(action_type: ActionType) -> str:
        mapping = {
            ActionType.DELEGATE: "delegation",
            ActionType.ESCALATE: "escalation",
            ActionType.REQUEST_HELP: "help_request",
            ActionType.CONFRONT: "confrontation",
            ActionType.OFFER_HELP: "offer_help",
            ActionType.SOCIALIZE: "social",
        }
        return mapping.get(action_type, "social")

    # --- Data collection ---

    def _record_daily_snapshot(self, human: DigitalHuman, action: Any) -> None:
        tasks_completed = len(human.work_context.performance_history)

        snapshot = DailySnapshot(
            sim_day=self.sim_day,
            human_id=human.human_id,
            action_taken=action.action_type.value,
            stress_score=human.stress.score,
            stress_sources=dict(human.stress.sources),
            task_queue_size=len(human.work_context.task_queue),
            tasks_completed=tasks_completed,
            feelings=dict(human.inner_world.feelings),
            beliefs_count=len(human.inner_world.beliefs),
            memory_count=len(human.memory.episodes),
            performance_history_len=tasks_completed,
        )

        self.dataset.daily_snapshots.append(snapshot)
        self.dataset.action_counts[human.human_id][action.action_type.value] += 1

    def _snapshot_relationships(self) -> None:
        for human in self.human_list:
            for other_id, rel in human.relationships.items():
                logs = human.memory.conversation_logs.get(other_id, [])
                bond_type = "stranger"
                if logs:
                    recomputed = recompute_from_logs(logs, "a")
                    bond_type = recomputed["bond_type"]

                self.dataset.relationship_snapshots.append(RelationshipSnapshot(
                    sim_day=self.sim_day,
                    from_id=human.human_id,
                    to_id=other_id,
                    trust=rel.trust,
                    respect=rel.respect,
                    cooperation=rel.cooperation,
                    familiarity=rel.familiarity,
                    warmth=rel.warmth,
                    gratitude=rel.gratitude,
                    bond_type=bond_type,
                    conversation_history_score=rel.conversation_history_score,
                ))

    def _recompute_relationships(self) -> None:
        for human in self.human_list:
            for other_id in list(human.relationships.keys()):
                logs = human.memory.conversation_logs.get(other_id, [])
                if logs:
                    recomputed = recompute_from_logs(logs, "a")
                    apply_recomputed_relationship(
                        human, other_id, recomputed, self.sim_day
                    )

    def _apply_behavioral_consequences(self, actions: dict[str, Any]) -> None:
        for human_id, action in actions.items():
            human = self.humans[human_id]
            at = action.action_type

            if at == ActionType.DELEGATE:
                target_id = action.target_id
                if target_id and target_id in self.humans:
                    target = self.humans[target_id]
                    if target_id not in human.relationships:
                        human.relationships[target_id] = Relationship(target_id=target_id)
                    if human_id not in target.relationships:
                        target.relationships[human_id] = Relationship(target_id=human_id)

                    penalty = -0.01
                    target_overloaded = (
                        len(target.work_context.task_queue)
                        > target.work_context.capacity_threshold
                    )
                    if target_overloaded:
                        penalty *= 2

                    rel_target = target.relationships[human_id]
                    rel_target.behavioral_adjustments["cooperation"] = (
                        rel_target.behavioral_adjustments.get("cooperation", 0.0) + penalty
                    )

                    rel_self = human.relationships[target_id]
                    rel_self.behavioral_adjustments["cooperation"] = (
                        rel_self.behavioral_adjustments.get("cooperation", 0.0) - 0.003
                    )

            elif at == ActionType.CUT_CORNERS:
                for other in self.human_list:
                    if other.human_id == human_id:
                        continue
                    if other.human_id not in human.relationships:
                        continue
                    if other.human_id in human.relationships:
                        other_rel = other.relationships.get(human_id)
                        if other_rel is None:
                            other.relationships[human_id] = Relationship(target_id=human_id)
                            other_rel = other.relationships[human_id]
                        # Magnitudes calibrated for single application
                        # with x0.995/day decay: at Casey-like cut
                        # rates (~20% of days) the respect ledger
                        # equilibrates near -0.8, low enough to cross
                        # the justice-confrontation threshold.
                        other_rel.behavioral_adjustments["trust"] = (
                            other_rel.behavioral_adjustments.get("trust", 0.0) - 0.01
                        )
                        other_rel.behavioral_adjustments["respect"] = (
                            other_rel.behavioral_adjustments.get("respect", 0.0) - 0.02
                        )
                        other_rel.behavioral_adjustments["warmth"] = (
                            other_rel.behavioral_adjustments.get("warmth", 0.0) - 0.01
                        )

            elif at == ActionType.REDUCE_EFFORT:
                for other in self.human_list:
                    if other.human_id == human_id:
                        continue
                    other_overloaded = (
                        len(other.work_context.task_queue)
                        > other.work_context.capacity_threshold
                    )
                    if other_overloaded:
                        if human_id not in other.relationships:
                            other.relationships[human_id] = Relationship(target_id=human_id)
                        other_rel = other.relationships[human_id]
                        other_rel.behavioral_adjustments["respect"] = (
                            other_rel.behavioral_adjustments.get("respect", 0.0) - 0.003
                        )

            elif at in (ActionType.PROCESS_TASK, ActionType.PUSH_THROUGH):
                for other in self.human_list:
                    if other.human_id == human_id:
                        continue
                    if human_id not in other.relationships:
                        other.relationships[human_id] = Relationship(target_id=human_id)
                    other_rel = other.relationships[human_id]
                    # Calibrated for single application: keeps the
                    # ledger equilibrium small enough that respect
                    # for a steady worker oscillates below the 1.0
                    # cap instead of pegging there.
                    other_rel.behavioral_adjustments["respect"] = (
                        other_rel.behavioral_adjustments.get("respect", 0.0) + 0.001
                    )

            elif at == ActionType.TAKE_INITIATIVE:
                for other in self.human_list:
                    if other.human_id == human_id:
                        continue
                    if human_id not in other.relationships:
                        other.relationships[human_id] = Relationship(target_id=human_id)
                    other_rel = other.relationships[human_id]
                    other_rel.behavioral_adjustments["trust"] = (
                        other_rel.behavioral_adjustments.get("trust", 0.0) + 0.0015
                    )

            elif at == ActionType.OFFER_HELP:
                target_id = action.target_id
                if target_id and target_id in self.humans:
                    if human_id not in self.humans[target_id].relationships:
                        self.humans[target_id].relationships[human_id] = Relationship(target_id=human_id)
                    other_rel = self.humans[target_id].relationships[human_id]
                    other_rel.behavioral_adjustments["warmth"] = (
                        other_rel.behavioral_adjustments.get("warmth", 0.0) + 0.005
                    )
                    other_rel.behavioral_adjustments["gratitude"] = (
                        other_rel.behavioral_adjustments.get("gratitude", 0.0) + 0.005
                    )

            elif at == ActionType.CONFRONT:
                target_id = action.target_id
                if target_id and target_id in self.humans:
                    target = self.humans[target_id]
                    target.stress.sources["confronted"] = 5.0
                    target._last_confronted_day = self.sim_day
                    if "injustice" in human.stress.sources:
                        human.stress.sources["injustice"] = max(
                            0.0, human.stress.sources["injustice"] - 5.0
                        )

    def _decay_behavioral_adjustments(self) -> None:
        """Reputations heal: the adjustment ledger decays x0.995/day
        (half-life ~140 days). The ledger is applied to relationship
        values only at the 30-day recompute — never live — so each
        delta counts exactly once."""
        for human in self.human_list:
            for rel in human.relationships.values():
                for dim in rel.behavioral_adjustments:
                    rel.behavioral_adjustments[dim] *= 0.995

    def _generate_internal_dialogues(self, actions: dict[str, Any]) -> None:
        today_convs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in self.dataset.conversation_events:
            if event.sim_day == self.sim_day:
                for hid, other_id in [
                    (event.initiator_id, event.receiver_id),
                    (event.receiver_id, event.initiator_id),
                ]:
                    other = self.humans.get(other_id)
                    other_name = other.name if other else other_id
                    quality = "well" if event.total_score > 0 else "poorly" if event.total_score < -2 else "okay"
                    today_convs[hid].append({
                        "other_name": other_name,
                        "reason": event.reason,
                        "quality": quality,
                    })

        for human in self.human_list:
            human.generate_internal_dialogue(
                self.sim_day,
                conversations=today_convs.get(human.human_id),
            )

    def _default_interaction_rules(self) -> list[InteractionRule]:
        rules = []
        ids = [h.human_id for h in self.human_list]
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                rules.append(InteractionRule(
                    human_a_id=id_a, human_b_id=id_b,
                    reason="monday_sync", frequency=7,
                ))
                rules.append(InteractionRule(
                    human_a_id=id_a, human_b_id=id_b,
                    reason="social", frequency=5,
                ))
        return rules
