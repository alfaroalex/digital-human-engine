from __future__ import annotations

import math
import random
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from digital_human.types import (
    JJDIDTIEBUCKLE,
    SPECIAL,
    SPECIAL_TRAIT_NAMES,
    Action,
    ActionType,
    Concern,
    CopingStyle,
    Drives,
    Episode,
    InnerWorld,
    LLMConfig,
    Memory,
    Option,
    Relationship,
    StressState,
    Task,
    WorkContext,
)


@dataclass
class Perception:
    stress_score: float
    stress_sources: dict[str, float]
    task_count: int
    overloaded: bool
    recent_memories: list[Episode]
    feelings_snapshot: dict[str, float]
    personal_concerns: list[Concern]
    new_tasks: list[Task]
    environmental_signals: list[str]
    observable_others: list[dict[str, Any]]
    perception_quality: float
    capacity: int = 5


@dataclass
class Stimulus:
    sim_day: int
    new_tasks: list[Task] = field(default_factory=list)
    environmental_signals: list[str] = field(default_factory=list)
    observable_others: list[dict[str, Any]] = field(default_factory=list)


class DigitalHuman:
    """A self-contained simulation of a human being.

    Person first. Work context is optional.
    Daily cycle: perceive(stimuli) -> understand() -> act() -> Action
    The human does NOT execute its action — it returns it for The World.
    """

    def __init__(
        self,
        human_id: str,
        name: str,
        special: SPECIAL,
        jjdidtiebuckle: JJDIDTIEBUCKLE,
        drives: Drives,
        work_context: WorkContext | None = None,
        inner_world: InnerWorld | None = None,
        llm_config: LLMConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.human_id = human_id
        self.name = name
        # Instance RNG for internal noise/perception sampling.
        # Falls back to the module RNG so standalone humans stay
        # reproducible under random.seed().
        self._rng: Any = rng if rng is not None else random
        self.special = special
        self.jjdidtiebuckle = jjdidtiebuckle
        self.drives = drives
        self.work_context = work_context or WorkContext()
        self.inner_world = inner_world or InnerWorld()
        self.memory = Memory()
        self.relationships: dict[str, Relationship] = {}
        self.stress = StressState(
            coping_style=self._derive_coping_style(),
            threshold=self._derive_stress_threshold(),
            recovery_rate=self._derive_recovery_rate(),
        )

        self.llm_config = llm_config
        self.memory_nodes: list[tuple[int, str]] = []
        self._llm_perception: str | None = None

        self._current_perception: Perception | None = None
        self._current_options: list[Option] = []
        self._sim_day: int = 0
        self._recent_actions: deque[ActionType] = deque(maxlen=14)
        self._last_confronted_day: int | None = None
        self._last_raw_impulse: str | None = None
        self._last_action: Action | None = None

    # --- Core capabilities (spec §2.1) ---

    @property
    def processing_capacity(self) -> float:
        base = (self.special.intelligence + self.special.perception) / 20.0
        stress_penalty = self.stress.score / 200.0
        return max(0.1, base - stress_penalty)

    @property
    def visualization_capacity(self) -> float:
        return (self.special.perception + self.special.intelligence) / 20.0

    @property
    def conversation_readiness(self) -> float:
        special_part = self.special.charisma / 10.0
        jjd_part = (self.jjdidtiebuckle.tact + self.jjdidtiebuckle.enthusiasm) / 20.0
        return (special_part + jjd_part) / 2.0

    # --- Derived attributes ---

    def _derive_coping_style(self) -> CopingStyle:
        s = self.special
        j = self.jjdidtiebuckle
        push_score = s.strength + s.endurance + j.bearing + j.endurance
        seek_score = s.charisma + j.tact + j.unselfishness
        withdraw_score = (10 - s.agility) + (10 - j.initiative) + (10 - j.courage)
        extern_score = j.courage + (10 - j.tact) + s.strength

        scores = {
            CopingStyle.PUSH_THROUGH: push_score,
            CopingStyle.SEEK_HELP: seek_score,
            CopingStyle.WITHDRAW: withdraw_score,
            CopingStyle.EXTERNALIZE: extern_score,
        }
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def _derive_stress_threshold(self) -> float:
        raw = self.special.endurance
        pro = self.jjdidtiebuckle.endurance
        return 40.0 + (raw / 10.0) * 25.0 + (pro / 10.0) * 25.0

    def _derive_recovery_rate(self) -> float:
        return 1.0 + (self.special.agility / 10.0) * 2.0 + (self.jjdidtiebuckle.bearing / 10.0) * 2.0

    # --- Daily Cycle Step 1: PERCEIVE (spec §3.1) ---

    def perceive(self, stimulus: Stimulus) -> Perception:
        self._sim_day = stimulus.sim_day

        for task in stimulus.new_tasks:
            task.assigned_day = stimulus.sim_day
            self.work_context.task_queue.append(task)

        perception_quality = self.special.perception / 10.0

        task_count = len(self.work_context.task_queue)
        overloaded = task_count > self.work_context.capacity_threshold

        visible_signals = stimulus.environmental_signals
        visible_others = stimulus.observable_others

        # Low perception = noisy sampling, not a fixed prefix. A
        # perception-4 human notices a random subset each day.
        if perception_quality < 0.8:
            keep_ratio = perception_quality
            if visible_signals:
                n_keep = max(1, int(len(visible_signals) * keep_ratio))
                visible_signals = self._rng.sample(visible_signals, n_keep)
            if visible_others:
                n_keep = max(1, int(len(visible_others) * keep_ratio))
                visible_others = self._rng.sample(visible_others, n_keep)

        perception = Perception(
            stress_score=self.stress.score,
            stress_sources=dict(self.stress.sources),
            task_count=task_count,
            overloaded=overloaded,
            recent_memories=self.memory.recent(5),
            feelings_snapshot=dict(self.inner_world.feelings),
            personal_concerns=list(self.inner_world.personal_concerns),
            new_tasks=stimulus.new_tasks,
            environmental_signals=visible_signals,
            observable_others=visible_others,
            perception_quality=perception_quality,
            capacity=self.work_context.capacity_threshold,
        )

        if "quality_review" in visible_signals:
            self.stress.sources["social_pressure"] = 15.0
        elif self.stress.sources.get("social_pressure", 0) > 0:
            self.stress.sources["social_pressure"] = max(
                0.0, self.stress.sources["social_pressure"] - 3.0
            )

        self._current_perception = perception
        return perception

    # --- Daily Cycle Step 2: UNDERSTAND (spec §3.2) ---

    def understand(self) -> list[Option]:
        if self._current_perception is None:
            raise RuntimeError("Must call perceive() before understand()")

        p = self._current_perception
        options: list[Option] = []

        # Always-available options
        if p.task_count > 0:
            options.append(Option(
                action_type=ActionType.PROCESS_TASK,
                reasoning="Tasks in queue to process",
            ))

        rest_score = 0.0
        if p.stress_score > 30:
            rest_score += (p.stress_score - 30) / 70.0 * 3.0
        options.append(Option(
            action_type=ActionType.REST,
            score=rest_score,
            reasoning="Recovery option",
        ))

        if p.task_count > 2:
            options.append(Option(
                action_type=ActionType.PLAN,
                reasoning="Multiple tasks warrant planning",
            ))

        # Trait-gated options
        j = self.jjdidtiebuckle

        # CUT_CORNERS: only low-integrity (<=4) and overloaded
        if j.integrity <= 4 and p.task_count > 0 and p.overloaded:
            options.append(Option(
                action_type=ActionType.CUT_CORNERS,
                reasoning="Low integrity + overloaded -> corner-cutting available",
            ))

        # REDUCE_EFFORT: low-dependability (<=4) or high comfort drive
        if j.dependability <= 4 or self.drives.comfort >= 7.0:
            options.append(Option(
                action_type=ActionType.REDUCE_EFFORT,
                reasoning="Low dependability or high comfort drive -> effort reduction available",
            ))

        # TAKE_INITIATIVE: high-initiative (>=6)
        if j.initiative >= 6:
            options.append(Option(
                action_type=ActionType.TAKE_INITIATIVE,
                reasoning="High initiative -> proactive action available",
            ))

        # PUSH_THROUGH: high endurance+bearing (sum>=12) and stressed
        if j.endurance + j.bearing >= 12 and p.stress_score > 40:
            options.append(Option(
                action_type=ActionType.PUSH_THROUGH,
                reasoning="High endurance + bearing -> push through stress",
            ))

        # OFFER_HELP: high unselfishness (>=6) with visible others
        if j.unselfishness >= 6 and p.observable_others:
            options.append(Option(
                action_type=ActionType.OFFER_HELP,
                reasoning="High unselfishness -> willing to help others",
            ))

        # REQUEST_HELP: coping style or social ability
        if (
            self.stress.coping_style == CopingStyle.SEEK_HELP
            and p.stress_score > 40
        ) or (self.special.charisma >= 6 and j.tact >= 5 and p.overloaded):
            options.append(Option(
                action_type=ActionType.REQUEST_HELP,
                reasoning="Coping style or social skills -> help request available",
            ))

        # DELEGATE: tasks exist and others visible, gated by cooperation + authority
        if p.task_count > 0 and p.observable_others:
            my_auth = self.work_context.authority_level
            has_delegatable_target = False
            for other in p.observable_others:
                other_id = other.get("id", "")
                other_auth = other.get("authority_level", 0)
                # Can only delegate to strictly lower authority.
                if other_auth >= my_auth:
                    continue
                rel = self.relationships.get(other_id)
                coop = rel.cooperation if rel else 0.5
                if coop >= 0.3:
                    has_delegatable_target = True
                    break
            if has_delegatable_target:
                options.append(Option(
                    action_type=ActionType.DELEGATE,
                    reasoning="Tasks available and delegatable targets visible -> delegation possible",
                ))

        # CONFRONT: high courage (>=6) with visible others
        if j.courage >= 6 and p.observable_others:
            options.append(Option(
                action_type=ActionType.CONFRONT,
                reasoning="High courage -> confrontation available",
            ))

        # CONFRONT from justice: low respect for a known coworker
        if j.justice >= 6 and p.observable_others:
            for other in p.observable_others:
                other_id = other.get("id", "")
                rel = self.relationships.get(other_id)
                if rel and rel.respect < 0.2:
                    options.append(Option(
                        action_type=ActionType.CONFRONT,
                        reasoning=f"Low respect for {other_id} — justice demands confrontation",
                        target_id=other_id,
                    ))
                    break

        # ESCALATE: stressed with tasks AND has a superior who is visible
        if p.stress_score > 50 and p.task_count > 0 and self.work_context.reports_to:
            superior_visible = any(
                o.get("id") == self.work_context.reports_to
                for o in p.observable_others
            )
            if superior_visible:
                options.append(Option(
                    action_type=ActionType.ESCALATE,
                    reasoning="High stress + tasks + superior available -> escalation available",
                    target_id=self.work_context.reports_to,
                ))

        # SOCIALIZE: charisma>=3 or enthusiasm>=6 with visible others
        if (self.special.charisma >= 6 or j.enthusiasm >= 6) and p.observable_others:
            options.append(Option(
                action_type=ActionType.SOCIALIZE,
                reasoning="Social tendency -> socialization available",
            ))

        # Score all options
        for opt in options:
            opt.score = self._score_option(opt, p)

        # Keyword bridge: recent LLM memory nodes influence scoring
        if self.memory_nodes:
            recent_nodes = [text for _, text in self.memory_nodes[-5:]]
            combined = " ".join(recent_nodes).lower()
            for opt in options:
                if opt.action_type in (ActionType.REDUCE_EFFORT, ActionType.REST) and re.search(
                    r"\bquit\b|\bquitting\b|\bleaving\b", combined
                ):
                    opt.score += 2.0
                if opt.action_type == ActionType.CONFRONT and re.search(
                    r"\bconfront\b|\bfed up\b", combined
                ):
                    opt.score += 2.0
                if opt.action_type in (
                    ActionType.PROCESS_TASK,
                    ActionType.OFFER_HELP,
                ) and re.search(r"\bgrateful\b|\bgood day\b", combined):
                    opt.score += 2.0

        options.sort(key=lambda o: -o.score)
        self._current_options = options
        return options

    def _score_option(self, option: Option, p: Perception) -> float:
        score = option.score
        j = self.jjdidtiebuckle
        d = self.drives
        at = option.action_type
        dv = math.sqrt

        if at == ActionType.PROCESS_TASK:
            score += dv(d.mission) * 0.3 + dv(d.recognition) * 0.2 + dv(d.compensation) * 0.1
            score += j.dependability * 0.2 + j.knowledge * 0.1

        elif at == ActionType.REST:
            score += dv(d.comfort) * 0.4 + dv(d.security) * 0.1
            score += p.stress_score / 25.0
            score -= dv(d.mission) * 0.1

        elif at == ActionType.CUT_CORNERS:
            score += dv(d.comfort) * 0.3 + (10 - j.integrity) * 0.2
            score += p.stress_score / 30.0
            if p.overloaded:
                score += (10 - j.integrity) * 0.15 + min(p.task_count, 10) * 0.15
            score -= dv(d.mission) * 0.2
            if self._last_confronted_day is not None:
                days_since = self._sim_day - self._last_confronted_day
                if 0 <= days_since <= 7:
                    score -= 2.0

        elif at == ActionType.REDUCE_EFFORT:
            score += dv(d.comfort) * 0.5
            score -= dv(d.mission) * 0.2
            score -= dv(d.recognition) * 0.1

        elif at == ActionType.TAKE_INITIATIVE:
            score += dv(d.growth) * 0.3 + dv(d.recognition) * 0.2 + dv(d.mission) * 0.2
            score += j.initiative * 0.2
            if not p.overloaded and p.task_count < p.capacity * 0.5:
                score += j.initiative * 0.1

        elif at == ActionType.PUSH_THROUGH:
            score += dv(d.mission) * 0.3 + dv(d.compensation) * 0.1
            score += j.endurance * 0.2 + j.bearing * 0.1
            score -= dv(d.comfort) * 0.2

        elif at == ActionType.OFFER_HELP:
            score += j.unselfishness * 0.3
            score += dv(d.social_status) * 0.1 + dv(d.recognition) * 0.1

        elif at == ActionType.REQUEST_HELP:
            score += p.stress_score / 30.0
            score += self.special.charisma * 0.1

        elif at == ActionType.DELEGATE:
            score += j.judgment * 0.2 + j.decisiveness * 0.1
            leadership = (j.initiative + j.decisiveness) / 20.0
            score += min(p.task_count, 10) * 0.15 * (0.5 + leadership)
            avg_coop = self._avg_target_cooperation(p)
            if avg_coop < 0.3:
                score -= 3.0
            elif avg_coop < 0.5:
                score -= 1.5

        elif at == ActionType.CONFRONT:
            score += j.courage * 0.3 + j.justice * 0.2
            score -= j.tact * 0.1
            if option.target_id:
                rel = self.relationships.get(option.target_id)
                if rel and rel.respect < 0.2:
                    score += 2.0
                for other in p.observable_others:
                    if other.get("id") == option.target_id:
                        if other.get("apparent_regard", 0.5) < 0.3:
                            score += 1.0
                        other_auth = other.get("authority_level", 0)
                        my_auth = self.work_context.authority_level
                        if other_auth > my_auth:
                            score -= (other_auth - my_auth) * 0.5
                        break

        elif at == ActionType.ESCALATE:
            score += p.stress_score / 20.0
            score += j.judgment * 0.1

        elif at == ActionType.SOCIALIZE:
            score += self.special.charisma * 0.15 + j.enthusiasm * 0.2 + j.tact * 0.15
            score += dv(d.social_status) * 0.15 + dv(d.recognition) * 0.1
            score -= p.stress_score / 50.0
            if not p.overloaded and p.task_count <= p.capacity * 0.5:
                score += self.special.charisma * 0.05

        elif at == ActionType.PLAN:
            score += self.special.intelligence * 0.15 + j.judgment * 0.2
            score += min(p.task_count, 10) * 0.1

        elif at == ActionType.ADJUST_EFFORT:
            score += j.initiative * 0.1

        # Inner world modifiers
        satisfaction = self.inner_world.feelings.get("job_satisfaction", 0.0)
        if at in (ActionType.PROCESS_TASK, ActionType.TAKE_INITIATIVE):
            score += satisfaction * 0.5
        if at in (ActionType.CUT_CORNERS, ActionType.REDUCE_EFFORT):
            score -= satisfaction * 0.3

        # Memory modifiers
        recent = self.memory.recent(3)
        negative_recent = sum(1 for e in recent if e.emotional_valence < -0.3)
        if at in (ActionType.REST, ActionType.REQUEST_HELP):
            score += negative_recent * 0.5
        if at in (ActionType.PROCESS_TASK, ActionType.TAKE_INITIATIVE):
            score -= negative_recent * 0.3

        # Intelligence noise
        noise_amplitude = (10 - self.special.intelligence) / 10.0 * 1.5
        if noise_amplitude > 0:
            score += self._rng.gauss(0, noise_amplitude)

        # Diminishing returns — 7-day window. Sign-aware: the penalty
        # always pushes downward, so repeating a negative-scored
        # action never makes it more attractive.
        recent_7 = list(self._recent_actions)[-7:]
        repeat_count = sum(1 for a in recent_7 if a == at)
        if repeat_count > 0:
            score -= abs(score) * repeat_count * 0.1

        # Minimum variety — 14-day window
        if len(self._recent_actions) >= 7:
            distinct = len(set(self._recent_actions))
            if distinct < 3:
                count_this = sum(1 for a in self._recent_actions if a == at)
                if count_this == 0:
                    score += 1.0

        return score

    def _avg_target_cooperation(self, p: Perception) -> float:
        if not p.observable_others:
            return 0.5
        cooperations = []
        for other in p.observable_others:
            other_id = other.get("id", "")
            rel = self.relationships.get(other_id)
            if rel:
                cooperations.append(rel.cooperation)
            else:
                cooperations.append(0.5)
        return sum(cooperations) / len(cooperations) if cooperations else 0.5

    # --- Trait score lookup ---

    def _get_trait_score(self, trait_name: str) -> int:
        if trait_name in SPECIAL_TRAIT_NAMES:
            return getattr(self.special, trait_name)
        return getattr(self.jjdidtiebuckle, trait_name, 5)

    # --- LLM-driven Perceive ---

    def _llm_perceive(self, stimulus: Stimulus) -> None:
        if not self.llm_config or not self.llm_config.enabled:
            return

        import logging

        from digital_human.llm import generate_perception

        s = self.special
        special_dict = {
            "strength": s.strength, "perception": s.perception,
            "endurance": s.endurance, "charisma": s.charisma,
            "intelligence": s.intelligence, "agility": s.agility,
            "luck": s.luck,
        }

        task_data = []
        for t in stimulus.new_tasks[:5]:
            task_data.append({
                "description": t.description,
                "requirements": dict(t.requirements),
            })

        try:
            result = generate_perception(
                self.llm_config,
                name=self.name,
                role=self.work_context.role,
                special=special_dict,
                stress_score=self.stress.score,
                feelings=dict(self.inner_world.feelings),
                new_tasks=task_data,
                signals=stimulus.environmental_signals,
                others=stimulus.observable_others,
                recent_nodes=self.memory_nodes[-3:],
            )
            self._llm_perception = result
        except Exception as e:
            logging.getLogger(__name__).warning(
                "LLM perceive failed for %s: %s", self.name, e,
            )
            self._llm_perception = None

    # --- LLM-driven Understand ---

    def _llm_understand(self) -> dict[str, Any] | None:
        if not self.llm_config or not self.llm_config.enabled:
            return None
        if not self._llm_perception:
            return None

        import logging

        from digital_human.llm import generate_decision

        state = self._serialize_for_llm()

        j = self.jjdidtiebuckle
        jjd_full = ", ".join(
            f"{attr}={getattr(j, attr)}"
            for attr in (
                "justice", "judgment", "dependability", "initiative",
                "decisiveness", "tact", "integrity", "enthusiasm",
                "bearing", "unselfishness", "courage", "knowledge",
                "loyalty", "endurance",
            )
        )
        state["jjd_full"] = jjd_full
        state["coping_style"] = self.stress.coping_style.value
        state["authority_level"] = self.work_context.authority_level
        state["reports_to"] = self.work_context.reports_to

        if self.work_context.task_queue:
            next_task = self.work_context.task_queue[0]
            if next_task.requirements:
                state["task_requirements"] = ", ".join(
                    f"{k}={v}" for k, v in next_task.requirements.items()
                )

        state["recent_nodes"] = self.memory_nodes[-5:]

        try:
            return generate_decision(
                self.llm_config, state, self._llm_perception,
            )
        except Exception as e:
            logging.getLogger(__name__).warning(
                "LLM understand failed for %s: %s", self.name, e,
            )
            return None

    # --- Act-layer tagging ---

    def _compute_act_tags(self, action: Action) -> None:
        j = self.jjdidtiebuckle
        s = self.special
        at = action.action_type

        action.metadata["tact"] = j.tact / 10.0
        action.metadata["composure"] = j.bearing / 10.0
        action.metadata["competence"] = (j.knowledge + s.intelligence) / 20.0
        action.metadata["reliability"] = j.dependability / 10.0

        if action.target_id:
            if j.tact < 4 and at == ActionType.CONFRONT:
                action.metadata["reception_prediction"] = "likely_hostile_reception"
            elif s.charisma > 7 and at == ActionType.DELEGATE:
                action.metadata["reception_prediction"] = "likely_accepted"
            elif at == ActionType.OFFER_HELP:
                action.metadata["reception_prediction"] = "likely_welcomed"
            elif at == ActionType.ESCALATE:
                action.metadata["reception_prediction"] = "depends_on_relationship"

        task_actions = {
            ActionType.PROCESS_TASK, ActionType.CUT_CORNERS,
            ActionType.PUSH_THROUGH,
        }
        if at in task_actions and self.work_context.task_queue:
            next_task = self.work_context.task_queue[0]
            if next_task.requirements:
                matches = []
                neg_gaps = 0
                for trait, req in next_task.requirements.items():
                    score = self._get_trait_score(trait)
                    matches.append(min(1.0, score / max(1, req)))
                    if score - req <= -2:
                        neg_gaps += 1
                action.metadata["trait_match"] = sum(matches) / len(matches)
                action.metadata["error_probability"] = neg_gaps / len(matches)
            else:
                action.metadata["trait_match"] = 1.0
                action.metadata["error_probability"] = 0.0
            action.metadata["predicted_quality"] = max(
                0.0,
                action.metadata["competence"] - self.stress.score / 200.0,
            )

        impacts: dict[str, str] = {}
        if at == ActionType.CONFRONT:
            impacts["trust_with_target"] = "negative"
            impacts["respect_from_observers"] = "mixed"
        elif at == ActionType.OFFER_HELP:
            impacts["warmth_with_target"] = "positive"
        elif at == ActionType.CUT_CORNERS:
            impacts["respect_from_observers"] = "negative"
        elif at == ActionType.DELEGATE:
            impacts["cooperation_with_target"] = "negative_if_overloaded"
        elif at in (ActionType.PROCESS_TASK, ActionType.PUSH_THROUGH):
            impacts["respect_from_observers"] = "positive"
        elif at == ActionType.TAKE_INITIATIVE:
            impacts["trust_from_observers"] = "positive"
        if impacts:
            action.metadata["relationship_impacts"] = impacts

        raw = action.metadata.get("raw_impulse", at.value)
        if raw in self.EXTREME_IMPULSES and raw != at.value:
            action.metadata["suppression_cost"] = 0.7
        elif raw != at.value:
            action.metadata["suppression_cost"] = 0.3
        else:
            action.metadata["suppression_cost"] = 0.0

        energizing = {ActionType.SOCIALIZE, ActionType.REST, ActionType.OFFER_HELP}
        draining = {ActionType.PUSH_THROUGH, ActionType.CONFRONT, ActionType.PROCESS_TASK}
        if at in energizing:
            action.metadata["effort_cost"] = "energizing"
        elif at in draining:
            action.metadata["effort_cost"] = "draining"
        else:
            action.metadata["effort_cost"] = "neutral"

        relief: list[str] = []
        if at == ActionType.REST and self.stress.score > 30:
            relief.append("stress_reduction")
        if at == ActionType.CONFRONT and "injustice" in self.stress.sources:
            relief.append("injustice")
        if at == ActionType.DELEGATE and len(self.work_context.task_queue) > self.work_context.capacity_threshold:
            relief.append("workload")
        if at == ActionType.ESCALATE and self.stress.score > 50:
            relief.append("stress_offloading")
        action.metadata["relief_potential"] = relief

    # --- Daily Cycle Step 3: ACT (spec §3.3) ---
    # The human decides. The World executes. No task dequeuing here.
    # Flow: generate raw impulse → filter through traits → log the gap.

    EXTREME_IMPULSES: frozenset[str] = frozenset({
        "aggressive_confrontation", "quit", "disappear", "scream",
    })

    def act(self, llm_decision: dict[str, Any] | None = None) -> Action:
        if llm_decision is not None:
            action_str = llm_decision.get("action", "process_task")
            try:
                action_type = ActionType(action_str)
            except ValueError:
                action_type = ActionType.PROCESS_TASK

            raw_impulse = llm_decision.get("want", action_type.value)
            target_id = llm_decision.get("target")
            reasoning = llm_decision.get("reason", "")
            conflict = llm_decision.get("conflict")

            metadata: dict[str, Any] = {"llm_driven": True}
            if raw_impulse != action_type.value:
                metadata["raw_impulse"] = raw_impulse
            if conflict:
                metadata["conflict"] = conflict

            action = Action(
                action_type=action_type,
                reasoning=reasoning,
                target_id=target_id,
                sim_day=self._sim_day,
                metadata=metadata,
            )
        else:
            if not self._current_options:
                raise RuntimeError("Must call understand() before act()")

            chosen = self._current_options[0]
            raw_impulse = self._generate_raw_impulse(chosen)
            filtered_type, metadata = self._apply_filter(raw_impulse, chosen)

            action = Action(
                action_type=filtered_type,
                reasoning=chosen.reasoning,
                target_id=chosen.target_id,
                task_id=chosen.task_id,
                sim_day=self._sim_day,
                metadata=metadata,
            )

        self._log_impulse_gap(raw_impulse, action)
        self._compute_act_tags(action)

        self._recent_actions.append(action.action_type)

        self._update_stress(action)
        self._update_memory(action, raw_impulse=raw_impulse)
        self._update_inner_world(action)

        self._last_raw_impulse = raw_impulse
        self._last_action = action

        self._current_perception = None
        self._current_options = []
        self._llm_perception = None

        return action

    def _generate_raw_impulse(self, chosen: Option) -> str:
        stress_ratio = self.stress.score / max(1.0, self.stress.threshold)

        if stress_ratio < 0.5:
            return chosen.action_type.value

        j = self.jjdidtiebuckle

        if chosen.action_type == ActionType.CONFRONT and (
            stress_ratio > 0.7 or j.tact <= 4
        ):
            return "aggressive_confrontation"

        if stress_ratio > 0.8 and self.drives.comfort >= 7:
            return "quit"

        if stress_ratio > 0.7 and self.stress.coping_style == CopingStyle.WITHDRAW:
            return "disappear"

        if stress_ratio > 0.7 and self.stress.coping_style == CopingStyle.EXTERNALIZE:
            return "scream"

        return chosen.action_type.value

    def _apply_filter(
        self, raw_impulse: str, chosen: Option,
    ) -> tuple[ActionType, dict[str, Any]]:
        metadata: dict[str, Any] = {}

        if raw_impulse not in self.EXTREME_IMPULSES:
            return chosen.action_type, metadata

        j = self.jjdidtiebuckle
        filter_sum = j.bearing + j.tact + j.judgment

        stress_ratio = self.stress.score / max(1.0, self.stress.threshold)
        endurance_penalty = max(0.0, stress_ratio - 0.5) * (10 - self.special.endurance)
        effective_filter = filter_sum - endurance_penalty

        professional_map: dict[str, ActionType] = {
            "aggressive_confrontation": ActionType.CONFRONT,
            "quit": ActionType.REDUCE_EFFORT,
            "disappear": ActionType.REST,
            "scream": ActionType.CONFRONT,
        }
        professional_action = professional_map.get(raw_impulse, chosen.action_type)

        metadata["raw_impulse"] = raw_impulse

        if effective_filter > 15:
            metadata["filter_result"] = "suppressed"
            return professional_action, metadata
        elif effective_filter >= 10:
            metadata["filter_result"] = "partial"
            metadata["tone"] = "harsh"
            return professional_action, metadata
        else:
            metadata["filter_result"] = "failed"
            metadata["tone"] = "extreme"
            return professional_action, metadata

    def _log_impulse_gap(self, raw_impulse: str, action: Action) -> None:
        if raw_impulse not in self.EXTREME_IMPULSES:
            if self.stress.sources.get("suppression", 0) > 0:
                self.stress.sources["suppression"] = max(
                    0.0, self.stress.sources["suppression"] - 1.0,
                )
            return

        filter_result = action.metadata.get("filter_result", "")
        if filter_result == "suppressed":
            self.stress.sources["suppression"] = (
                self.stress.sources.get("suppression", 0.0) + 3.0
            )
        elif filter_result == "partial":
            self.stress.sources["suppression"] = (
                self.stress.sources.get("suppression", 0.0) + 1.5
            )

    # --- Full daily cycle convenience ---

    def daily_cycle(self, stimulus: Stimulus) -> Action:
        if self.llm_config and self.llm_config.enabled:
            self._llm_perceive(stimulus)

        self.perceive(stimulus)

        llm_decision = None
        if self.llm_config and self.llm_config.enabled and self._llm_perception:
            llm_decision = self._llm_understand()

        if llm_decision is None:
            self.understand()

        return self.act(llm_decision=llm_decision)

    # --- Internal state updates (called during act) ---

    def _update_stress(self, action: Action) -> None:
        task_count = len(self.work_context.task_queue)
        capacity = self.work_context.capacity_threshold

        if task_count > capacity:
            overload_ratio = (task_count - capacity) / max(1, capacity)
            self.stress.sources["workload"] = min(40.0, overload_ratio * 20.0)
        elif task_count > 0:
            self.stress.sources["workload"] = (task_count / capacity) * 5.0
        else:
            self.stress.sources["workload"] = 0.0

        concern_load = sum(c.weight for c in self.inner_world.personal_concerns if c.affects_work)
        self.stress.sources["personal"] = concern_load * 10.0

        total_source_stress = sum(
            v for k, v in self.stress.sources.items() if not k.startswith("_")
        )

        prev_score = self.stress.score

        if action.action_type == ActionType.REST:
            self.stress.score -= self.stress.recovery_rate * 3.0

        if total_source_stress < self.stress.score:
            self.stress.score -= self.stress.recovery_rate
        elif total_source_stress > self.stress.score:
            delta = (total_source_stress - self.stress.score) * 0.3
            self.stress.score += delta

        if self.stress.score > prev_score + 0.5:
            self.stress.trajectory = "rising"
        elif self.stress.score < prev_score - 0.5:
            self.stress.trajectory = "falling"
        else:
            self.stress.trajectory = "stable"

        self.stress.clamp()

    def _update_memory(
        self, action: Action, raw_impulse: str | None = None,
    ) -> None:
        valence_map: dict[ActionType, float] = {
            ActionType.PROCESS_TASK: 0.1,
            ActionType.REST: 0.2,
            ActionType.PLAN: 0.1,
            ActionType.TAKE_INITIATIVE: 0.3,
            ActionType.PUSH_THROUGH: -0.1,
            ActionType.CUT_CORNERS: -0.3,
            ActionType.REDUCE_EFFORT: 0.0,
            ActionType.OFFER_HELP: 0.3,
            ActionType.REQUEST_HELP: -0.1,
            ActionType.DELEGATE: 0.1,
            ActionType.CONFRONT: -0.2,
            ActionType.ESCALATE: -0.2,
            ActionType.SOCIALIZE: 0.3,
            ActionType.ADJUST_EFFORT: 0.0,
        }
        valence = valence_map.get(action.action_type, 0.0)

        if self.stress.is_critical():
            valence -= 0.2

        episode = Episode(
            sim_day=action.sim_day,
            event_type="decision",
            content=f"Chose to {action.action_type.value}: {action.reasoning}",
            emotional_valence=max(-1.0, min(1.0, valence)),
            people_involved=[action.target_id] if action.target_id else [],
            tags=[action.action_type.value],
            raw_impulse=raw_impulse,
        )
        self.memory.record(episode)

    def _update_inner_world(self, action: Action) -> None:
        at = action.action_type

        if at in (ActionType.PROCESS_TASK, ActionType.TAKE_INITIATIVE):
            delta = 0.02 * (self.drives.mission / 10.0)
            self.inner_world.update_feeling("job_satisfaction", delta)
        elif at in (ActionType.CUT_CORNERS, ActionType.REDUCE_EFFORT):
            self.inner_world.update_feeling("job_satisfaction", -0.02)
            if self.drives.comfort >= 7:
                self.inner_world.update_feeling("motivation", 0.01)
        elif at == ActionType.REST:
            self.inner_world.update_feeling("motivation", 0.03)

        if self.stress.is_critical():
            self.inner_world.update_feeling("motivation", -0.03)
            self.inner_world.update_feeling("job_satisfaction", -0.02)

    # --- LLM integration ---

    def _serialize_for_llm(self) -> dict[str, Any]:
        j = self.jjdidtiebuckle
        jjd_scores = {
            attr: getattr(j, attr)
            for attr in (
                "justice", "judgment", "dependability", "initiative",
                "decisiveness", "tact", "integrity", "enthusiasm",
                "bearing", "unselfishness", "courage", "knowledge",
                "loyalty", "endurance",
            )
        }
        top_3_jjd = sorted(jjd_scores.items(), key=lambda x: -x[1])[:3]

        s = self.special
        special_summary = (
            f"S{s.strength} P{s.perception} E{s.endurance} "
            f"C{s.charisma} I{s.intelligence} A{s.agility} L{s.luck}"
        )
        traits_summary = f"SPECIAL=[{special_summary}], top JJD=[{', '.join(f'{k}:{v}' for k, v in top_3_jjd)}]"

        top_drives = ", ".join(f"{n}:{v:.1f}" for n, v in self.drives.top_drives(3))

        top_source = ""
        if self.stress.sources:
            top_source = max(self.stress.sources, key=self.stress.sources.get)  # type: ignore[arg-type]

        rel_summaries = []
        for rid, rel in self.relationships.items():
            rel_summaries.append(
                f"{rid}: trust={rel.trust:.2f} respect={rel.respect:.2f} warmth={rel.warmth:.2f}"
            )

        return {
            "name": self.name,
            "role": self.work_context.role,
            "traits_summary": traits_summary,
            "top_drives": top_drives,
            "stress_score": self.stress.score,
            "stress_trajectory": self.stress.trajectory,
            "top_stress_source": top_source,
            "feelings": dict(self.inner_world.feelings),
            "raw_impulse": self._last_raw_impulse or "",
            "filtered_action": self._last_action.action_type.value if self._last_action else "",
            "task_queue_size": len(self.work_context.task_queue),
            "task_descriptions": [
                t.description for t in list(self.work_context.task_queue)[:3]
            ],
            "relationships": "; ".join(rel_summaries) if rel_summaries else "",
            "recent_nodes": self.memory_nodes[-3:],
        }

    def _serialize_conversations(self, conversations: list[dict[str, Any]]) -> str:
        parts = []
        for c in conversations:
            parts.append(
                f"Talked to {c.get('other_name', 'someone')} about {c.get('reason', '?')}. "
                f"It went {c.get('quality', 'okay')}."
            )
        return " ".join(parts)

    def generate_internal_dialogue(
        self,
        sim_day: int,
        conversations: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self.llm_config or not self.llm_config.enabled:
            return

        import logging

        from digital_human.llm import generate_memory_node

        state = self._serialize_for_llm()
        if self._last_action and self._last_action.metadata.get("llm_driven"):
            m = self._last_action.metadata
            state["llm_want"] = m.get("raw_impulse", "")
            state["llm_decide"] = self._last_action.action_type.value
            state["llm_conflict"] = m.get("conflict", "")
        conv_summary = self._serialize_conversations(conversations or [])

        try:
            node_text = generate_memory_node(self.llm_config, state, conv_summary)
        except Exception as e:
            logging.getLogger(__name__).warning(
                "LLM call failed for %s on day %d: %s", self.name, sim_day, e,
            )
            return
        if node_text is None:
            return

        self.memory_nodes.append((sim_day, node_text))

        self.memory.record(Episode(
            sim_day=sim_day,
            event_type="internal_dialogue",
            content=node_text,
            emotional_valence=0.0,
            tags=["internal_dialogue"],
        ))

    # --- Config loading ---

    @classmethod
    def from_config(cls, config: dict[str, Any], llm_config: LLMConfig | None = None) -> DigitalHuman:
        special = SPECIAL(**config["special"])
        jjd = JJDIDTIEBUCKLE(**config["jjdidtiebuckle"])
        drives = Drives(**config["drives"])

        iw = InnerWorld()
        if "inner_world" in config:
            iw_cfg = config["inner_world"]
            if "feelings" in iw_cfg:
                iw.feelings = dict(iw_cfg["feelings"])
            if "beliefs" in iw_cfg:
                for topic, b in iw_cfg["beliefs"].items():
                    iw.set_belief(topic, b["value"], b["confidence"], 0)
            if "concerns" in iw_cfg:
                for c in iw_cfg["concerns"]:
                    iw.personal_concerns.append(
                        Concern(c["topic"], c["weight"], c["affects_work"])
                    )

        wc = None
        if "work_context" in config:
            wc_cfg = config["work_context"]
            wc = WorkContext(
                role=wc_cfg.get("role", ""),
                responsibilities=wc_cfg.get("responsibilities", []),
                hours_per_week=wc_cfg.get("hours_per_week", 40.0),
                compensation=wc_cfg.get("compensation", {}),
                capacity_threshold=wc_cfg.get("capacity_threshold", 5),
                skills=wc_cfg.get("skills", []),
                authority_level=wc_cfg.get("authority_level", 0),
                reports_to=wc_cfg.get("reports_to", None),
            )

        return cls(
            human_id=config["id"],
            name=config["name"],
            special=special,
            jjdidtiebuckle=jjd,
            drives=drives,
            work_context=wc,
            inner_world=iw,
            llm_config=llm_config,
        )
