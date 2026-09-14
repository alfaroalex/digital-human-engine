from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any

from digital_human.types import ActionType, LLMConfig

logger = logging.getLogger(__name__)


def summarize_recent_nodes(nodes: list[tuple[int, str]]) -> str:
    if not nodes:
        return ""
    themes = []
    for _, text in nodes[-3:]:
        first_sentence = text.split(".")[0].strip()
        if first_sentence:
            themes.append(first_sentence)
    if not themes:
        return ""
    return "Lately you've been thinking: " + "; ".join(themes) + "."


def _cap_node_length(text: str, max_sentences: int = 3) -> str:
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    capped = ". ".join(sentences[:max_sentences])
    return capped + "." if capped else text


def _call_ollama(
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int | None = None,
) -> str | None:
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": max_tokens or config.max_tokens,
        },
    }
    url = f"{config.base_url.rstrip('/')}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        content = result.get("message", {}).get("content", "").strip()
        return content if content else None
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        return None


# --- Internal dialogue (existing) ---

SYSTEM_PROMPT = (
    "You are the raw inner voice of a real person. Not a narrator. "
    "Not a coach. The actual unfiltered thought stream inside their "
    "head at the end of a workday.\n\n"
    "Rules:\n"
    "- 2-4 sentences max\n"
    "- First person only, no quotation marks\n"
    "- Be messy and honest — real thoughts include frustration, doubt, "
    "pettiness, relief, boredom, resentment, pride, anxiety, hope\n"
    "- Match the personality: a blunt person thinks bluntly, an anxious "
    "person spirals, a lazy person rationalizes, an ambitious person schemes\n"
    "- Reference specific people and events from today, not generalities\n"
    "- Never sound like a motivational poster or self-help book\n"
    "- If they're stressed, their thoughts should feel scattered\n"
    "- If they're coasting, their thoughts should feel detached\n"
    "- If they're angry, let the anger show\n"
    "- CRITICAL: Never start with the person's name. Never start with "
    "'Ugh' or 'Alright' or any habitual opener. Start each thought "
    "mid-stream, as if catching a person in the middle of thinking. "
    "Examples of good openers: 'That meeting was...', 'Can't believe...', "
    "'Something about today...', 'Not sure why I...', 'Honestly...', "
    "'Ten tasks and counting...'\n"
    "- Do NOT say 'let's do this' or 'time to roll up my sleeves' or "
    "any similar cheerful nonsense unless the person genuinely feels that way\n"
    "- Each day's thought must be COMPLETELY DIFFERENT from previous days. "
    "Never repeat or rephrase what you thought before. New day, new thought. "
    "If you're still frustrated about the same thing, express it differently "
    "— find a new angle, a new detail, a new emotion."
)


def build_user_prompt(state: dict[str, Any], conversations_summary: str) -> str:
    lines = [
        f"I'm {state['name']}, {state['role']}.",
    ]

    feelings = state.get("feelings", {})
    stress = state["stress_score"]
    if isinstance(feelings, dict):
        sat = feelings.get("job_satisfaction", 0)
        mot = feelings.get("motivation", 0)
        if stress > 70:
            emotion_line = "I'm near my breaking point."
        elif stress > 50:
            emotion_line = "I'm feeling the pressure today."
        elif stress > 30:
            emotion_line = "Mildly stressed but managing."
        elif sat < -0.3:
            emotion_line = "I'm really not feeling this job lately."
        elif mot < -0.2:
            emotion_line = "Hard to care today."
        elif sat > 0.3:
            emotion_line = "Feeling pretty good about things."
        else:
            emotion_line = "Just another day."
    else:
        emotion_line = f"Emotional state: {feelings}"
    lines.append(emotion_line)

    lines.append(f"Key traits: {state['traits_summary']}")
    lines.append(f"Top drives: {state['top_drives']}")
    lines.append(f"Stress: {stress:.0f}/100 ({state['stress_trajectory']})")
    if state.get("top_stress_source"):
        lines.append(f"Main stress source: {state['top_stress_source']}")
    lines.append(f"Raw impulse: {state['raw_impulse']}")
    lines.append(f"Actual action: {state['filtered_action']}")

    if state.get("raw_impulse") != state.get("filtered_action"):
        lines.append(
            f"What I held back: I wanted to {state['raw_impulse']} "
            f"but instead I {state['filtered_action']}."
        )

    task_descs = state.get("task_descriptions") or []
    queue_size = state["task_queue_size"]
    if task_descs:
        sample = "', '".join(task_descs[:2])
        if queue_size > len(task_descs[:2]):
            lines.append(
                f"You have {queue_size} tasks including '{sample}'."
            )
        else:
            lines.append(f"You have {queue_size} tasks: '{sample}'.")
    else:
        lines.append(f"Tasks queued: {queue_size}")

    if state.get("relationships"):
        lines.append("People on my mind:")
        for rel_str in state["relationships"].split("; "):
            lines.append(f"  {rel_str}")

    if conversations_summary:
        lines.append(f"Today's conversations: {conversations_summary}")

    want = state.get("llm_want")
    decide = state.get("llm_decide")
    conflict = state.get("llm_conflict")
    if want and decide:
        lines.append(f"What I wanted to do: {want}")
        lines.append(f"What I actually did: {decide}")
        if conflict:
            lines.append(f"What I gave up: {conflict}")

    return "\n".join(lines)


def generate_memory_node(
    config: LLMConfig,
    state: dict[str, Any],
    conversations_summary: str = "",
) -> str | None:
    user_prompt = build_user_prompt(state, conversations_summary)
    raw = _call_ollama(config, SYSTEM_PROMPT, user_prompt)
    if raw:
        return _cap_node_length(raw)
    return None


# --- LLM-driven Perceive ---

PERCEIVE_SYSTEM_PROMPT = (
    "You are experiencing a moment in someone's workday. Given their "
    "personality and awareness level, describe what they noticed and "
    "how they interpreted it. First person, 2-3 sentences. Include what "
    "they missed if their perception is low. Include misinterpretations "
    "if their intelligence is low. Include emotional coloring based on "
    "their current stress and feelings."
)


def generate_perception(
    config: LLMConfig,
    name: str,
    role: str,
    special: dict[str, int],
    stress_score: float,
    feelings: dict[str, float],
    new_tasks: list[dict[str, Any]],
    signals: list[str],
    others: list[dict[str, Any]],
    recent_nodes: list[tuple[int, str]],
) -> str | None:
    lines = [
        f"I'm {name}, a {role}.",
        f"My awareness: perception={special.get('perception', 5)}/10, "
        f"intelligence={special.get('intelligence', 5)}/10",
        f"Stress level: {stress_score:.0f}/100",
    ]

    if feelings:
        parts = [f"{k}={v:.2f}" for k, v in feelings.items()]
        lines.append(f"How I feel: {', '.join(parts)}")

    if new_tasks:
        descs = []
        for t in new_tasks[:5]:
            desc = t.get("description", "task")
            reqs = t.get("requirements", {})
            if reqs:
                req_str = ", ".join(f"{k}={v}" for k, v in reqs.items())
                descs.append(f"{desc} (needs: {req_str})")
            else:
                descs.append(desc)
        lines.append(f"New tasks arriving: {'; '.join(descs)}")

    if signals:
        lines.append(f"What's happening around me: {', '.join(signals)}")

    if others:
        parts = []
        for o in others[:4]:
            regard = o.get("apparent_regard", 0.5)
            parts.append(
                f"{o.get('id', '?')} (stress={o.get('stress', 0):.0f}, "
                f"workload={o.get('workload', 0)}, regard={regard:.2f})"
            )
        lines.append(f"People I can see: {', '.join(parts)}")

    node_summary = summarize_recent_nodes(recent_nodes)
    if node_summary:
        lines.append(node_summary)

    return _call_ollama(config, PERCEIVE_SYSTEM_PROMPT, "\n".join(lines))


# --- LLM-driven Understand ---

UNDERSTAND_SYSTEM_PROMPT = (
    "You are the mind of a person deciding what to do right now at work. "
    "Given who they are, what they just perceived, and their history, "
    "think through their options and make a decision.\n\n"
    "You must respond in EXACTLY this format, nothing else:\n\n"
    "WANT: [what you actually want to do, raw and honest — the impulse]\n"
    "REASON: [why you want that — the emotional/motivational driver]\n"
    "DECIDE: [what you will actually do — the reasoned choice]\n"
    "ACTION: [one of: process_task, rest, plan, cut_corners, reduce_effort, "
    "take_initiative, push_through, delegate, escalate, request_help, "
    "confront, offer_help, socialize]\n"
    "TARGET: [human_id of who this involves, or none]\n"
    "CONFLICT: [what you're sacrificing by choosing this — the internal "
    "tension, or none]\n\n"
    "Do not repeat reasoning from previous days. Find fresh angles."
)


def parse_decision(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    result: dict[str, Any] = {}
    for field in ("WANT", "REASON", "DECIDE", "ACTION", "TARGET", "CONFLICT"):
        match = re.search(rf"^{field}:\s*(.+?)$", text, re.MULTILINE)
        if match:
            result[field.lower()] = match.group(1).strip().strip("[]")

    if "action" not in result:
        return None

    action_str = result["action"].lower().strip()
    valid = {at.value for at in ActionType}
    if action_str not in valid:
        return None
    result["action"] = action_str

    target = result.get("target", "none").lower().strip()
    result["target"] = None if target == "none" else target

    conflict = result.get("conflict", "none").strip()
    result["conflict"] = None if conflict.lower() == "none" else conflict

    return result


def generate_decision(
    config: LLMConfig,
    state: dict[str, Any],
    perception: str,
) -> dict[str, Any] | None:
    lines = [
        f"I'm {state['name']}, {state['role']}.",
        "",
        "What I just noticed:",
        f"  {perception}",
        "",
        "Who I am:",
        f"  Traits: {state['traits_summary']}",
        f"  Full professional scores: {state.get('jjd_full', '')}",
        f"  Drives: {state['top_drives']}",
        f"  Stress: {state['stress_score']:.0f}/100 ({state['stress_trajectory']})",
    ]

    if state.get("top_stress_source"):
        lines.append(f"  Main stress: {state['top_stress_source']}")
    lines.append(f"  Coping style: {state.get('coping_style', 'unknown')}")

    if state.get("feelings"):
        parts = [f"{k}={v:.2f}" for k, v in state["feelings"].items()]
        lines.append(f"  Feelings: {', '.join(parts)}")

    lines.append("")
    lines.append("My work situation:")
    lines.append(f"  Tasks queued: {state['task_queue_size']}")
    lines.append(f"  Authority level: {state.get('authority_level', 0)}")
    if state.get("reports_to"):
        lines.append(f"  I report to: {state['reports_to']}")
    if state.get("task_requirements"):
        lines.append(f"  Next task needs: {state['task_requirements']}")

    if state.get("relationships"):
        lines.append("")
        lines.append("People I work with:")
        for rel_str in state["relationships"].split("; "):
            lines.append(f"  {rel_str}")

    node_summary = summarize_recent_nodes(state.get("recent_nodes") or [])
    if node_summary:
        lines.append("")
        lines.append(node_summary)

    raw = _call_ollama(
        config, UNDERSTAND_SYSTEM_PROMPT, "\n".join(lines), max_tokens=250,
    )
    return parse_decision(raw) if raw else None


# --- LLM task generation ---

TASK_GEN_SYSTEM_PROMPT = (
    "Generate realistic work tasks for a specific role. Each task needs "
    "a short description and trait requirements (which traits are needed "
    "and at what level 1-10).\n\n"
    "Valid trait names: strength, perception, endurance, charisma, "
    "intelligence, agility, luck, justice, judgment, dependability, "
    "initiative, decisiveness, tact, integrity, enthusiasm, bearing, "
    "unselfishness, courage, knowledge, loyalty.\n\n"
    "Respond in EXACTLY this JSON format, no other text:\n"
    '[{"description": "...", "requirements": {"trait": score, ...}, '
    '"complexity": 0.3, "importance": 5.0}]'
)


def generate_tasks(
    config: LLMConfig,
    role: str,
    responsibilities: list[str],
    n_tasks: int = 20,
) -> list[dict[str, Any]] | None:
    user_prompt = (
        f"Role: {role}\n"
        f"Responsibilities: {', '.join(responsibilities)}\n"
        f"Generate {n_tasks} varied tasks."
    )

    raw = _call_ollama(config, TASK_GEN_SYSTEM_PROMPT, user_prompt, max_tokens=2000)
    if not raw:
        return None

    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start < 0 or end <= start:
            return None
        tasks = json.loads(raw[start:end])
        if not isinstance(tasks, list):
            return None

        valid: list[dict[str, Any]] = []
        for t in tasks:
            if not isinstance(t, dict) or "description" not in t:
                continue
            reqs = {}
            for k, v in t.get("requirements", {}).items():
                if isinstance(v, (int, float)) and 1 <= v <= 10:
                    reqs[k] = int(v)
            if not reqs:
                continue
            valid.append({
                "description": str(t["description"]),
                "requirements": reqs,
                "complexity": max(0.1, min(1.0, float(t.get("complexity", 0.7)))),
                "importance": max(1.0, min(10.0, float(t.get("importance", 5.0)))),
            })
        return valid if valid else None
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        return None
