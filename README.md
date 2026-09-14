# Digital Human

A self-contained behavioral simulation engine that models human beings.

A Digital Human is a simulation entity with its own perception, cognition,
memory, emotions, relationships, and decision-making. External systems supply
stimuli — events, tasks, environmental changes. The human processes them
internally and produces behavior. The World runs the clock, delivers events,
and records what happens.

The model is person-first: the base human is defined by SPECIAL attributes,
JJDIDTIEBUCKLE scores, drives, an inner world, memory, relationships, and
stress. It is designed to simulate any person type, not one narrow role. Work
context is an optional layer on top.

## Requirements

- Python 3.12
- Dependencies: `pytest`, `pyyaml` (the engine itself is stdlib-only)
- Optional: a local [Ollama](https://ollama.com) instance for the LLM-backed
  cognition paths (default endpoint `http://localhost:11434`)

## Setup

```bash
pip install -r requirements.txt
```

## Run tests

```bash
pytest
```

The full suite takes roughly three minutes — the world-simulation tests are
slow by nature, not hung. Every LLM-path test is mock-based, so no Ollama
instance is required to run the suite.

## Layout

```
digital_human/
  types.py          Core dataclasses — SPECIAL, JJDIDTIEBUCKLE, drives, config
  human.py          The DigitalHuman entity: perceive, understand, act
  conversation.py   The two-human interaction protocol
  world.py          The clock, event delivery, and data collection
  llm.py            Ollama-backed cognition paths
configs/
  test_profiles.yaml  Three synthetic humans for testing
tests/              147 tests
```

## Configuration

Humans are defined in YAML. See `configs/test_profiles.yaml` for three fully
specified synthetic profiles. These are fabricated for testing and are not
modeled on real people.

## LLM configuration

The LLM paths read an `llm` section from world config, with defaults:

| Key | Default |
|---|---|
| `base_url` | `http://localhost:11434` |
| `max_tokens` | `150` |

No API keys are used or required.
