# Digital Human — Engine Repo

## Context

Digital Human is a self-contained behavioral simulation engine. Architecture is
**person-first**: the base DigitalHuman (SPECIAL scores, JJDIDTIEBUCKLE scores,
drives, inner world, memory, relationships, stress) is the core. It must
simulate ANY person type — employee, customer, regulator, competitor — not just
employees. Work context is an optional layer.

The architecture spec is not in this repo. Read it before design work — do not
proceed from memory of what it "probably" says.

## Working Rules (permanent)

- **Don't invent** — flag ambiguities as numbered questions and stop for
  resolution.
- **Pre-design before code** — propose, stop for sign-off, then implement.
- **All existing tests stay green.** Test command: `pytest`
- **Never commit** — maintainers run commits themselves (message format
  `"vX.Y: description"`).
- This repo stays self-contained: no imports from downstream product code.
- **Public repo.** No credentials, no internal hostnames, no strategy or
  roadmap documents. Design docs belong in the private docs repo.

## Environment

- Python 3.12. Primary development is on Windows/PowerShell; CI runs Linux.
- Ollama running locally (qwen2.5 3b/7b/14b/32b) for LLM paths.
- Full test suite ~3min; the world-sim tests are slow, not hung. LLM-path
  tests are mock-based and need no Ollama.
