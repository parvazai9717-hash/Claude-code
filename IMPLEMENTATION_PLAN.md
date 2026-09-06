# Implementation Plan — `local-agent`

A local-first, provider-independent autonomous agent runtime with a secure tool layer.

This document is maintained during the build. It records stages, decisions, completed
work, test results, and remaining limitations.

## Repository baseline

Inspected at the start of the build:

- `/home/user/Claude-code` was an **empty git repository** (no commits, no tracked files).
- Branch: `claude/new-session-orxpij`. Remote: `parvazai9717-hash/Claude-code`.
- Platform: Ubuntu 24.04, Linux 6.18, `x86_64`.
- Python 3.11.15 (3.12 and 3.13 also installed), `uv`, `pip`, `ruff`, `mypy`, `pytest`, `make`, `git`.

Because the repository was empty there was no existing work to preserve, so the project
was scaffolded at the repository root: the distribution is named `local-agent`, the
importable package is `agent`, and sources live under `src/agent/`.

## Architectural decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Package root is `src/agent/`, distribution name `local-agent` | `src` layout prevents accidental imports of the working tree and keeps the installed package honest. |
| 2 | Pydantic v2 models for every normalized type | Validation at the boundary is the cheapest place to reject malformed model output. |
| 3 | Providers own **all** wire-format knowledge | `AgentRunner` never sees a Gemini or Ollama shape; a new provider is one file. |
| 4 | Security is a separate package, not decorators on tools | Permission checks must be impossible to skip by registering a tool incorrectly. |
| 5 | The runtime — never the model — decides permissions and executes | A prompt-injected model can request anything; it can authorise nothing. |
| 6 | Approval defaults to `risky` and shell defaults to a **deny-by-default allowlist** | Safe defaults over convenient ones, per the operating rules. |
| 7 | `sqlite3` from the standard library, idempotent schema + `user_version` migrations | No ORM, no extra dependency, and the schema is inspectable with any SQLite client. |
| 8 | Verification is a first-class lifecycle phase with recorded evidence | The runtime must never report success without evidence. |
| 9 | Foreground task lifecycle only; `TaskExecutor` is an interface with no background worker | Detached execution is out of scope for the first stable release. |
| 10 | Browser and sub-agent boundaries return a structured capability error | An interface must not make a capability silently available. |
| 11 | The Gemini adapter targets the installed `google-genai` 2.x SDK, verified by introspection | Rule 7: inspect the installed SDK rather than assume an older API shape. |
| 12 | Tests never touch `$HOME`; every test gets a `tmp_path` workspace and SQLite file | Rule: never modify the user's real home directory during tests. |

## Ambiguities resolved (safest reasonable behaviour, documented)

- **Where does the project live?** The prompt shows a `local-agent/` directory; the repository
  was empty, so the repository root *is* the project root. No nested directory.
- **Chain of thought.** `TaskState` stores plans, observations, actions, verification evidence
  and errors — never hidden reasoning. Provider "thought" parts are dropped, not persisted.
- **`verify_result`.** Implemented as a real read-only verifier (file existence, content,
  content match, shell exit status re-check) rather than a model-asserted claim, because a
  model-asserted verification would violate the "no success without evidence" rule.
- **Local models without tool support.** `OllamaProvider` raises a structured capability
  error instead of falling back to text parsing, so a tool call is never fabricated.
- **`automatic` approval mode.** Only tools on an explicit `auto_approve_tools` allowlist run
  unattended; hard restrictions (workspace containment, shell policy, secret redaction) stay on.

## Sequence adjustment (documented, per rule 20)

The mandated order places security (phase 5) after the read-only tools (phase 4). The
tools were written against `security/paths.py` from the start rather than being written
unsafely and retrofitted, so `security/` landed before `tools/`. Every phase still ended
runnable and tested; only the file-creation order changed, not the scope or the outcome.

## Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Inspect and scaffold: plan, packaging, source tree, test tree, config examples, basic CLI | complete |
| 2 | Core types and configuration: messages, tools, results, task state, events, errors, config | complete |
| 3 | Mock provider and runtime lifecycle state machine | complete |
| 4 | Tool registry and read-only tools; workspace initialisation | complete |
| 5 | Security: paths, permissions, approvals, limits, redaction | complete |
| 6 | Write, shell and verification tools; replan behaviour | complete |
| 7 | SQLite persistence, task lifecycle, restart recovery, summaries | complete |
| 8 | Skills: manifest, discovery, validation, permission matching | complete |
| 9 | Gemini adapter | complete (offline tests; live call pending credentials) |
| 10 | Ollama adapter | complete (offline tests; live call pending a local server) |
| 11 | CLI polish: all commands, doctor, progress events, data clearing | complete |
| 12 | Browser boundary: normalized interface + unavailable implementation | complete |
| 13 | Quality and documentation | complete |

## Test results

Final run, offline, with no network, no API key, no Ollama server and no browser:

```text
$ .venv/bin/python -m pytest
418 passed in 3.83s

$ .venv/bin/python -m pytest tests/unit          341 passed
$ .venv/bin/python -m pytest tests/integration    77 passed

$ .venv/bin/python -m ruff format src tests scripts   all formatted
$ .venv/bin/python -m ruff check  src tests scripts   All checks passed!
$ .venv/bin/python -m mypy                            no issues in 44 source files
$ .venv/bin/python scripts/demo_offline.py            DEMO PASSED
```

Coverage by area: configuration and precedence · normalized models · provider factory ·
mock provider · malformed provider responses · Gemini adapter (fake client) · Ollama
adapter (`httpx.MockTransport`) · tool schemas · registry behaviour · path traversal and
symlink escape · secret-file protection · file limits · output truncation · shell timeout
and policy · approval modes · redaction · task-state transitions · pause, resume and
cancellation · verification and failed verification · replan limits · SQLite persistence ·
durable-memory approval · skill metadata and permissions · unavailable browser · step and
tool-call limits · the interactive approval prompt · the `chat` command and its slash
commands · the CLI end to end.

The ten required integration scenarios all pass: conversation with no tools; read then
verified answer; approved write; approval denial; tool failure and replan; verification
failure and bounded retry; pause and resume across a simulated process restart;
cancellation; provider switching under the same runtime tests; and recovery after restart.

## Bugs found and fixed during the build

Recorded because each was caught by a test rather than by inspection:

1. **`Authorization:` header redaction was incomplete** — the pattern matched only the
   first token after the colon, so `Bearer <token>` survived with the token intact. Now the
   whole remainder of the line is removed.
2. **`list_files` descended one level too deep** — `depth: 1` returned the contents of
   subdirectories. The guard now stops before the level that would exceed the requested depth.
3. **`RunResult.summary()` listed read-only actions as "unverified"** — a read makes no
   claim about the world, so listing it diluted the one signal that matters. Only
   side-effecting actions are reported as unverified now.
4. **`TaskStore.recover_interrupted` contained a malformed expression** left from an
   earlier draft; it now records a proper failure entry on recovery.
5. **`list` as a method name shadowed the builtin inside its own class**, which mypy caught
   as a genuine typing hazard. Renamed to `list_tasks` / `list_facts` / `list_summaries`.
6. **Three untested paths were reported as complete**: `ConsoleApprover`, the `chat`
   command, and the slash commands had no automated coverage, and the first was wrongly
   described as untestable here. All three are now covered.

## Manual tests that could not be run here

These need credentials or a local server, neither of which exists in this environment.
They are **pending**, not passing:

| Test | Command | Why it could not run |
|---|---|---|
| Live Gemini generation | `local-agent -p gemini chat "..."` | No `GEMINI_API_KEY` available. |
| Live Gemini health check | `local-agent -p gemini doctor` | Same. Verified only that it reports the missing key correctly. |
| Live Ollama generation | `local-agent -p ollama chat "..."` | No Ollama server running. |
| Ollama model capability probe | `local-agent -p ollama doctor` | Same. |

An earlier draft of this table also listed the interactive approval prompt as
untestable "because it needs a TTY". That was wrong: `ConsoleApprover` writes to an
injectable `rich` console and reads one keypress from stdin, both of which a test can
supply. The gap was missing coverage, not an environmental limit, and it is now closed by
`tests/unit/test_console_approver.py` (13 cases) and `tests/integration/test_cli_chat.py`
(18 cases covering `chat`, the approval prompt in situ, and every slash command).


Both adapters are covered offline: Gemini through a fake client that asserts schema
conversion, response parsing, thought-part suppression, disabled automatic function
calling and the full error-mapping table; Ollama through `httpx.MockTransport` covering
text replies, tool calls, missing models, malformed JSON, connection failures, timeouts
and the tool-capability probe.

## Known limitations

Tracked in `README.md` ("Limitations") and `SECURITY.md`. In short: foreground tasks only;
no browser; no sub-agents; no arbitrary code execution; extractive summarisation only; no
semantic retrieval; argv-only shell; and structural verification that checks whether a
file exists or contains text, not whether its content is *correct*.
