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
| 14 | Multimodal: images and audio as first-class input | complete |
| 15 | Connectors: MCP servers, security-gated | complete (offline tests; live server pending) |
| 16 | Management API and connector CLI | complete |
| 17 | UI master prompt | complete (the UI itself is not built) |

## Phases 14-17: connectors, multimodal, and the UI brief

Added after the first stable release, at the user's request: MCP connectors, image
and audio understanding, a UI-facing management API, and a master prompt for building
the UI itself.

### Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 13 | MCP servers are modelled as **connectors**, not a bespoke subsystem | "Add MCP and connectors" is one concept, not two: a connector is a named source of tools, and MCP is its first kind. One UI page, one CRUD surface. |
| 14 | Connector tools go through the **existing registry**, unchanged | A second, weaker path into the machine is the thing worth not building. Namespacing, approval, redaction and limits all apply unchanged. |
| 15 | Remote tools are **side-effecting until an operator says otherwise** | The runtime cannot know whether a remote `search` writes to something. Guessing optimistically is how an agent deletes a database. |
| 16 | Server-supplied tool descriptions are **untrusted input** | A hostile server can put instructions in a description. They are attributed, defanged and capped, and the prompt states they are documentation. |
| 17 | Only **named** environment variables reach a stdio server | Mirrors the shell tool's scrubbed environment: a connector must not inherit every credential the agent holds. |
| 18 | Media type is decided by **magic number, never extension** | A zip renamed `.png` reaching a model as an image is a real hazard; a declared type disagreeing with the bytes is an error, not a guess. |
| 19 | `view_media` is **not registered** for a text-only model | A tool whose result the model cannot perceive is worse than no tool: it burns a call and invites a fabricated description. |
| 20 | **Video is opt-in and off by default** | The request said "if videos are difficult, then not videos". The type exists and Gemini declares support, but nothing accepts video unless it is explicitly enabled. |
| 21 | `manage.py` is a **plain-Python API**, and the CLI is a renderer over it | A UI is another renderer over the same calls, so there is one implementation and no privileged path. |
| 22 | Connector definitions live in **`connectors.json`**, not `config.yaml` | They are the part of configuration edited *through* the application; a UI writing them must not clobber a hand-written config file. |

### Scope note

Adding MCP genuinely widens the attack surface — a connector is third-party code
providing tools to an autonomous agent. That is the user's call to make, and it was
made explicitly. The build's response was to give connectors a tighter default leash
than anything built in (disabled, namespaced, approval-gated, credential-scoped) and
to document plainly, in `SECURITY.md`, what the remaining exposure is: an enabled
connector you approve an action for can do whatever that action does remotely.

## Test results

Final run, offline, with no network, no API key, no Ollama server and no browser:

```text
$ .venv/bin/python -m pytest
558 passed in 4.27s

$ .venv/bin/python -m pytest tests/unit          443 passed
$ .venv/bin/python -m pytest tests/integration   115 passed
$ .venv/bin/python -m pytest --cov=agent         91% statement coverage

$ .venv/bin/python -m ruff format src tests scripts   all formatted
$ .venv/bin/python -m ruff check  src tests scripts   All checks passed!
$ .venv/bin/python -m mypy                            no issues in 53 source files
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
7. **The extension fallback defeated content verification.** The first draft of
   `media.py` fell back to the filename when the bytes matched no signature, so a zip
   renamed `.png` passed as an image. Detection is now content-only, and an
   unrecognised file is refused with a message saying renaming will not help.
8. **The test suite wrote to the real home directory on Windows.** The CLI
   fixtures set only `HOME`, which isolates nothing on Windows: `expanduser`
   reads `USERPROFILE` there and ignores `HOME` entirely. A plain `pytest` run
   therefore left its tasks, facts and connectors in the user's real
   `~/.local-agent`. Found by a user running the installer on Windows and
   noticing `doctor` report 13 tasks and three connectors on a fresh install.
   The fixtures are now one shared `isolated_home` in `conftest.py` that sets
   `HOME`, `USERPROFILE`, `HOMEDRIVE` and `HOMEPATH`, asserts the redirection
   took effect before any test uses it, and is pinned by
   `tests/integration/test_home_isolation.py` — which exercises `posixpath` and
   `ntpath` expansion side by side so a Linux run catches a Windows-only hole.
9. **Redaction missed most Google credential formats.** Only `AIza…` keys were
   matched. AI Studio issues keys in an `AQ.…` form, and OAuth uses `ya29.…`,
   `GOCSPX-…` and `1//…`; none matched, so such a credential was redacted only
   when written as `NAME=value`, and passed through untouched in prose or in a
   JSON value. Found when a user pasted a real key of that shape. All five forms
   are now matched and pinned by a test that checks each one in five contexts.
10. **A truncated id could not be pasted back.** `rich` clips a long id to fit the
   terminal (`sess_f42d09c07e…`), so a user copying one off their own screen got
   "no session named …". Sessions now resolve by unambiguous prefix — as tasks already
   did — and both strip a trailing ellipsis. Found by a test that scraped a rendered
   table, which is exactly what a user does by hand.

## Manual tests that could not be run here

These need credentials or a local server, neither of which exists in this environment.
They are **pending**, not passing:

| Test | Command | Why it could not run |
|---|---|---|
| Live Gemini generation | `local-agent -p gemini chat "..."` | No `GEMINI_API_KEY` available. |
| Live Gemini health check | `local-agent -p gemini doctor` | Same. Verified only that it reports the missing key correctly. |
| Live Ollama generation | `local-agent -p ollama chat "..."` | No Ollama server running. |
| Ollama model capability probe | `local-agent -p ollama doctor` | Same. |
| A live MCP server over stdio | `local-agent connectors test NAME` | No MCP server installed (`npx` unavailable). Partially exercised: a real connect was attempted and correctly timed out without breaking the run. |
| A live MCP server over HTTP | `local-agent connectors test NAME` | No reachable MCP endpoint. |
| Image understanding against a real model | `local-agent -p gemini chat "describe files/x.png"` | Needs a Gemini key. The attachment path is covered offline end to end. |

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
