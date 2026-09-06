# Architecture

`local-agent` separates four concerns and keeps the boundaries between them strict:

| Concern | Component | Rule |
|---|---|---|
| The brain | `providers/` | Proposes actions. Authorises nothing. |
| Orchestration | `runtime.py` | The only component that executes anything. |
| The hands | `tools/` | Small, auditable capabilities behind one registry. |
| The computer | `workspace/` | The hard filesystem boundary. |

The controlling idea: **the model proposes; only the runtime disposes.** A model
that is confused, jailbroken or prompt-injected can request anything at all, and
still cannot leave the workspace, run an unapproved command, read a credential, or
report a success it did not achieve.

---

## Module map

```text
src/agent/
├── __main__.py     `python -m agent` entry point
├── cli.py          Typer commands; no privileged path into the runtime
├── config.py       Validated settings, precedence, limits, shell policy
├── runtime.py      The lifecycle state machine + the composition root
├── prompts.py      System prompt built from live configuration
├── messages.py     Normalized data model (Message, ToolCall, ToolResult, …)
├── events.py       Lifecycle events, the bus, JSONL logging
├── errors.py       The shared error taxonomy
├── task_state.py   TaskState, statuses, and the transition table
├── providers/      base · factory · gemini · ollama · mock
├── tools/          base · registry · time · filesystem · search · shell ·
│                   memory_tools · verification
├── memory/         database · conversations · facts · tasks · summaries
├── security/       paths · permissions · approvals · redaction · limits
├── skills/         manifest · loader · registry
├── connectors/     config · store · client · tools · manager   (MCP servers)
├── media.py        attachments, content sniffing, per-kind limits
├── manage.py       the management API a CLI or UI drives
└── browser/        base · unavailable        (boundary only)
```

---

## The model/provider boundary

```text
        normalized                        provider-native
runtime ───────────────▶ ModelProvider ───────────────────▶ API
  ▲     Message              adapter          (Gemini / Ollama)
  │     ToolDefinition          │
  └─────────────────────────────┘
        ModelResponse
```

`AgentRunner` sends `list[Message]` and `list[ToolDefinition]`; it receives a
`ModelResponse`. It contains **no** Gemini-specific or Ollama-specific code, and
the integration test `test_the_runtime_is_provider_agnostic` runs the same scenario
under three provider identities to keep it that way.

Each adapter owns:

- role and message conversion (Gemini has no `system` role and no `tool` role;
  Ollama has both, shaped differently);
- schema conversion (Gemini rejects `additionalProperties`, so `sanitize_schema`
  strips what its dialect does not accept — the registry still enforces it locally);
- response parsing, including dropping "thought" parts so private reasoning never
  reaches the runtime, the logs or SQLite;
- retries with exponential backoff and jitter, and timeouts;
- error mapping onto `ErrorCategory`: auth, unavailable, timeout, rate limit,
  invalid request, malformed response, unsupported, unknown;
- explicit capabilities — above all, whether native tool calling is supported.

Credentials come only from the environment. `GeminiProvider` holds the key
privately, never logs it, and scrubs it from any error message it re-raises.

`MockProvider` replays a script of responses (or callables that see the
conversation), which is what lets the entire runtime be tested offline.

---

## The agent lifecycle

```text
                  ┌──────────────────────────────┐
                  ▼                              │
GOAL ──▶ PLAN ──▶ OBSERVE ──▶ ACT ──▶ VERIFY ──▶ FINISH
                                │        │
                                └──▶ REPLAN ◀───┘
```

One iteration of the loop is one **step**. Within a step:

1. **OBSERVE** — compact the conversation if needed, then ask the provider.
2. If the reply has no tool calls and has text → **FINISH**.
3. **ACT** — for each requested call, in order:
   - honour a pending pause or cancel *before* the action, never mid-write;
   - check the per-step and total tool-call budgets;
   - check the retry budget for this exact action signature;
   - hand the call to the registry, which validates, authorises and executes it;
   - record an `ActionRecord`, append a `tool` message, persist to SQLite.
4. **VERIFY** — a `verify_result` call becomes a `VerificationRecord`. A
   successful call to a tool marked `requires_verification` injects a prompt
   demanding evidence before success may be claimed.
5. **REPLAN** — a failure or a failed verification records the failure, increments
   the replan counter and injects a concrete replan hint, bounded by
   `max_retries_per_action` and `max_replans`.

### Termination

`_finish` classifies the outcome rather than trusting the model's own wording:

| Outcome | Meaning |
|---|---|
| `completed` | Finished with no failures and no unverified side effects. |
| `unverified` | Side-effecting actions succeeded but produced no evidence. |
| `partial` | Finished, but failures or denials are on record. |
| `failed` | The provider failed, or the model produced nothing usable. |
| `cancelled` | A cancellation was honoured. |
| `paused` | Stopped cleanly before the next consequential action. |

`RunResult.summary()` separates **verified** work from **completed but not
verified**, from **denied**, from **failed**. That distinction is the whole point:
the system reports what it can prove, not what it intended.

### Handled conditions

Provider failure · malformed tool calls · unknown tools · invalid arguments ·
permission denial · approval denial · tool failure · tool timeout · context
overflow · cancellation · pause · maximum steps · maximum tool calls · failed
verification · repeated identical failures.

---

## The tool registry

`ToolRegistry.execute` is the single gate. Every call passes through, in order:

```text
tool call
   │
   ├─ 1. does the tool exist?          → unknown_tool
   ├─ 2. do the arguments validate?    → invalid_arguments
   ├─ 3. does policy permit it?        → permission_denied
   ├─ 4. does a human approve?         → approval_denied / cancelled
   ├─ 5. execute under a timeout       → tool_timeout / tool_failed
   ├─ 6. redact the output
   └─ 7. truncate long strings         → truncated: true
```

Nothing raises for an expected failure: every problem returns a structured,
unsuccessful `ToolResult`, because the model has to be able to see it and react.

Argument validation is a small JSON-Schema subset implemented in-tree
(`validate_arguments`) supporting types, `required`, `enum`, numeric and length
bounds, array items, and defaults. `additionalProperties` defaults to **false**, so
an unknown argument is rejected rather than silently ignored.

Tools are registered from Python at startup. There is no tool that registers a
tool, and the test `test_the_model_cannot_register_tools` asserts that no tool
schema even mentions registration.

### The tool set

| Tool | Read-only | Approval | Verification |
|---|---|---|---|
| `get_current_time` | yes | no | – |
| `list_files` | yes | no | – |
| `read_file` | yes | no | – |
| `search_files` | yes | no | – |
| `verify_result` | yes | no | – |
| `write_file` | no | **required** | **required** |
| `run_shell` | no | **required** | **required** |
| `remember_fact` | no | **required** | – |

---

## The security layer

Five modules, each with one job:

- **`paths.py`** — resolves every path before any I/O. Rejects traversal, absolute
  escapes, symlinks (a symlink is not followed at all by default), credential files
  and directories, hidden paths, and oversized files.
- **`permissions.py`** — decides whether a tool may run at all, and whether a human
  must be asked. Owns the shell allowlist, the permanently-forbidden command set,
  and the refusal of shell metacharacters.
- **`approvals.py`** — asks a human. `ConsoleApprover` for terminals,
  `PolicyApprover` for tests, `AutoDenyApprover` when no human is present, and
  `UnsafeAutoApprover` behind an explicit, warned flag.
- **`redaction.py`** — removes secrets from strings and nested structures, by
  pattern *and* by the literal values of credential-shaped environment variables.
- **`limits.py`** — counts steps, tool calls, retries and replans against configuration.

Two properties matter most:

1. **Permission before approval.** A forbidden command is refused by policy and
   never reaches a human — so nobody can be socially engineered into approving
   `rm -rf /`. The test `test_destructive_command_is_refused_before_approval`
   asserts the approver was never consulted.
2. **Redaction is central.** The `EventBus` redacts once, so the terminal, the JSONL
   log and the SQLite event table all receive the same already-safe payload.

---

## Task state

`TaskState` is the serialisable record of a task. Statuses and the transition
table live in `task_state.py`, and an illegal transition raises rather than
silently corrupting state:

```text
created ──▶ running ──▶ completed
   │           │  ▲├──▶ failed
   │           │  ││
   │           ▼  ││
   │   waiting_for_approval
   │           │  ││
   │           ▼  ││
   │        paused ┘│
   ▼                ▼
cancelled ◀─────────┘
```

It stores plans, observations, actions, verification evidence, failures and
outcomes — **never hidden chain-of-thought**.

Persistence writes the whole state as JSON alongside indexed columns after every
step, so a task survives a restart. `TaskStore.recover_interrupted` moves tasks
stranded in `running` by a dead process to `paused`, so the CLI never shows a task
as alive when nothing is executing it.

---

## Memory

Five separate layers, deliberately not merged:

1. **Working memory** — `AgentRunner.messages` for the current run.
2. **Conversation history** — `sessions` + `messages` tables.
3. **Durable memory** — the `facts` table. Only human-approved rows are recalled,
   and `remember_fact` refuses to store anything that looks like a credential.
4. **Task state** — the `tasks` table plus `tool_calls`, `verifications`, `events`.
5. **Summaries** — extractive compaction when a conversation exceeds
   `max_conversation_messages`. It never asks the model to summarise itself, and it
   never splits an assistant tool-call message from the `tool` messages answering it.

SQLite via the standard library, with idempotent `user_version` migrations. No
vector database; a `Retriever` protocol exists so semantic search can be added
without changing a caller.

---

## The workspace

```text
workspace/{files,projects,downloads,outputs,temp,state}
```

Created idempotently at startup. `PathPolicy` treats the resolved root as an
absolute boundary, and `PathPolicy.relative` renders paths for the model without
ever revealing the absolute prefix.

---

## Skills

A skill is metadata plus a workflow, discovered from `skills/*/SKILL.md`. The
invariant is enforced in `SkillRegistry.check`:

> A skill declares what it needs. The registry checks that against the tools
> actually registered and the approval mode in force. **A skill can never grant
> itself a permission.**

A malformed skill is reported, not fatal: discovery keeps the valid ones usable and
returns the problems for `local-agent skills list` to display.

---

## Connectors

A **connector** is a named source of extra tools — today, an MCP server over
stdio or streamable HTTP.

```text
ConnectorConfig  what/where/credentials-by-name       config.py
ConnectorStore   JSON at <data_dir>/connectors.json   store.py
MCPConnection    transport, discovery, invocation     client.py
MCPTool          wraps a remote tool as a local Tool  tools.py
ConnectorManager connect, discover, health, shutdown  manager.py
```

The governing idea is that **there is no second path into the machine**. Once
wrapped, a connector's tool is indistinguishable to the runtime from a built-in
one: same registry, same argument validation, same permission check, same
approval prompt, same timeout, same redaction.

What differs is the *default posture*, because a connector is code we did not
write and cannot audit:

- **Disabled by default**, enabled per connector.
- **Namespaced** `mcp__<connector>__<tool>`, so a connector cannot shadow a
  built-in tool. The prefix also makes the origin visible in every approval
  prompt and log line.
- **Side-effecting until declared otherwise.** The runtime cannot know whether a
  remote `search` writes to something, so every remote tool requires approval
  unless an operator has explicitly listed it as read-only.
- **Descriptions are untrusted input.** A hostile server can put instructions in
  a tool description. `sanitize_description` attributes the text to its server,
  neutralises instruction-shaped phrasing, and caps its length; the system prompt
  states that `mcp__` descriptions are documentation, never instruction.
- **Only named environment variables** reach a stdio server, mirroring the shell
  tool's scrubbed environment.
- **Failure is isolated.** Connectors are contacted concurrently and a broken one
  contributes no tools and a reported reason, rather than failing the run.

## Media

`media.py` carries images, audio and (opt-in) video as `Attachment` objects.

- **Content decides the type.** Detection is by magic number; the filename is
  used only in error messages. A declared type that disagrees with the bytes is a
  hard error, because a mislabelled file reaching a model as something it is not
  is exactly the hazard worth refusing.
- **Bounded** per kind, and checked *before* the file is read into memory.
- **Capability-gated.** `ProviderCapabilities` declares `vision`, `audio` and
  `video`. Gemini declares all three; Ollama declares vision only when
  `/api/show` reports it, and never audio or video. `view_media` refuses to load
  what the active model cannot perceive, and `build_runner` does not register the
  tool at all for a text-only model.
- Attachments ride on `ToolResult` under a private key, so they never pass
  through redaction and truncation as if they were text, and never reach the
  model as base64.

## The management API

`manage.py` exposes configuration and inspection as plain Python returning plain
dictionaries: connector CRUD, tool and provider listings, media support, skills,
tasks, sessions, facts, and a dashboard summary. The CLI is a renderer over it,
and a UI would be another — so there is one implementation, not two, and no
privileged path. It never returns a credential: connectors name environment
variables and the API reports only whether they are set.

## Browser boundary

`browser/base.py` defines the normalized capability set (`open_url`, `inspect_page`,
`click`, `type`, `press_key`, `scroll`, `select`, `screenshot`, `download`,
`upload`, `close`), a `BrowserPolicy` that is **deny-by-default** (an empty domain
allowlist permits nothing), and the set of actions that must always require approval.

`browser/unavailable.py` is the only implementation shipped. Every action returns a
structured capability error.

A future adapter must satisfy, before it is enabled: an isolated browser profile
(never the user's real one); a domain allowlist checked before navigation; a
workspace download directory; upload restrictions; navigation and action timeouts;
redacted page data (no cookies, storage, headers or form values to the model);
approval for login, purchases, messages, submissions and account changes; and human
takeover for CAPTCHA, authentication, payment and private data. Unrestricted
browser debugging protocols against a real profile must never be used.

---

## Sub-agent boundary

Not implemented, by design. A future sub-agent system would have to enforce a
restricted objective, a restricted tool set, a workspace scope, a time limit, a
token or cost limit, cancellation, an output contract, conflict handling, and
parent-agent approval — and a model must never be able to create agents, assign
permissions, or recursively spawn uncontrolled tasks.

---

## Composition root

`build_runner()` in `runtime.py` is the one place that knows how everything fits
together: it builds the redactor, the event bus, the workspace, the stores, the
permission checker, the registry with its tools, the skill registry, and the runner.
The CLI calls it like any other caller — there is no privileged path.
