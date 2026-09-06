# local-agent

A local-first autonomous AI agent with a secure tool layer.

The **model is the brain**, the **runtime is the orchestration layer**, **tools are the
hands**, and the **workspace is the computer**. The model proposes actions; only the
runtime validates permissions and executes them.

Works with **Google Gemini** through its API and with **local models** through an
Ollama-compatible API. The provider boundary is strict: switching between them
changes nothing in the runtime, the tools or the security layer.

---

## Contents

- [What it does](#what-it-does)
- [Installation](#installation)
- [Configuration](#configuration)
- [Gemini setup](#gemini-setup)
- [Ollama setup](#ollama-setup)
- [The workspace](#the-workspace)
- [CLI](#cli)
- [Approvals](#approvals)
- [Memory](#memory)
- [Tasks](#tasks)
- [Skills](#skills)
- [Connectors and MCP servers](#connectors-and-mcp-servers)
- [Images and audio](#images-and-audio)
- [Building a UI](#building-a-ui)
- [Security](#security)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

---

## What it does

`local-agent` runs a goal through an explicit state machine rather than an
uncontrolled loop:

```text
GOAL → PLAN → OBSERVE → ACT → VERIFY → FINISH or REPLAN
```

- **PLAN** — interpret the goal; a minimal plan for simple work, a short checklist for complex work.
- **OBSERVE** — gather state with read-only tools. Nothing is modified.
- **ACT** — validate the tool name and arguments, check permissions, ask for approval if required, execute under limits.
- **VERIFY** — after every consequential action, gather **evidence**. No evidence, no claim of success.
- **REPLAN** — on failure or failed verification, revise, with bounded retries.
- **FINISH** — distinguish completed work, partial work, failures, denied approvals and unverified outcomes.

### Capabilities in the first stable release

Conversational interaction · multi-step planning · structured model tool calls ·
workspace inspection · safe file reading, searching and writing · controlled shell
commands · persistent workspace · explicit task state · pause / resume / cancel /
restart recovery · durable conversation history · user-approved memory · reusable
skill metadata · Gemini or local-model switching · verification after consequential
actions · structured progress events, approvals, errors and summaries ·
**MCP server connectors** · **image and audio understanding**.

### Deliberately *not* enabled

Browser automation, screenshots and computer vision, background daemons,
scheduled tasks, sub-agents, external messaging, payments, account actions,
deployments, and arbitrary Python execution. Interfaces exist for the browser
and sub-agent boundaries; **an interface does not make a capability available**.

---

## Installation

Requires Python 3.11 or newer. Works on macOS, Linux and Windows.

**New here? [`INSTALL.md`](INSTALL.md) is a step-by-step guide** covering all three
platforms, both providers, and the first run. The short version follows.

```bash
git clone <this-repository>
cd <this-repository>

# with uv (recommended)
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev,gemini]"

# or with pip
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,gemini]"
```

Or simply:

```bash
make install
```

Verify:

```bash
.venv/bin/local-agent --help
.venv/bin/local-agent doctor
```

The `gemini` extra installs the Google Gen AI SDK. Omit it if you only use Ollama
or the offline mock provider.

---

## Configuration

Two sources, with environment variables taking precedence:

1. `config.yaml` (or `config.toml`) in the working directory — copy `config.example.yaml`.
2. Environment variables, from the shell or a `.env` file — copy `.env.example`.

Precedence, highest first: **CLI flags → environment variables → config file → defaults**.

```bash
cp config.example.yaml config.yaml
cp .env.example .env      # .env is git-ignored; put credentials only here
```

Every setting can be overridden with a `LOCAL_AGENT_` prefix:

```bash
LOCAL_AGENT_PROVIDER=gemini LOCAL_AGENT_APPROVAL_MODE=always local-agent chat
```

Key settings:

| Setting | Default | Meaning |
|---|---|---|
| `provider` | `ollama` | `gemini`, `ollama` or `mock` |
| `workspace` | `./workspace` | Hard filesystem boundary for every file tool |
| `data_dir` | `~/.local-agent` | SQLite database and JSONL event log |
| `approval_mode` | `risky` | `always`, `risky` or `automatic` |
| `shell_allowed_commands` | read-only inspection commands | Deny-by-default shell allowlist |
| `max_steps` | `20` | Maximum agent steps in one run |
| `max_total_tool_calls` | `40` | Maximum tool calls in one run |
| `max_file_bytes` | `1000000` | Refusal threshold for file reads and writes |
| `max_tool_output_chars` | `12000` | Tool output is truncated above this |

`local-agent config show` prints the effective configuration. **API keys are never
printed** — only whether the variable is set.

---

## Gemini setup

1. Create an API key at <https://aistudio.google.com/apikey>.
2. Put it in `.env` (never in `config.yaml`):

   ```bash
   GEMINI_API_KEY=your-key-here
   ```

3. Select the provider:

   ```yaml
   provider: gemini
   gemini_model: gemini-2.5-flash
   gemini_timeout_seconds: 120
   ```

4. Check it:

   ```bash
   local-agent doctor --provider gemini
   local-agent chat --provider gemini --model gemini-2.5-flash
   ```

The adapter uses the official `google-genai` SDK and disables the SDK's automatic
function calling: the runtime executes tools itself, so that the security layer
sits between the model and every action.

---

## Ollama setup

1. Install Ollama from <https://ollama.com> and start it:

   ```bash
   ollama serve
   ```

2. Pull a **tool-capable** model — tool calling is required:

   ```bash
   ollama pull llama3.1
   ```

3. Select the provider:

   ```yaml
   provider: ollama
   ollama_base_url: http://localhost:11434
   ollama_model: llama3.1
   ```

4. Check it:

   ```bash
   local-agent doctor --provider ollama
   local-agent chat --provider ollama --model llama3.1
   ```

`doctor` reports whether the server is reachable, whether the model is installed,
and whether it advertises tool support. A model without tool support produces a
clear capability error — the agent never fabricates a tool call from free text.

---

## The workspace

The workspace is the agent's computer and its hard boundary:

```text
workspace/
├── files/       general working files
├── projects/    code or documents being worked on
├── downloads/   fetched content (reserved for future use)
├── outputs/     generated results
├── temp/        scratch space
└── state/       runtime state files
```

Every filesystem tool refuses:

- `..` traversal and absolute paths outside the workspace;
- symlink escapes (a symlink is not followed at all by default);
- credential files (`.env`, `id_rsa`, `*.pem`, `credentials.json`, …);
- credential directories (`.ssh`, `.aws`, `.gnupg`, browser profiles, …);
- hidden files, unless explicitly requested and still not credential-bearing;
- files above `max_file_bytes`, and binary files for text tools.

---

## CLI

```bash
local-agent chat                                   # interactive session
local-agent chat "summarise files/notes.txt"       # one goal, then exit
local-agent chat --provider gemini --model gemini-2.5-flash
local-agent chat --provider ollama --model llama3.1

local-agent doctor                                 # health checks
local-agent config show                            # effective configuration

local-agent task create "GOAL"
local-agent task list
local-agent task run TASK_ID
local-agent task status TASK_ID
local-agent task pause TASK_ID
local-agent task resume TASK_ID
local-agent task cancel TASK_ID
local-agent task events TASK_ID
local-agent task recover                           # after a crash

local-agent memory list
local-agent memory remove ID

local-agent sessions list
local-agent sessions show ID
local-agent sessions delete ID

local-agent skills list
local-agent skills show NAME

local-agent clear-data                             # delete all local data
```

In interactive chat: `/help`, `/status`, `/tools`, `/model`, `/memory`, `/tasks`,
`/clear`, `/quit`. **Ctrl-C** requests cancellation and the run stops cleanly
before its next consequential action; it does not kill the process mid-write.
**Ctrl-D** exits.

---

## Approvals

| Mode | Behaviour |
|---|---|
| `always` | Every action that is not read-only is confirmed. |
| `risky` *(default)* | Writes, shell, network, deletion, installs, auth, external effects and memory changes are confirmed. Reads run automatically. |
| `automatic` | Only tools on `auto_approve_tools` run. Everything else is **denied**, because no human is available to ask. |

An approval prompt always shows the tool, the **redacted** arguments, the risk
level and category, the concrete target, and whether the action is reversible.
Choices are: approve once, approve for this run, deny, or cancel the run.

`unsafe_disable_approvals` exists for tests only. It is never the default, and the
CLI prints a warning whenever it is active.

---

## Memory

Five separate layers:

1. **Working memory** — the in-memory conversation for the current run.
2. **Conversation history** — persisted sessions and messages (SQLite).
3. **Durable memory** — facts a human explicitly approved. Nothing reaches this table without approval.
4. **Task state** — the full lifecycle record of each task.
5. **Summaries** — long conversations are compacted extractively, never by asking the model to summarise itself.

The model can *propose* a fact with `remember_fact`; a human must approve it before
it is stored, and a proposal that looks like a credential is refused outright.

```bash
local-agent memory list
local-agent memory remove 3
local-agent clear-data
```

No vector database is used. A `Retriever` interface exists so semantic search can
be added later without changing any caller.

---

## Tasks

Tasks are foreground only in this release: nothing keeps running after the CLI exits.

```bash
local-agent task create "Audit the project files and write a summary"
local-agent task run task_ab12cd34
local-agent task pause task_ab12cd34     # stops before the next consequential action
local-agent task resume task_ab12cd34    # revalidates configuration, then continues
local-agent task cancel task_ab12cd34
local-agent task status task_ab12cd34    # plan, evidence, failures, outcome
local-agent task events task_ab12cd34
```

Task state is persisted after every step, so a task survives a process restart.
After a crash, `local-agent task recover` moves tasks stranded in `running` to
`paused` so they can be resumed rather than appearing falsely alive.

Statuses: `created`, `running`, `waiting_for_approval`, `paused`, `completed`,
`failed`, `cancelled`. Transitions are validated; an invalid transition raises.

---

## Skills

A skill is a directory containing `SKILL.md` with YAML front matter:

```text
skills/
└── example/
    ├── SKILL.md
    ├── templates/   (optional)
    └── scripts/     (optional)
```

`SKILL.md` declares its name, description, activation conditions, required tools,
allowed paths and domains, expected inputs and outputs, approval requirements and
safety limitations.

**A skill cannot grant itself permissions.** It declares what it needs; the registry
checks that against the tools actually registered and the approval mode in force.
A skill needing an unavailable tool simply becomes unavailable. Scripts inside a
skill run through the same tool and security layer as anything else.

```bash
local-agent skills list      # shows why any skill is unavailable
local-agent skills show example
```

---

## Security

Full detail is in [`SECURITY.md`](SECURITY.md). In short:

- The model **proposes**; only the runtime **validates and executes**.
- The model cannot approve its own risky actions, and has no tool that reaches an approver.
- Filesystem access is confined to the workspace, with credential paths refused.
- Shell is **deny-by-default**: an allowlist, argv execution with no shell, no metacharacters, a scrubbed environment, timeouts and process-group cleanup.
- Secrets are redacted before display, logging, persistence **and model context**.
- The system prompt is not a security boundary; every restriction is enforced in code.
- Nothing claims success without verification evidence.

---

## Testing

The whole suite runs offline: **no network, no API key, no Ollama server, no
browser, no external account.** Tests use temporary directories and temporary
SQLite databases, and `~` is redirected into a temporary directory on every
platform — `HOME`, `USERPROFILE`, `HOMEDRIVE` and `HOMEPATH` are all set,
because `expanduser` consults different ones on POSIX and Windows.
`tests/integration/test_home_isolation.py` asserts this directly.

```bash
make test                    # everything
make test-unit
make test-integration
make check                   # format + lint + typecheck + test
```

---

## Troubleshooting

**`GEMINI_API_KEY is not set`** — put the key in `.env` or export it. `local-agent doctor`
reports whether it is present without printing it.

**`cannot reach the Ollama server`** — start it with `ollama serve` and check
`ollama_base_url`.

**`the model X is not installed`** — `ollama pull X`.

**`the local model X does not support tool calling`** — choose a tool-capable model
(llama3.1, qwen2.5, mistral-nemo, …).

**`path escapes the workspace boundary`** — working as designed. Use a
workspace-relative path.

**`'curl' is not in the shell allowlist`** — also working as designed. Add the
command to `shell_allowed_commands` only if you are certain, and note that some
commands can never be allowlisted.

**`reached the maximum of N agent steps`** — raise `max_steps`, or break the goal into
smaller tasks.

**A task looks stuck in `running`** — its process died. Run `local-agent task recover`,
then `task resume`.

---

## Limitations

- **Foreground only.** No daemon, no background worker, no scheduled tasks.
- **No browser.** The interface exists; every action returns a capability error.
- **No sub-agents.** The boundary is documented; nothing is implemented.
- **No arbitrary code execution.** No `exec`, no `eval`, no unrestricted Python.
- **Extractive summarisation.** Compaction never asks the model to summarise itself.
- **No semantic retrieval.** Recent-message retrieval only; the interface allows more later.
- **Shell is intentionally narrow.** No pipes, redirection, or shell metacharacters — argv only.
- **Verification is structural.** It checks that a file exists, contains text, is absent, or that a command exits zero. It cannot judge whether the *content* is correct.
- **Live provider calls are not covered by the test suite** — adapters are tested with fakes, since the suite must run with no network or credentials.
- **Connectors are tested against a fake MCP session**, not a live server, for the same reason.
- **Video input is defined but off by default**, and only Gemini declares support for it.
- **No UI yet.** The management API and the brief for building one are in place; the UI itself is not built.

---

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the model/provider boundary, lifecycle, registry, security layer, task state, memory, workspace, skills, and the browser and sub-agent boundaries.
- [`SECURITY.md`](SECURITY.md) — threat model, assumptions, secret handling, filesystem and shell restrictions, approvals, data deletion, logging, safe deployment.
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — phases, decisions, test results, remaining limitations.

## License

MIT.
