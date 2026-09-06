# MASTER PROMPT: Build a Web UI for the `local-agent` Runtime

> **How to use this document.** Hand it to Claude Code (or any capable coding
> agent) in a checkout of this repository. It describes what already exists, what
> to build, and the rules the UI must respect. Everything in Part 1 is *fact about
> this codebase*, not aspiration — the agent, connectors and multimodal support
> are implemented and tested. Part 2 onward is the work to do.

---

## Part 0 — Role and non-negotiable rules

You are the lead engineer building a local web UI for an existing, working
autonomous agent runtime. Build it in this repository. Inspect the code before
you change it. Do not stop at mockups, scaffolding or TODOs.

1. **Never bypass the runtime's security layer.** The UI is a *renderer* over
   `agent.manage.AgentManager` and `agent.runtime.AgentRunner`. If the UI needs a
   capability, add it to the management API where the existing checks apply —
   never reach around into the filesystem, the shell, or a provider directly.
2. **Never send a credential to the browser.** The API reports whether an
   environment variable is *set*. It never returns its value. No exceptions.
3. **Approvals are the whole point.** Every approval-gated action must reach a
   human in the UI and wait for a real answer. An auto-approving UI is a bug.
4. **The server binds to localhost only** by default, and requires an explicit,
   documented flag to do anything else.
5. **Tests must run offline** — no network, no API key, no MCP server, no browser.
   The existing suite does this; keep it that way.
6. **Do not weaken existing tests to make new code pass.** If a test fails,
   either the code is wrong or the test encodes a decision worth revisiting —
   say which.
7. If something is ambiguous, choose the safest behaviour, document it, continue.

---

## Part 1 — What already exists

### 1.1 The agent, in one paragraph

`local-agent` is a local-first autonomous agent. The **model is the brain**, the
**runtime is the orchestration layer**, **tools are the hands**, and the
**workspace is the computer**. The model *proposes* actions; only the runtime
validates permissions and executes them. It works with Google Gemini and with
local models through an Ollama-compatible API, and the provider boundary is
strict — swapping one for the other changes nothing in the runtime, the tools or
the security layer.

### 1.2 The lifecycle

```text
GOAL → PLAN → OBSERVE → ACT → VERIFY → FINISH or REPLAN
```

- **PLAN** — interpret the goal. Minimal for simple work, a short checklist for complex.
- **OBSERVE** — gather state with read-only tools. Nothing is modified.
- **ACT** — validate the tool name and arguments, check permissions, ask for
  approval if required, execute under a timeout.
- **VERIFY** — after every consequential action, gather **evidence**. No
  evidence, no claim of success.
- **REPLAN** — on failure or failed verification, revise. Retries are bounded.
- **FINISH** — classify the outcome honestly.

One loop iteration is one **step**. Outcomes are classified by the runtime, not
by the model's own wording:

| Outcome | Meaning |
|---|---|
| `completed` | Finished, no failures, no unverified side effects. |
| `unverified` | Side effects succeeded but produced no evidence. |
| `partial` | Finished, but failures or denials are on record. |
| `failed` | The provider failed, or the model produced nothing usable. |
| `cancelled` | A cancellation was honoured. |
| `paused` | Stopped cleanly before the next consequential action. |

Task statuses: `created`, `running`, `waiting_for_approval`, `paused`,
`completed`, `failed`, `cancelled`. Transitions are validated; an illegal one raises.

### 1.3 The tools

| Tool | Read-only | Approval | Verification |
|---|---|---|---|
| `get_current_time` | yes | no | – |
| `list_files` | yes | no | – |
| `read_file` | yes | no | – |
| `search_files` | yes | no | – |
| `view_media` | yes | no | – |
| `verify_result` | yes | no | – |
| `write_file` | no | **required** | **required** |
| `run_shell` | no | **required** | **required** |
| `remember_fact` | no | **required** | – |

Plus any tools contributed by enabled connectors, namespaced `mcp__<connector>__<tool>`.

### 1.4 The security layer (`src/agent/security/`)

- **`paths.py`** — the workspace is a hard boundary. Rejects `..` traversal,
  absolute escapes, `~`, symlinks (not followed at all by default), credential
  files (`.env*`, `id_rsa`, `*.pem`, `credentials.json`, …), credential
  directories (`.ssh`, `.aws`, `.gnupg`, browser profiles, …), hidden files, and
  oversized files.
- **`permissions.py`** — decides whether a tool may run at all. Owns the
  deny-by-default shell allowlist and a permanently-forbidden command set that
  cannot be allowlisted (`sudo`, `rm`, `curl`, `bash`, `pip`, …).
- **`approvals.py`** — asks a human. `ConsoleApprover` (terminal),
  `PolicyApprover` (tests), `AutoDenyApprover` (no human present),
  `UnsafeAutoApprover` (explicitly flagged, warned, never default).
- **`redaction.py`** — removes secrets by key name, by value shape (Google/OpenAI/
  GitHub/AWS/Slack tokens, JWTs, PEM blocks, `Authorization:` headers) and by the
  literal values of credential-shaped environment variables.
- **`limits.py`** — steps, tool calls per step, total tool calls, retries, replans.

**Order matters: permission first, approval second.** A forbidden action is
refused by policy and never reaches a human, so nobody can be talked into
approving `rm -rf /`.

### 1.5 Memory (`src/agent/memory/`) — five separate layers

1. **Working memory** — the in-run message list.
2. **Conversation history** — `sessions` + `messages` tables.
3. **Durable memory** — the `facts` table; only human-approved rows are recalled.
4. **Task state** — `tasks`, `tool_calls`, `verifications`, `events`.
5. **Summaries** — extractive compaction; never asks the model to summarise itself.

SQLite via the standard library, `user_version` migrations. No vector database; a
`Retriever` protocol exists so semantic search can be added later.

### 1.6 Connectors and MCP (`src/agent/connectors/`) — **already built**

A **connector** is a named source of extra tools. Today that means an **MCP
server** over stdio or streamable HTTP.

```text
ConnectorConfig  → what/where/credentials-by-name    (config.py)
ConnectorStore   → JSON at <data_dir>/connectors.json (store.py)
MCPConnection    → stdio / streamable-HTTP transport  (client.py)
MCPTool          → wraps a remote tool as a local Tool (tools.py)
ConnectorManager → connect, discover, health, shut down (manager.py)
```

The security stance, because a connector is third-party code:

- **Disabled by default.** Each is enabled explicitly.
- **Namespaced** `mcp__<connector>__<tool>`, so a connector cannot shadow `read_file`.
- **Approval-gated by default.** The runtime cannot know whether a remote
  `search` writes to something. An operator may mark specific tools read-only.
- **Descriptions are untrusted.** A hostile server can put instructions in a tool
  description. They are attributed to their origin, neutralised of
  instruction-shaped phrasing, and length-capped. The system prompt tells the
  model these are documentation, never instruction.
- **Only named environment variables** are forwarded to a stdio server. It cannot
  inherit every credential the agent holds.
- **Bare command names only** — a stdio connector cannot point at an arbitrary path.
- **A broken connector never breaks a run.** Its tools are absent and the reason
  is reported.

### 1.7 Multimodal (`src/agent/media.py`) — **already built**

- **Images and audio are first-class.** The `view_media` tool loads a workspace
  file so the model can actually look at or listen to it.
- **Content decides the type, never the filename.** A zip renamed `.png` is
  refused. Detection is by magic number; a declared type that disagrees with the
  bytes is a hard error.
- **Per-kind size caps** (`max_image_bytes`, `max_audio_bytes`, `max_video_bytes`).
- **Capability-gated.** Gemini declares image + audio + video. Ollama declares
  vision only when `/api/show` says the model has it, and never audio or video.
  A text-only model is **not offered `view_media` at all** — a tool whose results
  the model cannot perceive is worse than no tool.
- **Video is opt-in and off by default** (`media.enable_video`), because it is
  large, costly, and unsupported by most local models. Enabling it does not make
  an incapable provider accept it.
- Connector tools returning image or audio blocks become attachments too.

### 1.8 The management API (`src/agent/manage.py`) — **build the UI on this**

`AgentManager` is the surface the CLI already uses, and the one the UI must use.
It returns plain dictionaries and never returns a credential.

```python
AgentManager(config=Config, database=Database)

# connectors
.list_connectors()             -> list[dict]   # credential *presence* only
.add_connector(name=, kind=, command=, args=, url=, env=, header_env=,
               description=, enabled=, tool_allowlist=, read_only_tools=,
               timeout_seconds=, replace=) -> dict
.remove_connector(name)        -> bool
.set_connector_enabled(name, enabled) -> dict | None
.test_connectors(name=None)    -> list[dict]   # async; contacts servers
.connector_manager(**kwargs)   -> ConnectorManager

# capabilities
.list_tools()                  -> list[dict]   # risk, approval, schema
.list_providers()              -> list[dict]   # capabilities, credential presence
.check_provider()              -> dict         # async health check
.media_support()               -> dict         # accepted kinds, limits, MIME types
.list_skills()                 -> list[dict]

# history
.list_tasks(status=, limit=)   -> list[dict]
.get_task(task_id)             -> dict | None
.list_sessions(limit=)         -> list[dict]
.list_facts(approved_only=)    -> list[dict]
.stats()                       -> dict         # dashboard summary
.safe_config()                 -> dict
```

### 1.9 Events — how the UI streams progress

`agent.events.EventBus` fans out already-redacted `Event` objects. Subscribe and
forward to the browser. The types the UI cares about:

```text
task_started · task_status_changed · task_finished
step_started · phase_entered · plan_updated · observation
provider_request · provider_response · provider_error
tool_requested · approval_requested · approval_result
tool_started · tool_result
verification_started · verification_result
replan · limit_reached · error · message · final_answer
```

Every event carries `type`, `timestamp`, and optionally `task_id`, `session_id`,
`step`, `phase`, `provider`, `model`, `message`, `data`. **Redaction happens once,
centrally, on the bus** — so anything the UI receives is already safe to display.

### 1.10 Running the tests

```bash
make test        # 552 tests, fully offline
make check       # format + lint + typecheck + test
make demo        # a complete agent loop offline, with a scripted provider
```

---

## Part 2 — What to build

A **local web UI** for this runtime: a small FastAPI (or Starlette) server plus a
single-page front end. It must not become a second implementation of the agent —
it is a window onto the one that exists.

### 2.1 Stack

- **Backend**: FastAPI, `uvicorn`, bound to `127.0.0.1` by default. Add
  `fastapi`, `uvicorn[standard]`, `sse-starlette` (or use raw SSE) to a new
  `ui` extra in `pyproject.toml`.
- **Transport**: **Server-Sent Events** for the event stream (one-way, simple,
  reconnects for free) plus ordinary JSON POSTs for actions. Use a WebSocket only
  if you find a concrete need SSE cannot meet — say what it was.
- **Front end**: whatever you can make excellent and keep simple. A single HTML
  page with vanilla JS or a small React build is fine. **No build step is
  strongly preferred** — the value here is a local tool that starts instantly.
- **New module**: `src/agent/ui/` — `server.py`, `routes.py`, `events.py`,
  `static/`. Do not put UI code anywhere else.
- **New command**: `local-agent ui --host 127.0.0.1 --port 8765 [--open]`.

### 2.2 Pages

**1. Chat** — the primary screen.

- A message thread: user goals, assistant replies, tool calls and results.
- **Live progress**: as the agent works, show the current step, the phase
  (plan/observe/act/verify/replan), each tool call, and each verification.
  Render a tool call as a collapsible row: name, redacted arguments, outcome,
  duration.
- **Attachments**: drag-and-drop an image or audio file. It must be written into
  the workspace and loaded through `view_media`, **not** read from an arbitrary
  path — the workspace boundary is not optional for the UI either. Show a
  thumbnail for images and an audio player for audio.
- **Approvals appear inline and block** — see §2.4.
- Buttons: pause, cancel, and a clear indication of the outcome when it finishes,
  with the verified/unverified distinction visible.

**2. Connectors** — the screen this whole exercise is about.

- Table of configured connectors: name, kind, enabled toggle, target
  (command+args or URL), credential status (**present/missing, never the value**),
  tool count.
- **Add connector** form:
  - name (validated `^[a-z][a-z0-9_]{1,31}$` — show the rule, do not just reject)
  - kind: stdio or HTTP
  - stdio: command (bare name only — explain why), arguments
  - HTTP: URL (must be `http://` or `https://`)
  - environment variables to forward (**names only**, with a live "set / not set"
    indicator per name)
  - HTTP headers as `Header → ENV_VAR` pairs
  - optional tool allowlist
  - optional read-only tool list — with a **clear warning** that marking a remote
    tool read-only means it will run *without asking*, and that you are asserting
    it has no side effects
  - timeout
- **Test connection** button → `test_connectors(name)`, showing every tool the
  server offers, or the failure reason and which variables are unset.
- Enable/disable toggle, and remove with confirmation.
- Show each connector's tool prefix so the namespacing is visible.

**3. Tools** — every registered tool with its risk, category, approval
requirement, reversibility and JSON schema. Mark which came from a connector.
This is the "what can this thing actually do" page; make it genuinely readable.

**4. Tasks** — list with status filter; detail view showing the plan, the actions
taken, the **verification evidence**, failures, and the final outcome. Pause,
resume, cancel. This is where the honesty of the runtime becomes visible, so do
not flatten `unverified` into `completed`.

**5. Memory & Sessions** — approved facts (with delete), conversation history
(with view and delete), and a clearly-marked destructive "clear all data" action
that shows row counts and requires confirmation.

**6. Settings** — read the effective config via `safe_config()`. Show provider and
model, approval mode, workspace path, limits, shell allowlist, and media support
(which kinds the active model accepts, and whether video is enabled). Editing:
start with provider/model/approval-mode; anything you cannot safely change at
runtime, display read-only and say where it is configured.

### 2.3 API routes

```text
GET  /api/stats                     dashboard summary
GET  /api/config                    safe_config()
GET  /api/tools                     list_tools()
GET  /api/providers                 list_providers()
GET  /api/media                     media_support()
GET  /api/skills                    list_skills()

GET  /api/connectors                list_connectors()
POST /api/connectors                add_connector(...)
POST /api/connectors/{name}/enable  {"enabled": bool}
POST /api/connectors/{name}/test    test_connectors(name)
DEL  /api/connectors/{name}         remove_connector(name)

GET  /api/tasks                     list_tasks(status=, limit=)
GET  /api/tasks/{id}                get_task(id)
POST /api/tasks                     create and start a run
POST /api/tasks/{id}/pause          request pause
POST /api/tasks/{id}/cancel         request cancel
GET  /api/tasks/{id}/events         recorded events

GET  /api/sessions                  list_sessions()
GET  /api/sessions/{id}             messages
DEL  /api/sessions/{id}             delete
GET  /api/facts                     list_facts()
DEL  /api/facts/{id}                delete
POST /api/data/clear                clear-all, with confirmation

POST /api/uploads                   write an attachment into the workspace
GET  /api/stream                    SSE: the live event stream
POST /api/approvals/{id}            answer a pending approval
```

### 2.4 Approvals in a UI — the hard part, get this right

The console approver blocks on `input()`. A web UI cannot. Implement a
`WebApprover` satisfying the same `Approver` protocol:

1. `request()` is called from the runtime and **must block** until answered.
2. It registers a pending approval with a unique id and pushes an
   `approval_requested` event to the SSE stream, carrying the tool name, the
   **redacted** arguments, the risk level and category, the concrete target, and
   whether the action is reversible.
3. It waits on an `asyncio.Event` (or a queue) with a **timeout**.
4. `POST /api/approvals/{id}` with `approve_once` / `approve_for_run` / `deny` /
   `cancel` resolves it.
5. **On timeout, or if the browser disconnects, it denies.** Never approve by
   default, never approve on timeout, never "remember" an answer the user did not
   give. `approve_for_run` applies to that tool for that run only.

The runtime calls approvers from async context; `ConsoleApprover.request` is
synchronous. Look at how the registry calls it and decide whether to make the
protocol async or to bridge with a thread — either is fine, but say which you
chose and why, and keep `ConsoleApprover` working.

Write a test that proves a **disconnected browser results in a denial**, not a hang
and not an approval.

### 2.5 Multimodal in the UI

- Accept drag-and-drop and paste for images and audio.
- Write the upload into `workspace/files/` (or `workspace/temp/`) through the
  path policy, then attach it via `view_media`. Reject anything the content
  sniffer does not recognise, and say *why* — "this is not a supported media
  format, and renaming it will not help" is a better message than "invalid file".
- **Show the user what the model can actually perceive.** If the active model is
  text-only, the attach button must be disabled with an explanation, not silently
  accept a file that will be dropped.
- Respect and display the per-kind size limits before uploading.
- Video: if `media.enable_video` is off, say so where a user would try to attach
  one. Do not hide the control with no explanation.

---

## Part 3 — Testing

Add tests alongside the implementation. They must run with no network, no API
key, no MCP server and no browser.

- **Routes**: use `fastapi.testclient.TestClient`. Every route: success, the
  not-found case, and the invalid-input case.
- **No credential leaks**: assert that no response body contains a credential
  value, with a real one set in the environment. This is the single most
  important test in the UI layer.
- **Approvals**: approve, deny, approve-for-run (asked once, not twice), timeout
  denies, disconnect denies.
- **SSE**: events reach a subscriber, and are already redacted.
- **Uploads**: a valid image is accepted; a mislabelled file is refused; a path
  outside the workspace is refused; an oversized file is refused.
- **Connector CRUD** through the API, including that a stored definition never
  contains a credential value.
- Keep the existing 552 tests green.

---

## Part 4 — Documentation

Update, do not append:

- **`README.md`** — a "Web UI" section: install, run, what each page does, and the
  localhost-only default with the security reason.
- **`ARCHITECTURE.md`** — how the UI layers onto the management API, and the
  approval bridge.
- **`SECURITY.md`** — the UI's threat model: it is a local, unauthenticated
  surface bound to loopback. Say plainly what exposing it to a network would
  mean, and what would have to be added first (authentication, CSRF protection,
  TLS, per-user workspaces). Do not ship a `--host 0.0.0.0` flag without that
  paragraph.
- **`IMPLEMENTATION_PLAN.md`** — phases, decisions, test results, limitations.

---

## Part 5 — Definition of done

- `local-agent ui` starts a server on localhost and opens a working page.
- A goal typed in the browser runs the real agent, with live progress.
- An approval-gated action shows a blocking prompt in the browser; denying it
  stops the action; a disconnect denies rather than approving or hanging.
- An MCP server can be added, tested, enabled and used **entirely from the UI**,
  and its tools appear namespaced and approval-gated.
- An image can be dropped into chat and the model actually receives it; a
  text-only model disables the control with an explanation.
- No response body, log line or page ever contains a credential value.
- Tasks show plan, evidence, failures and the honest outcome.
- All tests pass offline; `ruff` and `mypy` are clean.
- Documentation is updated.

---

## Part 6 — What NOT to build

Do not, in this pass:

- expose the server beyond localhost without the security work described above;
- add authentication theatre (a hard-coded password is worse than none);
- let the UI execute shell commands, read files, or call a provider directly —
  everything goes through the management API and the runtime;
- auto-approve anything, for any reason, including "convenience";
- implement browser automation, sub-agents, background workers, or arbitrary code
  execution — those boundaries are deliberate and documented;
- reimplement the agent loop in JavaScript.

---

## Appendix — Sequences worth understanding before you start

**A goal, end to end**

```text
browser POST /api/tasks
  → AgentRunner.run(TaskState)
      → provider.generate(messages, tools)        [PROVIDER_REQUEST/RESPONSE]
      → registry.execute(tool_call)               [TOOL_REQUESTED]
          → validate arguments        (invalid_arguments)
          → check permissions         (permission_denied)
          → WebApprover.request()     [APPROVAL_REQUESTED] → SSE → browser
              ← POST /api/approvals/{id}
          → run under timeout         [TOOL_STARTED/TOOL_RESULT]
          → redact + truncate
      → verification                              [VERIFICATION_RESULT]
      → replan on failure                         [REPLAN]
  → outcome classified               [FINAL_ANSWER / TASK_FINISHED]
```

**Adding an MCP server from the UI**

```text
POST /api/connectors  { name, kind, command, args, env: ["GITHUB_TOKEN"] }
  → AgentManager.add_connector(...)  → validated → ConnectorStore (disabled)
POST /api/connectors/github/test
  → ConnectorManager.health_check("github")
      → MCPConnection.connect()      (only named env vars forwarded)
      → session.list_tools()         (allowlist applied)
  ← tools: ["mcp__github__search_repositories", ...]  or the failure reason
POST /api/connectors/github/enable  { enabled: true }
  → next run: AgentRunner.load_connector_tools()
      → each remote tool wrapped as MCPTool
      → description sanitised and attributed
      → registered approval-gated unless declared read-only
```

**Looking at an image**

```text
drag-and-drop → POST /api/uploads
  → PathPolicy.resolve()  → written to workspace/files/
model requests view_media { path }
  → content sniffed (magic number, not extension)
  → size checked against media limits
  → provider capability checked — refuse rather than silently drop
  → Attachment rides back on the ToolResult
  → runtime places it on the answering message
  → Gemini: inline data part · Ollama: base64 image (vision models only)
```
