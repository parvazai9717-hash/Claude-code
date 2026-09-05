# Security

This document describes what `local-agent` defends against, what it does not, and
how each control is enforced.

The single most important property:

> **The system prompt is not a security boundary.** Every restriction described to
> the model is separately enforced in code. A model that ignores its instructions
> entirely still cannot leave the workspace, run an unapproved command, read a
> credential file, or report a success it did not achieve.

---

## Threat model

### Threats this design takes seriously

| Threat | Control |
|---|---|
| **Prompt injection** — content in a file or tool output instructs the model to exfiltrate data or run something destructive | The model can only *request*. Permissions, approvals, the workspace boundary and the shell allowlist are enforced by the runtime, which never reads instructions from model output. |
| **Credential exfiltration** | Credential files and directories are unreadable. Secrets are redacted before display, logging, persistence and model context. Subprocesses get a scrubbed environment. |
| **Path traversal / symlink escape** | Every path is resolved through `PathPolicy` before any I/O. Symlinks are not followed at all by default, and a resolved path outside the workspace is refused. |
| **Destructive commands** | Deny-by-default allowlist, plus a permanently-forbidden set that cannot be allowlisted at all. Refused by *policy*, before a human is ever asked. |
| **Shell injection** | Commands run as argv with no shell. Metacharacters are refused outright rather than silently doing something different from what the model intended. |
| **Runaway loops / cost** | Bounded steps, tool calls per step, total tool calls, retries per action and replans. |
| **False claims of success** | Consequential tools are marked `requires_verification`; the runtime classifies an outcome as `unverified` when evidence is missing, and says so in the summary. |
| **A model approving its own actions** | No tool reaches an approver. The approver is called only by the runtime, and its decision is final. |
| **Data left behind** | `clear-data` removes everything, after showing exactly what will be deleted. |

### Explicitly out of scope

- A **malicious local user**. Anyone who can edit `config.yaml` or run Python in the
  venv already has the user's privileges; this is not a sandbox against its owner.
- **Kernel or container isolation.** Subprocesses run as the same OS user. Use OS
  facilities (containers, `seccomp`, a dedicated user) if you need real isolation.
- **Model quality.** Bounds are enforced on *what the model may do*, never on
  whether its reasoning is correct.
- **Network egress from allowlisted programs.** `python3` and `git` are on the
  default allowlist and can reach the network. Remove them if that matters to you.

---

## Assumptions

1. The user owns the machine, the workspace and the data directory.
2. Credentials live in the environment or `.env`, never in `config.yaml` and never in
   the repository.
3. A human is present to answer approval prompts, unless `approval_mode` is
   `automatic` — in which case non-allowlisted actions are **denied**, not assumed.
4. `~/.local-agent` and the workspace are as sensitive as the data placed in them.
5. Tool output and file contents are **untrusted input**, and are treated as data.

---

## Secret handling

**Where credentials come from.** Only the environment (or a `.env` file loaded into
it). `GEMINI_API_KEY` is read once at provider construction and held privately.

**Where they never go.** Logs, SQLite, terminal output, the model's context, error
messages, subprocess environments, `config show`, and `doctor` — which reports only
whether a variable is *set*.

**Redaction** (`security/redaction.py`) runs on strings and nested structures:

- **By key** — any mapping key matching `api_key|secret|password|token|credential|
  authorization|auth|private_key|…` has its value replaced outright.
- **By value shape** — Google API keys, `sk-`/`rk-`/`pk-` keys, GitHub tokens, AWS
  access key IDs, Slack tokens, JWTs, PEM private-key blocks, `Authorization:`
  headers (the whole remainder of the line), `Bearer` tokens, and
  `NAME=value` assignments for credential-shaped names.
- **By literal** — the actual values of credential-shaped environment variables, so
  a key matching no known pattern is still removed.

`remember_fact` applies redaction on the way *in*: a proposed fact that changes
under redaction is refused rather than written to disk.

---

## Filesystem boundaries

The configured workspace is absolute. `PathPolicy.resolve` refuses:

- `..` traversal, in any form, before or after normalisation;
- absolute paths outside the workspace;
- `~`-relative paths, which would escape to the real home directory;
- paths containing a null byte;
- **symlinks** — not followed at all by default, and a resolved target outside the
  workspace is refused even when they are enabled;
- **credential files**: `.env*`, `.netrc`, `.pgpass`, `id_rsa`, `id_ed25519`,
  `credentials.json`, `client_secret.json`, `*.pem`, `*.key`, `*.p12`, `.npmrc`,
  `.git-credentials`, `secrets.*`, and more;
- **credential directories**: `.ssh`, `.gnupg`, `.aws`, `.azure`, `.kube`,
  `.docker`, `.gcloud`, browser profile directories, and `.local-agent` itself;
- hidden files, unless explicitly requested — and credential files stay refused
  even then;
- files above `max_file_bytes`, and binary files for text tools.

Writes are atomic (write to a temporary file in the same directory, then
`os.replace`) and keep a `.bak` of any file they replace or append to.

---

## Shell restrictions

Shell is the highest-risk capability, and is treated accordingly:

1. **Deny-by-default allowlist.** Only `shell_allowed_commands` may run. The default
   is read-only inspection commands.
2. **A permanently-forbidden set** that cannot be allowlisted whatever the config
   says: `sudo`, `su`, `rm`, `dd`, `mkfs`, `chmod`, `chown`, `mount`, `shutdown`,
   `curl`, `wget`, `nc`, `ssh`, `scp`, `rsync`, `pip`, `npm`, `apt`, `brew`,
   `docker`, `kubectl`, `systemctl`, `crontab`, `sh`, `bash`, `zsh`, `eval`, `exec`,
   and others.
3. **argv only.** Commands execute via `create_subprocess_exec` with **no shell**.
   Pipes, redirection, substitution and chaining are refused rather than silently
   ignored, because running something different from what was requested is worse
   than refusing.
4. **Bare command names only.** A path-qualified program (`/bin/ls`) is refused, so
   the allowlist cannot be sidestepped by pointing at a different binary.
5. **Scrubbed environment.** Only `PATH`, `LANG`, `LC_ALL`, `TERM` and `TZ` are
   passed through; `HOME` points at the workspace. No credential in the parent
   environment can reach a child.
6. **Workspace working directory**, validated through `PathPolicy`.
7. **Timeout with process-group cleanup.** The child runs in its own session;
   a timeout sends `SIGTERM` then `SIGKILL` to the whole group, so a spawned tree
   cannot survive.
8. **Output limits** on stdout and stderr, with truncation reported honestly.
9. **Double-checked.** The allowlist is verified by the permission layer *and* again
   immediately before spawning.

Shell can be removed entirely with `shell_enabled: false`.

---

## Approvals

| Mode | Behaviour |
|---|---|
| `always` | Every non-read-only action is confirmed. |
| `risky` *(default)* | Writes, shell, network, deletion, installs, auth, external effects and memory changes are confirmed. |
| `automatic` | Only `auto_approve_tools` run. Everything else is **denied** — there is no human to ask, so "unattended" never means "permitted". |

The prompt shows the tool, the **redacted** arguments, the risk level and category,
the concrete target, and whether the action is reversible. Choices: approve once,
approve for this run, deny, cancel the run.

Order matters: **permission first, approval second.** A forbidden action is refused
by policy and never reaches a human, so nobody can be talked into approving it.

`unsafe_disable_approvals` (and `--unsafe-no-approvals`) exist for tests only. They
are never a default, and the CLI prints a warning whenever they are active.

---

## Browser

Browser automation is **not enabled**. The interface in `agent/browser/` exists so a
future adapter has a shape to fit, and `UnavailableBrowser` refuses every action.

Before any adapter is enabled it must have: an isolated profile (never the user's
real browser profile, which holds session cookies and saved passwords); a domain
allowlist checked *before* navigation; a workspace download directory; upload
restrictions; navigation and action timeouts with process cleanup; redacted page
data; explicit approval for login, purchases, messages, submissions, account changes
and personal information; and human takeover for CAPTCHA, authentication, payment
and private data. Unrestricted debugging protocols against a live profile must never
be used — they hand over every credential the browser holds.

---

## Logging and observability

Structured JSONL at `~/.local-agent/events.jsonl`, plus an `events` table. Records
task start/end, status transitions, session, provider and model, lifecycle phase,
step number, tool requests, approval results, tool result summaries, verification
results, durations, safe error categories, and the final status.

**Never logged**: secrets, private file contents, authorization headers, or raw
environment variables. Redaction is applied once, centrally, on the event bus.
Verbose tool logging is opt-in and still redacted.

---

## Data deletion

```bash
local-agent memory list && local-agent memory remove ID   # one fact
local-agent sessions delete ID                            # one conversation
local-agent clear-data                                    # everything
```

`clear-data` prints per-table row counts and requires confirmation before deleting.
It does not touch workspace files — those are yours to remove.

---

## Safe deployment

1. **Keep the default approval mode.** `risky` is the right trade-off for a human at
   a terminal.
2. **Do not widen the shell allowlist casually.** Adding `python3` already allows
   arbitrary computation and network access; adding a package manager allows
   arbitrary installation.
3. **Use a dedicated workspace** — never point it at `$HOME`, a repository with
   credentials, or a mounted network share.
4. **Never commit `.env` or `config.yaml`.** Both are git-ignored.
5. **Review `~/.local-agent/events.jsonl`** periodically to see what actually ran.
6. **Run as a low-privilege user**, ideally in a container, if the agent will act on
   anything you would not hand to an untrusted script.
7. **Treat `automatic` mode with suspicion.** It exists for narrowly-scoped,
   allowlisted, read-mostly work.
8. **Prefer a local model** when the goal involves data you do not want to leave the
   machine. With `provider: ollama` nothing is sent to a third party.

---

## Reporting a vulnerability

Open an issue describing the impact and how to reproduce it. Please do not include
real credentials in the report.
