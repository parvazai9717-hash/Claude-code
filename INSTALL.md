# Installing `local-agent` on your own machine

Works on **macOS**, **Linux** and **Windows**. Takes about five minutes.

---

## Windows: the one-command version

If you are on Windows and just want it installed, open **PowerShell** in the
folder where you want it and run:

```powershell
git clone -b claude/new-session-orxpij https://github.com/parvazai9717-hash/Claude-code.git local-agent
cd local-agent
.\scripts\install.ps1
```

The script checks your Python and git, creates the virtual environment, installs
everything, and verifies it by running a complete agent loop offline. It needs no
admin rights, changes no system settings, and touches nothing outside this folder.
**Read `scripts/install.ps1` first if you like** — it is short and deliberately
does nothing surprising.

If PowerShell refuses to run it:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then run the script again. Everything below is the same thing done by hand, and
covers macOS and Linux too.

---

## 1. Check you have Python 3.11 or newer

```bash
python3 --version      # macOS / Linux
python --version       # Windows
```

If it is older than 3.11, or missing:

| OS | How to get it |
|---|---|
| **Windows** | <https://www.python.org/downloads/> — tick **"Add python.exe to PATH"** in the installer |
| **macOS** | `brew install python@3.12`, or the installer from python.org |
| **Linux (Debian/Ubuntu)** | `sudo apt install python3.11 python3.11-venv git` |
| **Linux (Fedora)** | `sudo dnf install python3.11 git` |

You also need **git**: `git --version`.

---

## 2. Get the code

The work lives on the branch `claude/new-session-orxpij` (it has not been merged
to the default branch yet).

```bash
git clone https://github.com/parvazai9717-hash/Claude-code.git local-agent
cd local-agent
git checkout claude/new-session-orxpij
```

---

## 3. Create a virtual environment and install

A virtual environment keeps this project's dependencies out of your system Python.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,gemini,mcp]"
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,gemini,mcp]"
```

> If PowerShell blocks the activate script, run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

**Windows (Command Prompt)**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e ".[dev,gemini,mcp]"
```

What the extras mean — drop any you do not want:

| Extra | Gives you |
|---|---|
| `gemini` | The Google Gen AI SDK, for `provider: gemini` |
| `mcp` | The MCP SDK, for connectors |
| `dev` | pytest, ruff, mypy — only needed to run the tests |

There is also a shortcut on macOS/Linux: `make install`.

---

## 4. Check it works — no API key needed

```bash
local-agent --version
local-agent -p mock doctor
python scripts/demo_offline.py
```

The last one runs a **complete agent loop offline** with a scripted model: it
reads a file, gets a path wrong on purpose, replans, writes a report, and
verifies the write. It should end with `DEMO PASSED`.

Run the test suite too, if you like — it needs no network, no API key and no
model server:

```bash
pytest          # 562 tests
```

---

## 5. Point it at a real model

You need **one** of these. Ollama is free and private; Gemini is faster to set up.

### Option A — Ollama (runs locally, nothing leaves your machine)

1. Install from <https://ollama.com/download>.
2. Start it and pull a **tool-capable** model — tool calling is required:

   ```bash
   ollama serve            # leave running (on Windows it starts automatically)
   ollama pull llama3.1
   ```

3. Configure:

   ```bash
   cp config.example.yaml config.yaml
   ```

   Edit `config.yaml`:

   ```yaml
   provider: ollama
   ollama_model: llama3.1
   ollama_base_url: http://localhost:11434
   ```

4. Check and run:

   ```bash
   local-agent doctor
   local-agent chat
   ```

`doctor` tells you whether the server is reachable, whether the model is
installed, and whether it supports tool calling.

**For image understanding on Ollama** you need a vision model — `ollama pull
llava` or `ollama pull llama3.2-vision`. `doctor` reports what the active model
can perceive.

### Option B — Gemini (cloud)

1. Get a key at <https://aistudio.google.com/apikey>.
2. Put it in a `.env` file in the project root — **never** in `config.yaml`:

   ```bash
   cp .env.example .env
   ```

   Edit `.env`:

   ```
   GEMINI_API_KEY=your-key-here
   ```

3. Configure `config.yaml`:

   ```yaml
   provider: gemini
   gemini_model: gemini-2.5-flash
   ```

4. Check and run:

   ```bash
   local-agent doctor
   local-agent chat
   ```

Gemini handles images, audio and (if you enable it) video.

`.env` is git-ignored. Do not commit it.

---

## 6. First run

```bash
local-agent chat
```

Then type a goal, for example:

```text
list the files in the workspace and summarise what you find
```

Things to know on your first run:

- **The workspace is the boundary.** It is `./workspace` by default. The agent
  cannot read or write outside it. Put files you want it to work on in
  `workspace/files/`.
- **You will be asked to approve** writes, shell commands and memory saves. The
  prompt shows the tool, the redacted arguments, the risk, the target, and
  whether it can be undone. `y` = once, `a` = for this run, `n` = deny,
  `c` = cancel.
- **Ctrl-C** cancels cleanly — it stops before the next consequential action
  rather than killing the process mid-write. **Ctrl-D** exits.
- Useful in-chat commands: `/help`, `/status`, `/tools`, `/connectors`,
  `/memory`, `/tasks`, `/quit`.

---

## 7. Optional: add an MCP connector

```bash
local-agent connectors add filesystem \
  --command npx --arg -y --arg @modelcontextprotocol/server-filesystem
local-agent connectors test filesystem      # see what it offers first
local-agent connectors enable filesystem
```

Requires Node.js for `npx`-based servers: <https://nodejs.org>.

Connectors start **disabled**, their tools are namespaced
`mcp__<connector>__<tool>`, and they require approval unless you explicitly mark
a tool read-only. Read the "Connectors and MCP servers" section of
[`SECURITY.md`](SECURITY.md) before enabling one — a connector is someone else's
code running with your agent's tools.

---

## Platform notes

Everything works on all three platforms, with one honest caveat.

**Windows.** The shell tool works, but the *timeout guarantee is weaker*. On
macOS and Linux a timed-out command is killed as a whole process group, so
anything it spawned dies with it. Windows has no `killpg`; the child is
terminated directly and the OS usually cleans up descendants, but not always. If
that matters to you, set `shell_enabled: false` in `config.yaml`.

The default shell allowlist also assumes Unix tools (`ls`, `cat`, `grep`). On
Windows, either use Git Bash / WSL, or edit `shell_allowed_commands` in
`config.yaml` to name the programs you actually have.

**All platforms.** The shell allowlist is deny-by-default, and some commands
(`sudo`, `rm`, `curl`, `bash`, `pip`, …) can never be allowlisted at all.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `local-agent: command not found` | The virtual environment is not active. Re-run the activate line from step 3. |
| Install seemed to finish, but `local-agent` and `yaml` are missing | The download was interrupted part-way. Just run the `pip install` line again — it is safe to repeat. |
| `python: command not found` (macOS/Linux) | Use `python3`. |
| PowerShell refuses to activate | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry. |
| `GEMINI_API_KEY is not set` | Put it in `.env`, or export it in your shell. `doctor` confirms presence without printing it. |
| `cannot reach the Ollama server` | Run `ollama serve`; check `ollama_base_url`. |
| `the model X is not installed` | `ollama pull X`. |
| `does not support tool calling` | Use a tool-capable model — llama3.1, qwen2.5, mistral-nemo. |
| `path escapes the workspace boundary` | Working as designed. Use a workspace-relative path. |
| `'curl' is not in the shell allowlist` | Also as designed. Add it to `shell_allowed_commands` only if you are sure. |
| A task is stuck in `running` | Its process died. `local-agent task recover`, then `task resume`. |

---

## Where things live

| Path | What |
|---|---|
| `./workspace/` | The agent's files. Its hard boundary. |
| `~/.local-agent/agent.sqlite3` | Sessions, tasks, approved memory, events |
| `~/.local-agent/events.jsonl` | Structured, redacted activity log |
| `~/.local-agent/connectors.json` | Connector definitions (**names** of env vars, never values) |
| `./config.yaml` | Your settings (git-ignored) |
| `./.env` | Your credentials (git-ignored) |

Delete everything the agent has stored with `local-agent clear-data`. It shows
row counts and asks first, and does not touch workspace files.

---

## Updating later

```bash
cd local-agent
git pull origin claude/new-session-orxpij
pip install -e ".[dev,gemini,mcp]"
```
