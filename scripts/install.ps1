<#
.SYNOPSIS
    Installs local-agent on Windows.

.DESCRIPTION
    Checks prerequisites, creates a virtual environment, installs the project,
    and verifies the install by running an offline agent loop. Nothing here needs
    an API key or a network connection to a model.

    Read this file before running it. It is deliberately short and does nothing
    surprising: no registry writes, no PATH changes, no admin rights, no
    downloads outside pip and git.

.PARAMETER Path
    Where to install. Defaults to the current directory.

.PARAMETER Branch
    Which branch to check out. Defaults to the development branch.

.PARAMETER SkipTests
    Skip the test suite, which takes about ten seconds.

.EXAMPLE
    .\scripts\install.ps1

.EXAMPLE
    .\scripts\install.ps1 -Path C:\Tools -SkipTests
#>

[CmdletBinding()]
param(
    [string] $Path = ".",
    [string] $Branch = "claude/new-session-orxpij",
    [switch] $SkipTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
# Seeded because StrictMode treats an unset variable as an error, and this is
# read after native commands that may not have run yet.
$global:LASTEXITCODE = 0

$RepoUrl = "https://github.com/parvazai9717-hash/Claude-code.git"
$MinPython = [Version] "3.11"

function Write-Step  { param([string] $Text) Write-Host "`n==> $Text" -ForegroundColor Cyan }
function Write-Good  { param([string] $Text) Write-Host "    OK  $Text" -ForegroundColor Green }
function Write-Warn2 { param([string] $Text) Write-Host "    !   $Text" -ForegroundColor Yellow }
function Write-Bad   { param([string] $Text) Write-Host "    X   $Text" -ForegroundColor Red }

function Stop-WithHelp {
    param([string] $Problem, [string] $Fix)
    Write-Bad $Problem
    Write-Host ""
    Write-Host "How to fix it:" -ForegroundColor Yellow
    Write-Host "  $Fix"
    Write-Host ""
    exit 1
}

# ---------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------
Write-Step "Checking prerequisites"

# Windows ships a `py` launcher that can select a version; prefer it, then fall
# back to whatever `python` resolves to.
$PythonExe = $null
foreach ($candidate in @(
    @{ Exe = "py";      Args = @("-3") },
    @{ Exe = "python";  Args = @() },
    @{ Exe = "python3"; Args = @() }
)) {
    $found = Get-Command $candidate.Exe -ErrorAction SilentlyContinue
    if (-not $found) { continue }

    # Build the argument list as a variable: `& $exe @(...)` is ambiguous
    # between splatting and an array subexpression, and reads badly either way.
    $probeArgs = @($candidate.Args) + @("-c", "import sys; print('%d.%d' % sys.version_info[:2])")
    $reported = $null
    try {
        $reported = & $candidate.Exe $probeArgs 2>$null
    } catch {
        $reported = $null
    }
    if (-not $reported -or $LASTEXITCODE -ne 0) { continue }

    $version = [Version] (($reported | Select-Object -First 1).ToString().Trim())
    if ($version -ge $MinPython) {
        $PythonExe  = $candidate.Exe
        $PythonArgs = $candidate.Args
        Write-Good "Python $version  (via '$($candidate.Exe)')"
        break
    }
    Write-Warn2 "'$($candidate.Exe)' is Python $version, which is too old"
}

if (-not $PythonExe) {
    Stop-WithHelp `
        "Python 3.11 or newer was not found." `
        "Install it from https://www.python.org/downloads/ and TICK 'Add python.exe to PATH' in the installer. Then close this window, open a new one, and run this script again."
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Stop-WithHelp `
        "git was not found." `
        "Install it from https://git-scm.com/download/win , then open a new window and run this script again."
}
Write-Good "git $((& git --version) -replace 'git version ', '')"

# ---------------------------------------------------------------------------
# 2. Get the code
# ---------------------------------------------------------------------------
Write-Step "Fetching the code"

$Root = Resolve-Path -LiteralPath $Path
$Target = Join-Path $Root "local-agent"

if (Test-Path -LiteralPath (Join-Path $Target ".git")) {
    Write-Good "Already cloned at $Target - updating"
    Push-Location $Target
    try {
        & git fetch origin $Branch
        if ($LASTEXITCODE -ne 0) { Stop-WithHelp "Could not fetch from GitHub." "Check your internet connection and try again." }
        & git checkout $Branch
        & git pull origin $Branch
    } finally { Pop-Location }
} else {
    & git clone --branch $Branch $RepoUrl $Target
    if ($LASTEXITCODE -ne 0) {
        Stop-WithHelp `
            "Could not clone the repository." `
            "Check your internet connection. If the repository is private, sign in first with: git credential-manager github login"
    }
    Write-Good "Cloned to $Target"
}

Set-Location -LiteralPath $Target

# ---------------------------------------------------------------------------
# 3. Virtual environment
# ---------------------------------------------------------------------------
Write-Step "Creating the virtual environment"

$VenvPython = Join-Path $Target ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $VenvPython) {
    Write-Good "Reusing the existing .venv"
} else {
    $venvArgs = @($PythonArgs) + @("-m", "venv", ".venv")
    & $PythonExe $venvArgs
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
        Stop-WithHelp `
            "Could not create the virtual environment." `
            "If you installed Python from the Microsoft Store, install it from python.org instead - the Store build restricts where it can write."
    }
    Write-Good "Created .venv"
}

# ---------------------------------------------------------------------------
# 4. Install
# ---------------------------------------------------------------------------
Write-Step "Installing dependencies (this takes a minute)"

& $VenvPython -m pip install --upgrade pip --quiet
& $VenvPython -m pip install -e ".[dev,gemini,mcp]"
if ($LASTEXITCODE -ne 0) {
    Write-Warn2 "The install failed. Retrying once - a partial download is the usual cause."
    & $VenvPython -m pip install -e ".[dev,gemini,mcp]"
    if ($LASTEXITCODE -ne 0) {
        Stop-WithHelp `
            "Installation failed twice." `
            "Scroll up for the pip error. If it mentions a proxy or SSL, you may be behind a corporate firewall; ask your IT team for the pip index settings."
    }
}
Write-Good "Dependencies installed"

# ---------------------------------------------------------------------------
# 5. Verify - all offline, no API key needed
# ---------------------------------------------------------------------------
Write-Step "Verifying the install"

$AgentExe = Join-Path $Target ".venv\Scripts\local-agent.exe"
if (-not (Test-Path -LiteralPath $AgentExe)) {
    Stop-WithHelp `
        "The install finished but local-agent.exe is missing." `
        "Run this again - the package install was probably interrupted part-way."
}
Write-Good "$((& $AgentExe --version))"

& $AgentExe -p mock doctor | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Warn2 "doctor reported a problem - run '.\.venv\Scripts\local-agent.exe -p mock doctor' to see it" }
else { Write-Good "doctor passed" }

$demo = & $VenvPython scripts\demo_offline.py 2>&1 | Select-Object -Last 1
if ($demo -match "DEMO PASSED") { Write-Good "Offline agent loop ran end to end" }
else { Write-Warn2 "The offline demo did not pass. Run: .\.venv\Scripts\python.exe scripts\demo_offline.py" }

if (-not $SkipTests) {
    Write-Step "Running the test suite (about ten seconds)"
    & $VenvPython -m pytest -q
    if ($LASTEXITCODE -eq 0) { Write-Good "All tests passed" }
    else { Write-Warn2 "Some tests failed. The agent will probably still work; scroll up for detail." }
}

# ---------------------------------------------------------------------------
# 6. What next
# ---------------------------------------------------------------------------
$activate = Join-Path $Target ".venv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "-----------------------------------------------------------" -ForegroundColor Green
Write-Host " local-agent is installed at $Target" -ForegroundColor Green
Write-Host "-----------------------------------------------------------" -ForegroundColor Green
Write-Host ""
Write-Host "Start using it:" -ForegroundColor Cyan
Write-Host "  cd `"$Target`""
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  local-agent -p mock doctor        # works with no API key"
Write-Host ""
Write-Host "To connect a real model, pick ONE:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Ollama - free, runs locally, nothing leaves your PC:"
Write-Host "    1. Install from https://ollama.com/download"
Write-Host "    2. ollama pull llama3.1"
Write-Host "    3. copy config.example.yaml config.yaml"
Write-Host "       then set:  provider: ollama"
Write-Host ""
Write-Host "  Gemini - cloud, quicker to set up:"
Write-Host "    1. Get a key at https://aistudio.google.com/apikey"
Write-Host "    2. copy .env.example .env"
Write-Host "       then put your key in .env as GEMINI_API_KEY=..."
Write-Host "    3. copy config.example.yaml config.yaml"
Write-Host "       then set:  provider: gemini"
Write-Host ""
Write-Host "  Then:  local-agent doctor  and  local-agent chat"
Write-Host ""
Write-Host "A note on Windows:" -ForegroundColor Yellow
Write-Host "  The default shell allowlist assumes Unix tools (ls, cat, grep)."
Write-Host "  Edit shell_allowed_commands in config.yaml to name programs you"
Write-Host "  have, or set shell_enabled: false to turn the shell tool off."
Write-Host ""
Write-Host "Full guide: INSTALL.md    Security notes: SECURITY.md"
Write-Host ""
