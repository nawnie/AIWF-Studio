# Scripts

Scripts are maintainer entry points for repeatable setup and validation. They
should be boring, explicit, and safe to run from PowerShell on Windows.

## Available Helpers

- `bootstrap_engine.ps1`: creates `engines/<name>/.venv`, installs CUDA torch if
  needed, then installs the engine requirements.
- `bootstrap_mmaudio.ps1`: sets up the optional MMAudio audio engine without
  installing it into the shared Studio venv.
- `bootstrap_ltx.ps1`: clones Lightricks LTX-2, creates `engines/ltx/.venv`,
  installs the LTX 2.3 runtime stack, and can enable the worker in `engines.json`.
- `verify_engine.ps1`: probes an engine worker with a small JSON request.
- `run_tests.py`: groups pytest files into practical suites so maintainers do
  not need to remember every test filename.
- `install_aiwf_studio.ps1 -UseBackendLock`: opt into the pinned Windows AMD64
  CPython 3.12 backend lock. It requires Python `>=3.12.13,<3.13`, uses uv's
  per-package PyTorch CUDA 13.0 index mapping, and retains extra packages in an
  existing app venv. The normal installer remains unchanged on other paths.

## Script Conventions

- Resolve paths from `$PSScriptRoot` or the repo root. Do not depend on the
  caller's current directory.
- Fail fast with clear errors when an engine folder, Python executable, or worker
  script is missing.
- Keep local secrets, tokens, model roots, and machine-specific paths in env vars
  or ignored config files.
- Do not mutate generated outputs unless the script's purpose is explicitly a
  probe or smoke artifact.
- Prefer small composable scripts over one large all-in-one bootstrap.

## Common Commands

```powershell
python scripts/run_tests.py --list
python scripts/run_tests.py core engines
python scripts/run_tests.py --test test_launch.py --pytest-arg=-x
.\scripts\bootstrap_engine.ps1 -Name wan
.\scripts\bootstrap_ltx.ps1 -Enable
.\scripts\verify_engine.ps1 -Name wan
.\scripts\install_aiwf_studio.ps1 -Mode express -UseBackendLock
```

The locked installer option is Windows AMD64 only. It stops with a clear error
for an unsupported host or Python version rather than falling back to a
different lock. The dedicated lock and FilterPy build constraints are under
`dependencies/windows-py312/`.

When a script installs heavy ML packages, note the expected Python/CUDA target in
the script or engine README. Dependency changes should also be reflected in
`docs/DEPENDENCY_POLICY.md`.
