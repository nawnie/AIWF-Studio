# Scripts

Scripts are maintainer entry points for repeatable setup and validation. They
should be boring, explicit, and safe to run from PowerShell on Windows.

## Available Helpers

- `bootstrap_engine.ps1`: creates `engines/<name>/.venv`, installs CUDA torch if
  needed, then installs the engine requirements.
- `bootstrap_mmaudio.ps1`: sets up the optional MMAudio audio engine without
  installing it into the shared Studio venv.
- `verify_engine.ps1`: probes an engine worker with a small JSON request.
- `run_tests.py`: groups pytest files into practical suites so maintainers do
  not need to remember every test filename.

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
.\scripts\verify_engine.ps1 -Name wan
```

When a script installs heavy ML packages, note the expected Python/CUDA target in
the script or engine README. Dependency changes should also be reflected in
`docs/DEPENDENCY_POLICY.md`.
