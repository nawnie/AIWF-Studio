# Pre-Change Snapshot

Created: `2026-06-13 20:52:04`

Purpose: keep a simple before-state in the project folder before a large change, using file dates and sizes instead of Git history.

Files created:

- `prechange-file-manifest-2026-06-13_20-52-04.csv`
- `prechange-root-manifest-2026-06-13_20-52-04.csv`

What is captured:

- `230` source and support files under `aiwf/`, `tests/`, `static/`, `docs/`, `plugins/`, `scripts/`, `workflows/`, `prompts/`, and `wildcards/`
- `15` root project files such as `AGENTS.md`, `README.md`, `launch.py`, `requirements.txt`, `config.json`, and `plan.md`
- Each row includes relative path, byte size, and last write time

What is excluded:

- `venv/`
- `models/`
- `outputs/`
- `datasets/`
- `assets/`
- `engines/`
- Python cache files such as `__pycache__/` and `*.pyc`

Quick compare idea for later:

```powershell
$before = Import-Csv .\prechange-file-manifest-2026-06-13_20-52-04.csv
$now = Get-ChildItem -Recurse -File aiwf,tests,static,docs,plugins,scripts,workflows,prompts,wildcards |
  Where-Object { $_.FullName -notmatch '\\__pycache__\\' -and $_.Extension -ne '.pyc' } |
  ForEach-Object {
    [PSCustomObject]@{
      Path = $_.FullName.Substring($PWD.Path.Length + 1)
      Bytes = $_.Length
      LastWriteTime = $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
    }
  }
Compare-Object $before $now -Property Path,Bytes,LastWriteTime
```
