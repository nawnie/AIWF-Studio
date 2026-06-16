# Project Evidence Checklist

Fill this before applying.

## Repository Links

- Public AIWF Studio repo: https://github.com/nawnie/AIWF-Studio
- Local branch/status: `v2-roadmap` in `C:\Users\Shawn\Desktop\AIWF-Studio`
- Related MoK repo: TODO

## Quantitative Evidence

| Evidence | Current Value | Proof Needed |
| --- | --- | --- |
| GitHub main test collection | 289 tests | Command output or CI badge |
| Local test collection | 715 tests | Command output after full local cleanup |
| Local test files | 87 | `Get-ChildItem tests -Filter 'test_*.py'` |
| Public test files | 56 | GitHub main |
| Relevant local files | 319 | Excludes vendored ED2/runtime folders |
| Relevant public files | 205 | GitHub main |

## Demo Evidence Needed

- Studio txt2img/img2img/inpaint screenshot.
- Wan video tab screenshot and short output clip.
- Training tab screenshot showing optional engine status and validation.
- Chat tab screenshot.
- Model Manager screenshot with downloads/model info.
- Architecture diagram showing UI -> services -> infrastructure/workers.
- 2 minute walkthrough video.

## Technical Evidence To Highlight

- `AppContext` composition root instead of global shared state.
- `GenerationRequest` and other typed domain models.
- `ProcessSupervisor` for long-running engine work.
- Worker protocol docs.
- `EngineSupervisor` / GPU tenant coordination.
- Optional-engine import discipline.
- Tests for dataset validation, training config builders, process supervision, Ollama client, model lookup, Wan, samplers, quantization, and video export.

## Gaps To Be Honest About

- Public README lags local functionality.
- Local changes need to be cleaned, staged, and pushed.
- Some acceleration paths are experimental and flag-gated.
- Optional engines must remain optional at boot.
- Performance claims require logged benchmarks.

