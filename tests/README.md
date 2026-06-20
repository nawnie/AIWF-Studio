# Tests

The test suite is a maintainer safety net for a local ML app with optional heavy
engines. Most tests assert contracts and startup behavior without requiring real
model weights.

## What The Tests Assert

- core boot and settings models stay importable.
- UI registration does not require missing optional engines.
- model catalogs and downloads avoid unsafe paths and preserve expected metadata.
- engine tenant resolution builds the right worker commands.
- generation, ControlNet, Wan, video tools, training, and services keep their
  request-shaping contracts.
- helper scripts expose stable suite names for maintainers.

## What Tests Do Not Prove

- a local GPU has enough VRAM for a chosen model.
- a generated image, video, or audio clip is good.
- a speed optimization is real.
- an upstream optional engine checkout is installed correctly unless a focused
  probe or smoke test is also run.

Use runtime receipts for those claims.

## Running Suites

```powershell
python scripts/run_tests.py --list
python scripts/run_tests.py core
python scripts/run_tests.py engines wan
python scripts/run_tests.py --full
python scripts/run_tests.py --test test_wan.py --pytest-arg=-x
```

Prefer the smallest suite that covers the changed contract while working. Run the
full suite after broad dependency, boot, app wiring, or engine-supervisor changes.

## Adding Tests

- Put focused tests under `tests/individual_tests/`.
- Keep tests deterministic and small by using temp directories and fake model
  files where possible.
- Avoid real network, heavyweight GPU inference, and writes to tracked source
  directories.
- If a test intentionally covers a soft-fail path, assert the preserved fallback
  artifact or status message so future maintainers know the behavior is
  deliberate.
- Update `scripts/run_tests.py` when a new test belongs in a named maintainer
  suite.

Generated caches under `__pycache__` and `.pytest_cache` are ignored and should
not be reviewed as source changes.
