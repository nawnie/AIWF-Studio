# Wan Video Engine

Optional isolated worker environment for Wan image-to-video generation.

The core AIWF Studio app must boot without this venv. When enabled in
`engines.json`, the main app launches this worker through `ProcessSupervisor`
using the engine's own Python executable.

Initial worker mode is `probe`; full generation dispatch is intentionally a
separate migration step after the tenant contract is verified.
