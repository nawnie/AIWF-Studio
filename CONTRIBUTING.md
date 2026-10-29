# Contributing

Thanks for helping improve AIWF Studio. The current Windows application is a
preview; please check the requirements and limitations in the README before
reporting a problem or proposing a change.

## Report a problem

Use the bug-report form and include the Windows version, Studio build, steps to
reproduce, and what happened. For image or video issues, include the GPU and
driver, backend, model family, and whether the model is a single file or a
folder. Remove API keys, personal paths, prompts, and other private information
from logs or screenshots before sharing them.

## Develop and test

See [BUILDING.md](BUILDING.md) for Windows build inputs and focused tests. Keep
the native shell and API compatibility metadata in sync, and include the tests
or manual checks relevant to your change. State what you could not test,
especially physical hardware, inference, and installation behavior.

Do not add model weights, user outputs, credentials, generated installers, or
local dependency caches to a pull request. Third-party components retain their
own licenses; update notices and provenance when redistribution inputs change.
