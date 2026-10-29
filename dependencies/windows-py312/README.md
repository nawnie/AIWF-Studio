# Windows Python 3.12 backend lock

This optional native-inference runtime targets Windows AMD64 and CPython
`>=3.12.13,<3.13`. It has its own locked dependency graph. The legacy Gradio
web interface is outside this runtime's scope: its pinned Gradio version
requires Pillow below 13, while the native API uses the security-patched
Pillow 13 series. Use a separate compatible environment for legacy web work.

The lock maps only `torch` and `torchvision` to the explicit official PyTorch
CUDA 13.2 index. Other packages use PyPI. Do not replace this with a global
PyTorch extra index.

`filterpy==1.4.5` has no wheel for this environment. Its sdist SHA-256 is in
`uv.lock`; `[tool.uv].build-constraint-dependencies` pins its isolated build
inputs to `setuptools==84.2.1` and `wheel==0.48.0`. The locked install therefore
builds FilterPy from the verified sdist with the recorded builder versions.
This does not claim bit-for-bit wheel output across Python/toolchain setups.

For the AES Studio preview, close the app and use the optional backend helper
from the installation root. It requires `uv` and can download several gigabytes:

```powershell
.\scripts\provision_python_backend.ps1 -StudioRoot .
```

For a disposable environment, set uv's project environment to the app venv and
run:

```powershell
$env:UV_PROJECT_ENVIRONMENT = (Resolve-Path .\venv).Path
uv sync --project .\dependencies\windows-py312 --locked --inexact --python .\venv\Scripts\python.exe
```

The helper validates the target Python and checks for processes using the venv
before syncing. `--inexact` enforces locked packages without removing unrelated
packages already installed in the app venv. An existing environment containing
the legacy Gradio stack can therefore retain incompatible packages; use a fresh
AES installation for this lock rather than removing packages from a shared venv.
The API package constraints match the preview's offline API requirements. The
resolved lock has been checked, but the complete optional inference install and
a real inference run still require validation on a clean machine.

The optional lock still includes packages with reported security advisories,
including Transformers and Jupyter components. The current Transformers
compatibility range excludes versions containing some fixes.
Review and resolve these advisories before treating the backend as production-ready.
