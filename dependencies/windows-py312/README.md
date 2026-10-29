# Windows Python 3.12 backend lock

This optional runtime snapshot targets Windows AMD64 and CPython
`>=3.12.13,<3.13`. It was resolved against the 36 ranges in the root
`requirements.txt` and the tested 187-package Windows freeze; it keeps the
root requirements file and other installation paths unchanged.

The lock maps only `torch` and `torchvision` to the explicit official PyTorch
CUDA 13.2 index. Other packages use PyPI. Do not replace this with a global
PyTorch extra index.

`filterpy==1.4.5` has no wheel for this environment. Its sdist SHA-256 is in
`uv.lock`; `[tool.uv].build-constraint-dependencies` pins its isolated build
inputs to `setuptools==78.1.0` and `wheel==0.48.0`. The locked install therefore
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
packages already installed in the app venv. The API package constraints have been
aligned with the preview's offline API requirements. The resolved lock has been
checked, but the complete optional inference install and a real inference run
still require validation on a clean machine.
