# Windows Python 3.12 backend lock

This optional runtime snapshot targets Windows AMD64 and CPython
`>=3.12.13,<3.13`. It was resolved against the 36 ranges in the root
`requirements.txt` and the tested 187-package Windows freeze; it keeps the
root requirements file and other installation paths unchanged.

The lock maps only `torch` and `torchvision` to the explicit official PyTorch
CUDA 13.0 index. Other packages use PyPI. Do not replace this with a global
PyTorch extra index.

`filterpy==1.4.5` has no wheel for this environment. Its sdist SHA-256 is in
`uv.lock`; `[tool.uv].build-constraint-dependencies` pins its isolated build
inputs to `setuptools==78.1.0` and `wheel==0.48.0`. The locked install therefore
builds FilterPy from the verified sdist with the recorded builder versions.
This does not claim bit-for-bit wheel output across Python/toolchain setups.

The supported installer entry point is:

```powershell
.\scripts\install_aiwf_studio.ps1 -Mode express -UseBackendLock
```

For a disposable environment, set uv's project environment to the app venv and
run:

```powershell
$env:UV_PROJECT_ENVIRONMENT = (Resolve-Path .\venv).Path
uv sync --project .\dependencies\windows-py312 --locked --inexact --python .\venv\Scripts\python.exe
```

The installer validates OS, architecture, implementation, and Python version
before syncing. `--inexact` enforces locked packages without removing unrelated
packages already installed in the app venv.
