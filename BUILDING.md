# Building the Windows preview

The repository contains source and dependency manifests, not model weights,
prebuilt application payloads, the CUDA engine binaries, or Python wheels.
Building an installer requires Windows x64, enough free disk space, and the
separate inputs below. The application is a preview; a successful build does
not establish clean-PC or model-generation compatibility.

## Build inputs

- Visual Studio 2022 or Build Tools with the C++ desktop workload and MSVC
  v143. The native project restores its locked Windows App SDK and SDK packages
  using NuGet. CPython 3.12.13 or newer below 3.13 is needed for build metadata.
- NSIS 3.13 and its standard plugins. The credential bridge must be **x86**
  because the NSIS installer process is 32-bit; the uninstall worker and the
  rest of the payload are x64.
- The unpacked Windows x86-64 CPython 3.12.13 install-only runtime with `venv`
  and `ensurepip`. Its original source and SHA-256 are recorded in
  `licenses/AES-PACKAGE-PROVENANCE.json`.
- The pinned API wheels listed in `packaging/wheel-provenance.json`. The stage
  script checks every wheel's filename and SHA-256 against the manifest.
- A compatible CUDA 13 stable-diffusion.cpp engine and matching cuBLAS DLLs in
  `dependencies/stable-diffusion.cpp-cuda13/`; the exact preview build's source,
  flags, checksums, and third-party notices are recorded in that directory.
- Microsoft Visual C++ x64 app-local redistributable DLLs and their `Redist.txt`
  in `packaging/vc-runtime/`. Consult Microsoft's redistribution terms. The
  staging script checks the files required by the preview build.

Acquiring and redistributing third-party components is subject to their own
licenses. Follow the provenance and [third-party notices](THIRD-PARTY-NOTICES.md).

## Build

From a checkout root, build the native shell and its matching API metadata:

```powershell
.\native\build.ps1 -Configuration Release
```

To avoid network access during restore, pass `-Offline -RestoreSource` with a
local NuGet feed directory. This also disables NuGet audit for that offline
restore. All locked packages must already be present in that feed or the NuGet
package cache; an incomplete offline restore will fail. Run dependency auditing
separately when network access is permitted.

Build the helpers using Visual Studio developer command prompts from the same
checkout root. First use the **x86** prompt for the NSIS credential bridge:

```bat
cl /nologo /O2 /MT /LD packaging\credential_bridge.c /Fopackaging\credential_bridge.obj /Fepackaging\AesCredentialBridge.dll /link /DEF:packaging\credential_bridge.def /IMPLIB:packaging\AesCredentialBridge.lib Advapi32.lib
```

Then use the **x64** prompt for the uninstall worker:

```bat
cl /nologo /std:c++17 /EHsc /O2 /MT packaging\uninstall_worker.cpp /Fopackaging\uninstall_worker.obj /Fepackaging\uninstall_worker.exe /link bcrypt.lib
```

Prepare a new stage using an unpacked Python runtime and the inputs above.
Paths below are examples; use paths on your own computer. The stage directory
must not exist. Staging performs no downloads and verifies the API/native
binding, locked wheels, required files, and helper architectures.

```powershell
py -3.12 -B .\packaging\stage_payload.py C:\build\aiwf-stage --python-runtime C:\build\python
& 'C:\Program Files (x86)\NSIS\makensis.exe' '/DSTAGE=C:\build\aiwf-stage' '/DOUTPUT=C:\build\AES-Studio-Setup-Preview.exe' .\packaging\AES-Studio-Setup.nsi
```

Use `--cpp-engine`, `--native-build`, `--vc-runtime`, `--wheelhouse`, or
`--credential-bridge` to supply inputs elsewhere. If the credential bridge is
outside `packaging/`, pass the same path to NSIS as `/DCREDENTIAL_BRIDGE=...`.
Do not overwrite an existing stage or installer that belongs to another build.

## Focused tests

The packaging tests use the minimal engine API packages, a locally built x64
uninstall worker, and NSIS for the installer requirements test. On Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-index --find-links .\packaging\wheelhouse -r .\native\AIWF.Desktop\engine-api-requirements.txt
$env:AES_NSIS_COMPILER = 'C:\Program Files (x86)\NSIS\makensis.exe'
.\.venv\Scripts\python.exe -B -m unittest discover -s packaging\tests -v
```

The tests do not replace an installer run on a fresh Windows account or another
physical PC. They do not load model weights. The optional Python backend lock
can be checked with `uv lock --check --offline --project dependencies\windows-py312`;
the complete backend installation and inference path still need validation.
