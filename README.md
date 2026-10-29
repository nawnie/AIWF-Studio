# AIWF Studio

AIWF Studio is a local-first creative application for Windows. The current
native preview installs as **AES Studio**, alongside the earlier AIWF Studio
application, with its own settings and project data.

> **Preview:** This is an unsigned tester build. Hardware and model support
> are limited, and several workflows require separately installed software.
> See the requirements and known limitations below before installing.

## What the preview does

- Provides a native Windows interface, a local API, and a first-run setup wizard.
- Uses a bundled CUDA build of stable-diffusion.cpp for supported SD 1.5 and
  SDXL single-file checkpoints. Model files are supplied separately.
- Offers experimental ComfyUI templates for Qwen Image 2.1 editing, MOSS Image,
  Wan 2.2 image-to-video, and LTX 2.5 GGUF image-to-video. These require an
  independently installed and running ComfyUI, suitable models, and the
  required nodes for each workflow.
- Keeps the optional Diffusers backend available after its separate Python
  environment has been provisioned. Its complete installation and generation
  path still need clean-machine validation.
- Stores optional Hugging Face and Civitai keys in Windows Credential Manager.
  Keys are not required for every model and are not validated during setup.

Finding a model or passing a dependency check does not establish that a
particular model, architecture, quantization, or GPU will run successfully.
Video templates currently produce short, 17-frame previews.

## Windows requirements

The shell targets 64-bit Windows 10 version 2004 (build 19041) or later.
Windows on ARM has not been tested. The bundled CUDA 13.2 image engine has
additional requirements:

- A supported NVIDIA GPU and driver; CUDA 13 minor-version compatibility
  requires driver 580 or newer, subject to NVIDIA's feature limitations.
- A CPU with AVX2, FMA, F16C, and BMI2 for the included engine build.
- Windows 10 22H2, or a supported Windows 11 release, for NVIDIA's CUDA 13.2
  support matrix.
- A compatible model and sufficient system memory, video memory, and disk
  space. The setup recommendation is a heuristic, not a fit guarantee.

The tester installer bundles the Windows App SDK, an offline Python API
runtime, app-local Microsoft Visual C++ libraries, and the C++ engine's CUDA
libraries. It does not include model weights, a full Python inference stack,
ComfyUI, its nodes, or an NVIDIA display driver. A CUDA Toolkit developer
installation is not needed just to run the bundled engine. For source builds,
see [BUILDING.md](BUILDING.md).

See NVIDIA's [CUDA 13.2 Windows support matrix](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-installation-guide-microsoft-windows/index.html)
and [driver compatibility guidance](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html)
for the current requirements and limitations.

## Install and data

An installer download has not yet been published to this repository. The
preview installer is per-user and allows a custom empty installation folder.
Its first-run wizard helps select data and model folders. The separate AES
profile defaults to `%LOCALAPPDATA%\Programs\AES Studio Data`.

The installer does not automatically download models. Uninstall preserves
models, outputs, settings, credentials, and modified files. Remove provider keys
in Configure Studio or Windows Credential Manager when desired. Back up your
data before changing installations.

## Verification and limitations

The v25 preview was exercised in Windows Sandbox and on the development PC.
The installed app generated SD 1.5 and MOSS images, a Qwen edit, and a short
Wan preview with existing models and ComfyUI. An LTX preview was tested through
the local workflow runner. The Wan sample had framing artifacts. A fresh Windows
account, a second physical PC, real provider credentials, and complete optional
Python inference remain unverified. The installer is unsigned.

The v25 installer's optional Python dependency lock differs from the bundled
API lock. The source lock in this repository has been aligned and checked
offline, but the complete Python backend has not yet been installed and tested.
Do not treat the v25 Python provisioning path as validated.

A subsequent installer was assembled with the aligned lock; its offline setup
and API startup were checked in an isolated directory. Its complete interactive
installer flow has not yet been tested. The Microsoft Windows App SDK's
redistribution conditions also need review before a binary is published.

## Development

The native shell lives in `native/AIWF.Desktop`; the local API and services are
in `aiwf/`; the preview installer and focused tests are in `packaging/`.
See [BUILDING.md](BUILDING.md) for build inputs, validation, and the distinction
between source and the redistributable installer. Do not commit model weights,
credentials, local user data, or generated installers.

## License

The Studio source is distributed under the [MIT license](LICENSE). Bundled
components have their own terms and notices; see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) and the `licenses/` directory.
