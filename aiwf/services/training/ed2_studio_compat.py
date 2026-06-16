from __future__ import annotations

import importlib.metadata as metadata
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ED2DependencyCheck:
    distribution: str
    minimum: str
    installed: str | None
    required: bool
    ok: bool
    note: str = ""


@dataclass(frozen=True)
class ED2StudioCompatResult:
    ok: bool
    checks: tuple[ED2DependencyCheck, ...]
    warnings: tuple[str, ...]

    @property
    def missing_required(self) -> tuple[str, ...]:
        return tuple(check.distribution for check in self.checks if check.required and not check.ok)


_CORE_REQUIREMENTS: tuple[tuple[str, str, bool, str], ...] = (
    ("torch", "2.3.0", True, "Studio CUDA torch; do not downgrade for ED2."),
    ("torchvision", "0.16.0", True, "Must match the installed torch family."),
    ("diffusers", "0.35.0", True, "Studio baseline; ED2 upstream pins an older version."),
    ("transformers", "4.38.2", True, "Must remain below transformers 5."),
    ("accelerate", "0.26.0", True, ""),
    ("peft", "0.9.0", True, ""),
    ("safetensors", "0.4.0", True, ""),
    ("numpy", "1.23.5", True, "Studio uses numpy 2.x; verify ED2 dataset paths before long runs."),
    ("compel", "1.1.3", True, "Studio may be newer than ED2's historical pin."),
    ("ftfy", "6.1.1", True, ""),
    ("torchsde", "0.2.0", True, ""),
    ("colorama", "0.4.6", True, ""),
)

_STUDIO_OVERLAY_REQUIREMENTS: tuple[tuple[str, str, bool, str], ...] = (
    ("tensorboard", "2.11.0", True, "ED2 writes training summaries."),
    ("omegaconf", "2.2.3", True, ""),
    ("pyre-extensions", "0.0.29", True, ""),
    ("lion-pytorch", "0.2.0", True, "Lion optimizer support."),
    ("tiktoken", "0.5.0", True, ""),
    ("aiohttp", "3.9.0", True, ""),
    ("wandb", "0.16.0", True, "Imported by ED2 train.py even when logging is disabled."),
    ("pynvml", "11.4.1", True, "Pinned to the ED2-compatible package shape; 13.x removes pynvml.smi."),
    ("bitsandbytes", "0.43.0", True, "Modern Windows wheels require torch >= 2.3."),
    ("pytorch-lightning", "2.2.0", False, "ED2 upstream lists Lightning; keep advisory until import path is proven."),
    ("dowg", "0.3.1", True, ""),
)


def check_ed2_studio_compat(installed_versions: dict[str, str | None] | None = None) -> ED2StudioCompatResult:
    """Check whether the main Studio runtime can plausibly run ED2.

    This is a preflight only. It never imports ED2, torch, diffusers, or any
    optional training package.
    """
    checks: list[ED2DependencyCheck] = []
    for distribution, minimum, required, note in (*_CORE_REQUIREMENTS, *_STUDIO_OVERLAY_REQUIREMENTS):
        installed = _installed_version(distribution, installed_versions)
        ok = _dependency_ok(distribution, installed, minimum)
        checks.append(
            ED2DependencyCheck(
                distribution=distribution,
                minimum=minimum,
                installed=installed,
                required=required,
                ok=ok,
                note=note,
            )
        )

    warnings = list(_compat_warnings(installed_versions))
    missing_required = [check for check in checks if check.required and not check.ok]
    return ED2StudioCompatResult(
        ok=not missing_required,
        checks=tuple(checks),
        warnings=tuple(warnings),
    )


def ed2_studio_compat_markdown(result: ED2StudioCompatResult) -> str:
    mark = "OK" if result.ok else "Missing"
    lines = [f"**ED2 on Studio runtime:** {mark}"]
    for check in result.checks:
        icon = "OK" if check.ok else ("Incompatible" if check.installed else ("Missing" if check.required else "Optional"))
        installed = check.installed or "not installed"
        suffix = f" - {check.note}" if check.note else ""
        lines.append(f"- {icon} `{check.distribution}` >= {check.minimum} ({installed}){suffix}")
    if result.warnings:
        lines.append("")
        lines.extend(f"- Warning: {warning}" for warning in result.warnings)
    return "\n".join(lines)


def _installed_version(distribution: str, installed_versions: dict[str, str | None] | None) -> str | None:
    key = distribution.lower()
    if installed_versions is not None:
        normalized = {str(k).lower(): v for k, v in installed_versions.items()}
        value = normalized.get(key)
        return None if value is None else str(value)
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _version_at_least(version: str, minimum: str) -> bool:
    current = _version_tuple(version)
    required = _version_tuple(minimum)
    width = max(len(current), len(required))
    return current + (0,) * (width - len(current)) >= required + (0,) * (width - len(required))


def _dependency_ok(distribution: str, installed: str | None, minimum: str) -> bool:
    if installed is None:
        return False
    if not _version_at_least(installed, minimum):
        return False
    if distribution == "pynvml":
        return not _version_at_least(installed, "13.0.0")
    return True


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value.split("+", 1)[0])[:4]) or (0,)


def _compat_warnings(installed_versions: dict[str, str | None] | None) -> tuple[str, ...]:
    warnings: list[str] = [
        "Do not install EveryDream2trainer/requirements.txt into the Studio runtime; it pins older torch, diffusers, numpy, protobuf, xformers, and compel.",
        "xformers is intentionally not part of the Studio ED2 overlay; use the Studio CUDA/SageAttention stack for generation and keep ED2 training preflighted.",
    ]
    transformers = _installed_version("transformers", installed_versions)
    if transformers is not None and _version_at_least(transformers, "5.0.0"):
        warnings.append("transformers 5.x is not supported by Studio checkpoint loading or this ED2 compatibility path.")
    pynvml = _installed_version("pynvml", installed_versions)
    if pynvml is not None and _version_at_least(pynvml, "13.0.0"):
        warnings.append("pynvml 13.x is incompatible with ED2's pynvml.smi import; install the AIWF ED2 overlay pin.")
    return tuple(warnings)
