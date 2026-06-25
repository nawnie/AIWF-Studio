from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[2] / "aiwf_update_bbb1cae_studio_v4.py"
    spec = importlib.util.spec_from_file_location("aiwf_update_bbb1cae_studio_v4_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v4_makes_checkpoint_warmup_opt_in():
    module = _module()
    source = '''def _background_model_warmup(ctx):\n    if os.environ.get("AIWF_BACKGROUND_WARMUP", "1").strip().lower() in {"0", "false", "no", "off"}:\n        return\n'''
    result = module.patch_app(source)
    assert 'AIWF_BACKGROUND_WARMUP", "0"' in result
    assert "Disabled by default" in result


def test_v4_pytest_warning_filter_is_portable():
    module = _module()
    source = "'ignore:Using `httpx` with `starlette\\.testclient` is deprecated.*:starlette.exceptions.StarletteDeprecationWarning'"
    result = module.patch_pyproject(source)
    assert "starlette.exceptions.StarletteDeprecationWarning" not in result
    assert result.endswith(":DeprecationWarning'")


def test_v4_accepts_previous_wan_hotfix_markers():
    module = _module()
    source = "\n".join(
        (
            "AIWF HOTFIX bbb1cae-v2: persistent cache for dequantized GGUF UMT5",
            "Scheduler/sigma/flow are cheap runtime objects",
            "_auto_chunked_vae_decode",
        )
    )
    assert module.patch_wan_pipeline(source) == source
