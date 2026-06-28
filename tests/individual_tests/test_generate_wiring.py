"""Guard Studio generate input order against controller index drift.

AST-level tests run without torch/gradio. A one-position shift here previously
caused stale checkpoint values to feed the wrong callback argument.
"""
from __future__ import annotations

import ast
from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "launch.py").is_file():
            return parent
    raise RuntimeError("Could not find AIWF repo root")


STUDIO = _repo_root() / "aiwf" / "web" / "studio" / "tab.py"
RUNNER = _repo_root() / "aiwf" / "web" / "studio" / "generation_runner.py"
REQUEST_BUILDER = _repo_root() / "aiwf" / "web" / "studio" / "request_builder.py"
REACTOR_WORKFLOW = _repo_root() / "aiwf" / "web" / "studio" / "reactor_workflow.py"
WAN_TAB = _repo_root() / "aiwf" / "web" / "tabs" / "wan_i2v.py"

ANCHORS = [
    ("mode_toggle", 0),
    ("show_editor", 1),
    ("checkpoint", 4),
    ("sampler", 5),
    ("scheduler", 6),
    ("workspace_image", 28),
    ("mask_editor_value", 29),
    ("state", 30),
    ("prompt_file", 33),
    ("style_select", 34),
    ("cn3_threshold_b", 63),
    ("inpaint_source", 64),
    ("continuous_toggle", 65),
    ("cooldown_seconds", 66),
    ("training_metadata_toggle", 67),
    ("reactor_at_gen", 68),
]

WAN_RUN_ANCHORS = [
    ("image", 0),
    ("runtime_mode_v", 17),
    ("rife_enabled_v", 30),
    ("reactor_enabled_v", 37),
    ("vsr_enabled_v", 49),
    ("videofx_denoise_enabled_v", 54),
    ("audio_enabled_v", 69),
    ("audio_cfg_v", 76),
]


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _generate_input_names() -> list[str]:
    for node in ast.walk(_tree(STUDIO)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "generate_inputs":
                    return [elt.id for elt in node.value.elts if isinstance(elt, ast.Name)]
    raise AssertionError("generate_inputs not found in studio/tab.py")


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in {path}")


def _module_constant(path: Path, name: str):
    for node in _tree(path).body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                if isinstance(node.value, ast.Constant):
                    return node.value.value
    raise AssertionError(f"{name} not found in {path}")


def test_generate_inputs_anchor_positions_align_with_runner_indexes():
    inputs_names = _generate_input_names()
    for component, expected_index in ANCHORS:
        assert component in inputs_names, f"{component} missing from generate_inputs"
        assert inputs_names.index(component) == expected_index


def test_studio_tab_delegates_generation_to_runner():
    node = _function_node(STUDIO, "run")
    assert _module_constant(RUNNER, "GENERATION_RUNNER_INPUT_COUNT") == 68
    assert any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "run"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "generation_runner"
        for call in ast.walk(node)
    )
    assert any(
        isinstance(subscript, ast.Subscript)
        and isinstance(subscript.value, ast.Name)
        and subscript.value.id == "args"
        and isinstance(subscript.slice, ast.Slice)
        and isinstance(subscript.slice.lower, ast.Name)
        and subscript.slice.lower.id == "GENERATION_RUNNER_INPUT_COUNT"
        for subscript in ast.walk(node)
    ), "ReActor arguments must start after training_metadata_toggle at index 68"
    assert any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "build_reactor_generation_postprocess_from_args"
        for call in ast.walk(node)
    ), "ReActor generation args should be unpacked outside the tab callback"


def test_reactor_generation_arg_count_matches_generate_inputs_tail():
    inputs_names = _generate_input_names()
    runner_count = _module_constant(RUNNER, "GENERATION_RUNNER_INPUT_COUNT")
    reactor_count = _module_constant(REACTOR_WORKFLOW, "REACTOR_GENERATION_ARG_COUNT")
    assert inputs_names[runner_count : runner_count + reactor_count] == [
        "reactor_at_gen",
        "reactor_source",
        "reactor_source_index",
        "reactor_target_index",
        "reactor_restore",
        "reactor_restorer",
        "reactor_visibility",
        "reactor_cf_weight",
        "reactor_model",
        "reactor_gender_source",
        "reactor_gender_target",
        "reactor_mask",
    ]
    assert inputs_names[runner_count + reactor_count :] == [
        "vsr_enabled",
        "vsr_scale",
        "vsr_mode",
        "vsr_strength",
    ]


def test_generation_runner_delegates_to_request_builder():
    node = _function_node(RUNNER, "build_request")
    assert any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "build_generation_request"
        for call in ast.walk(node)
    )


def test_wan_run_input_order_has_named_anchors():
    node = _function_node(WAN_TAB, "_run")
    args = [arg.arg for arg in node.args.args]
    for name, expected_index in WAN_RUN_ANCHORS:
        assert name in args, f"{name} missing from Wan _run arguments"
        assert args.index(name) == expected_index


def test_request_builder_normalizes_sampler_schedule_pair():
    node = _function_node(REQUEST_BUILDER, "build_generation_request")
    assert any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "normalize_schedule_id_for_sampler"
        for call in ast.walk(node)
    )


def test_request_builder_rejects_stale_checkpoint_selection():
    node = _function_node(REQUEST_BUILDER, "build_generation_request")
    assert any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "resolve_checkpoint_id"
        for call in ast.walk(node)
    )
