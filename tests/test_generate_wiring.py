"""Guard against misalignment between the generate event's inputs list and run()'s signature.

This is an AST-level test so it runs without torch/gradio installed. A one-position
shift here is what caused the "No checkpoint available. Refresh models." bug:
inpaint_source was inserted mid-list while run() expected it third-from-last,
so ckpt_map received the mask editor's value (None).
"""
from __future__ import annotations

import ast
from pathlib import Path

STUDIO = Path(__file__).resolve().parents[1] / "aiwf" / "web" / "studio.py"

# Anchor pairs: (component name in generate_inputs, parameter name in run()).
# Each pair must sit at the same position in both sequences.
ANCHORS = [
    ("mode_toggle", "mode_label"),
    ("checkpoint", "ckpt_title"),
    ("workspace_image", "source_image"),
    ("mask_editor", "editor_value"),
    ("state", "ckpt_map"),
    ("prompt_file", "prompt_file_path"),
    ("inpaint_source", "inpaint_source"),
    ("continuous_toggle", "continuous_enabled"),
    ("cooldown_seconds", "cooldown_wait"),
]


def _collect():
    tree = ast.parse(STUDIO.read_text(encoding="utf-8"))
    run_params = None
    inputs_names = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            run_params = [a.arg for a in node.args.args]
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "generate_inputs":
                    inputs_names = [
                        elt.id for elt in node.value.elts if isinstance(elt, ast.Name)
                    ]
    return run_params, inputs_names


def test_generate_inputs_match_run_signature_length():
    run_params, inputs_names = _collect()
    assert run_params is not None, "run() not found in studio.py"
    assert inputs_names is not None, "generate_inputs not found in studio.py"
    assert len(run_params) == len(inputs_names)


def test_generate_inputs_anchor_positions_align():
    run_params, inputs_names = _collect()
    for component, param in ANCHORS:
        assert component in inputs_names, f"{component} missing from generate_inputs"
        assert param in run_params, f"{param} missing from run() signature"
        ci = inputs_names.index(component)
        pi = run_params.index(param)
        assert ci == pi, (
            f"generate_inputs[{ci}]={component!r} feeds run() param "
            f"{run_params[ci]!r}, expected {param!r} (at position {pi})"
        )
