import json

from aiwf.web import paid_worker
from aiwf.web.paid_ext_api import _node_registry, _validate_workflow_payload


def test_paid_node_registry_includes_linear_code_block_nodes():
    ids = {node["id"] for node in _node_registry()}
    assert "generation-request" in ids
    assert "template-reference" in ids
    assert "imported-json" in ids


def test_linear_code_block_workflow_validates_without_graph_wires():
    payload = {
        "schema": "aiwf.workflow-code-blocks.v1",
        "blocks": [
            {
                "id": "block-1",
                "label": "Wan settings snapshot",
                "nodeId": "generation-request",
                "order": 1,
                "classes": {"requires": [], "produces": ["artifact"]},
                "payload": {"route": "wan-video", "settings": {"steps": 8}},
                "code": json.dumps({"route": "wan-video", "settings": {"steps": 8}}, indent=2),
            }
        ],
    }
    result = _validate_workflow_payload(payload)
    assert result["valid"] is True
    assert result["mode"] == "linear-code-blocks"
    assert result["blocks"][0]["nodeId"] == "generation-request"
    assert "artifact" in result["availableClasses"]


def test_linear_code_block_rejects_bad_json():
    result = _validate_workflow_payload(
        {
            "schema": "aiwf.workflow-code-blocks.v1",
            "blocks": [
                {"id": "broken", "label": "Broken", "nodeId": "generation-request", "code": "{not-json}"}
            ],
        }
    )
    assert result["valid"] is False
    assert any("not valid JSON" in error for error in result["errors"])


def test_existing_stage_validator_still_checks_stage_requirements():
    result = _validate_workflow_payload({"stages": ["model"]})
    assert result["valid"] is False
    assert any("missing" in error.lower() for error in result["errors"])


def test_workflow_plan_executor_is_validation_only_registered():
    assert "workflow-plan" in paid_worker.registered_kinds()
