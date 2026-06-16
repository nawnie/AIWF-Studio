"""Tests for aiwf/core/domain/engine.py — EngineTenant + switch domain models."""
from __future__ import annotations

import pytest

from aiwf.core.domain.engine import (
    EngineStatus,
    EngineSwitchRequest,
    EngineSwitchResult,
    EngineTenant,
)


# ---------------------------------------------------------------------------
# EngineTenant
# ---------------------------------------------------------------------------

class TestEngineTenant:
    def test_values_are_strings(self):
        assert EngineTenant.IDLE == "idle"
        assert EngineTenant.VIDEO == "video"
        assert EngineTenant.CHAT == "chat"

    def test_gpu_heavy_excludes_idle_and_chat(self):
        assert not EngineTenant.IDLE.is_gpu_heavy()
        assert not EngineTenant.CHAT.is_gpu_heavy()

    def test_gpu_heavy_includes_image_video_training(self):
        for tenant in (
            EngineTenant.IMAGE,
            EngineTenant.VIDEO,
            EngineTenant.LORA_TRAINING,
            EngineTenant.FULL_TRAINING,
            EngineTenant.ENHANCE,
        ):
            assert tenant.is_gpu_heavy(), f"{tenant} should be GPU-heavy"

    def test_friendly_name_returns_string(self):
        for tenant in EngineTenant:
            name = tenant.friendly_name()
            assert isinstance(name, str) and len(name) > 0

    def test_str_enum_usable_as_string_key(self):
        d = {EngineTenant.VIDEO: "video_val"}
        assert d["video"] == "video_val"


# ---------------------------------------------------------------------------
# EngineSwitchRequest
# ---------------------------------------------------------------------------

class TestEngineSwitchRequest:
    def test_defaults(self):
        req = EngineSwitchRequest(target=EngineTenant.VIDEO)
        assert req.reason == ""
        assert req.job_id == ""
        assert req.allow_wait is False

    def test_frozen(self):
        req = EngineSwitchRequest(target=EngineTenant.IMAGE, reason="test")
        with pytest.raises((AttributeError, TypeError)):
            req.target = EngineTenant.IDLE  # type: ignore[misc]

    def test_fields(self):
        req = EngineSwitchRequest(
            target=EngineTenant.CHAT,
            reason="user opened chat",
            job_id="chat",
            allow_wait=True,
        )
        assert req.target == EngineTenant.CHAT
        assert "chat" in req.reason
        assert req.job_id == "chat"
        assert req.allow_wait is True


# ---------------------------------------------------------------------------
# EngineSwitchResult
# ---------------------------------------------------------------------------

class TestEngineSwitchResult:
    def test_ok_result(self):
        r = EngineSwitchResult(ok=True, active=EngineTenant.VIDEO, message="switched")
        assert r.ok
        assert r.active == EngineTenant.VIDEO
        assert r.log_path is None

    def test_failed_result(self):
        r = EngineSwitchResult(ok=False, active=EngineTenant.IDLE, message="GPU busy")
        assert not r.ok

    def test_frozen(self):
        r = EngineSwitchResult(ok=True, active=EngineTenant.IDLE, message="idle")
        with pytest.raises((AttributeError, TypeError)):
            r.ok = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EngineStatus
# ---------------------------------------------------------------------------

class TestEngineStatus:
    def test_initial_state(self):
        s = EngineStatus()
        assert s.active == EngineTenant.IDLE
        assert s.activated_at is None
        assert s.switch_count == 0

    def test_record_switch(self):
        s = EngineStatus()
        s.record_switch(EngineTenant.VIDEO, "starting video job")
        assert s.active == EngineTenant.VIDEO
        assert s.activated_at is not None
        assert s.switch_count == 1
        assert "video" in s.last_switch_message.lower()

    def test_multiple_switches_increment_count(self):
        s = EngineStatus()
        s.record_switch(EngineTenant.IMAGE, "image job")
        s.record_switch(EngineTenant.IDLE, "done")
        assert s.switch_count == 2
        assert s.active == EngineTenant.IDLE

    def test_to_dict(self):
        s = EngineStatus()
        s.record_switch(EngineTenant.CHAT, "chat session")
        d = s.to_dict()
        assert d["active"] == "chat"
        assert d["switch_count"] == 1
        assert d["activated_at"] is not None


# ---------------------------------------------------------------------------
# Transition logic (no supervisor — pure domain)
# ---------------------------------------------------------------------------

class TestTransitions:
    """Verify that the domain correctly models common transition sequences."""

    def test_idle_to_video_is_gpu_heavy(self):
        req = EngineSwitchRequest(target=EngineTenant.VIDEO)
        assert req.target.is_gpu_heavy()

    def test_idle_to_chat_is_not_gpu_heavy(self):
        req = EngineSwitchRequest(target=EngineTenant.CHAT)
        assert not req.target.is_gpu_heavy()

    def test_chat_to_video_is_gpu_heavy(self):
        # Simulates: user clicks video tab while Ollama is active
        req = EngineSwitchRequest(
            target=EngineTenant.VIDEO,
            reason="user opened video tab",
        )
        assert req.target.is_gpu_heavy()
