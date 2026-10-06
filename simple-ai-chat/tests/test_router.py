from __future__ import annotations

import pytest

from simple_ai_chat.router import Attachment, Capability, ChatRequest, Router, RoutingError


def caps(steps):
    return [(s.capability, s.model_id) for s in steps]


@pytest.fixture
def router(small_registry):
    return Router(small_registry)


def test_plain_text_goes_to_brain_with_profile(router):
    steps = router.plan(ChatRequest("hello"))
    assert caps(steps) == [(Capability.CHAT, "brain")]
    assert steps[0].profile == "chat"
    assert router.plan(ChatRequest("fix this bug", mode="code"))[0].profile == "code"


def test_image_attachment_uses_vision(router):
    steps = router.plan(ChatRequest("what is this?", [Attachment("image", "a.png")]))
    assert caps(steps) == [(Capability.VISION, "brain")]


def test_audio_attachment_is_transcribed_first(router):
    steps = router.plan(ChatRequest("", [Attachment("audio", "a.wav")]))
    assert caps(steps) == [(Capability.ASR, "asr"), (Capability.CHAT, "brain")]


def test_image_command_runs_prompt_enhancer_then_image(router, small_registry):
    assert caps(router.plan(ChatRequest("/image a red fox"))) == [
        (Capability.PROMPT_ENHANCE, "pe"),
        (Capability.IMAGE_GEN, "image"),
    ]
    small_registry.toggles["prompt_enhancer"] = False
    assert caps(router.plan(ChatRequest("/image a red fox"))) == [(Capability.IMAGE_GEN, "image")]


def test_edit_needs_an_image(router):
    assert caps(router.plan(ChatRequest("/edit make it blue"))) == [(Capability.CHAT, "brain")]
    steps = router.plan(ChatRequest("/edit make it blue", [Attachment("image", "a.png")]))
    assert caps(steps) == [(Capability.PROMPT_ENHANCE_EDIT, "pe-edit"), (Capability.IMAGE_EDIT, "image")]


def test_speak_command_and_speak_replies_toggle(router, small_registry):
    assert caps(router.plan(ChatRequest("/speak hello"))) == [(Capability.TTS, "tts")]
    small_registry.toggles["speak_replies"] = True
    assert caps(router.plan(ChatRequest("hi"))) == [(Capability.CHAT, "brain"), (Capability.TTS, "tts")]


def test_moderation_toggle_runs_guard_first(router, small_registry):
    small_registry.toggles["moderation"] = True
    assert caps(router.plan(ChatRequest("hi"))) == [(Capability.MODERATE, "guard"), (Capability.CHAT, "brain")]


def test_simulation_command(router):
    assert caps(router.plan(ChatRequest("/sim rm -rf ./build"))) == [
        (Capability.SIMULATE, "sim"),
    ]


def test_tool_names_map_to_models(router):
    assert router.model_for_tool("generate_image").id == "image"
    assert router.model_for_tool("memory_search").id == "embed"
    with pytest.raises(RoutingError):
        router.model_for_tool("launch_rocket")


def test_disabled_capability_raises(router):
    with pytest.raises(RoutingError, match="web_simulate"):
        router.plan(ChatRequest("/websim click login"))
