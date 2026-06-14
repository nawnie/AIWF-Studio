from aiwf.core.model_profile import detect_model_profile


def test_lightning_detected():
    p = detect_model_profile("RealVisXL_V5.0_Lightning_fp16")
    assert p.family == "lightning" and p.is_distilled
    assert p.recommended_cfg <= 2.0 and p.cfg_max <= 2.5
    assert p.recommended_steps <= 8


def test_hypervae_is_NOT_distilled():
    # "Hyper VAE" is a baked VAE on a normal SD1.5 model, not Hyper-SD.
    p = detect_model_profile("realisticVisionV60B1_v51HyperVAE (1)")
    assert p.family == "standard"
    assert p.is_distilled is False
    assert p.recommended_cfg == 7.0


def test_hyper_sd_detected():
    for name in ["Hyper-SDXL-8step", "hyperSD_4step", "Hyper_SD_unet"]:
        p = detect_model_profile(name)
        assert p.is_distilled and p.family == "hyper", name


def test_turbo_and_lcm_and_tcd():
    assert detect_model_profile("sd_xl_turbo_1.0_fp16").family == "turbo"
    assert detect_model_profile("dreamshaper_8LCM").family == "lcm"
    assert detect_model_profile("somemodel-TCD").family == "tcd"


def test_standard_model():
    p = detect_model_profile("dreamshaper_8", "dreamshaper.safetensors")
    assert p.family == "standard" and not p.is_distilled
    assert p.recommended_cfg == 7.0


def test_help_text_and_title_present():
    p = detect_model_profile("foo_Lightning")
    assert "CFG" in p.help_text
    assert "Lightning" in p.title


def test_empty_names_safe():
    p = detect_model_profile(None, "", None)
    assert p.family == "standard"
