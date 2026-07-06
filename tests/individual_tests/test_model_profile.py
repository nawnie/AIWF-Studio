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


def test_flux_fusion_profile_uses_four_steps():
    p = detect_model_profile("fluxFusionV24StepsGGUFNF4_V2NF4.safetensors")
    assert p.family == "flux_fusion"
    assert p.is_distilled
    assert p.recommended_cfg == 1.0
    assert p.recommended_steps == 4
    assert p.recommended_sampler == "euler"


def test_flux_fill_profile_is_inpaint_not_turbo():
    p = detect_model_profile("flux1-fill-dev.safetensors")
    assert p.family == "flux_fill"
    assert p.is_distilled is False
    assert p.recommended_cfg == 3.5
    assert p.recommended_steps == 28
    assert p.recommended_sampler == "euler"


def test_flux2_klein_profile_uses_model_page_defaults():
    p = detect_model_profile("fluxtraitFLUX2KleinFLUXZ_klein9bV2Q4KM.gguf")
    assert p.family == "flux2_klein"
    assert p.recommended_cfg == 1.0
    assert p.cfg_max == 1.5
    assert p.recommended_steps == 12
    assert p.recommended_sampler == "euler"


def test_z_image_profile_wins_over_flux2_name_prefix():
    p = detect_model_profile("fluxtraitFLUX2KleinFLUXZ_zImageV2GgufQ4.gguf")
    assert p.family == "z_image"
    assert p.recommended_cfg == 1.0
    assert p.cfg_max == 1.5
    assert p.recommended_steps == 8
    assert p.recommended_sampler == "euler"


def test_krea2_turbo_profile_wins_over_generic_turbo():
    p = detect_model_profile("krea2_turbo_fp8_scaled.safetensors")
    assert p.family == "krea2_turbo"
    assert p.is_distilled
    assert p.recommended_cfg == 0.0
    assert p.recommended_steps == 8
    assert p.recommended_sampler == "euler"


def test_krea2_raw_and_anima_are_not_distilled_profiles():
    raw = detect_model_profile("Krea-2-Raw")
    assert raw.family == "krea2_raw"
    assert raw.is_distilled is False
    assert raw.recommended_cfg == 3.5
    assert raw.recommended_steps == 52

    anima = detect_model_profile("anima-base-v1.0.safetensors")
    assert anima.family == "anima"
    assert anima.is_distilled is False
    assert anima.recommended_cfg == 4.5
    assert anima.recommended_steps == 36


def test_qwen_nunchaku_profile_wins_over_generic_lightning():
    p = detect_model_profile("svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors")
    assert p.family == "qwen_image_nunchaku"
    assert p.recommended_cfg == 1.0
    assert p.cfg_max == 1.5
    assert p.recommended_steps == 4
    assert p.recommended_sampler == "euler"


def test_sana_sprint_profile_uses_two_step_default():
    p = detect_model_profile("Sana_Sprint_0.6B_1024px_diffusers")
    assert p.family == "sana_sprint"
    assert p.recommended_cfg == 4.5
    assert p.recommended_steps == 2
    assert p.recommended_sampler == "euler"


def test_sana_video_profile_uses_video_defaults():
    p = detect_model_profile("SANA-Video_2B_480p_diffusers")
    assert p.family == "sana_video"
    assert p.recommended_cfg == 6.0
    assert p.recommended_steps == 50
    assert p.recommended_sampler == "euler"


def test_sdxl_refiner_profile_is_second_pass():
    p = detect_model_profile("sd_xl_refiner_1.0.safetensors")
    assert p.family == "sdxl_refiner"
    assert p.is_distilled is False
    assert p.recommended_cfg == 6.0
    assert p.recommended_steps == 10
    assert p.recommended_sampler == "dpmpp_2m"


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
