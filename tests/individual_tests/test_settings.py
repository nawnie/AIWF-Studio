from aiwf.core.config.settings import UserSettings


def test_live_preview_disabled_returns_zero_interval():
    settings = UserSettings(enable_live_preview=False, show_progress_every_n_steps=3)
    assert settings.live_preview_interval() == 0
    assert settings.live_preview_summary() == "Live preview off"


def test_live_preview_enabled_clamps_interval():
    settings = UserSettings(enable_live_preview=True, show_progress_every_n_steps=5)
    assert settings.live_preview_interval() == 5
    assert settings.live_preview_summary() == "Live preview every 5 steps (VAE decode)"


def test_live_preview_every_step_summary():
    settings = UserSettings(enable_live_preview=True, show_progress_every_n_steps=1)
    assert settings.live_preview_interval() == 1
    assert settings.live_preview_summary() == "Live preview every step (VAE decode)"


def test_live_preview_unsupported_decoder_disables_interval():
    settings = UserSettings(enable_live_preview=True, live_preview_decoder="taesd")
    assert settings.live_preview_interval() == 0
    assert settings.live_preview_summary() == "Live preview off"


def test_saving_output_defaults_preserve_legacy_behavior():
    s = UserSettings()
    assert s.save_grid is False
    assert s.save_sidecar_txt is False
    assert s.filename_pattern == "[datetime]"
    assert s.save_before_hires is False
    assert s.save_interrupted is False
    assert s.live_preview_decoder == "vae"
    assert s.live_preview_title_progress is True
    assert s.metadata_include_model_hash is True
    assert s.metadata_include_vae_hash is True
    assert s.metadata_include_lora_hashes is True
    assert s.metadata_include_app_version is True
    assert s.metadata_include_optimization_profile is True
    assert s.optimization_profile_id == "balanced_sdpa_fp16"
    assert s.default_hr_upscaler == "lanczos"
    assert s.sdxl_refiner_enabled is False
    assert s.sdxl_refiner_checkpoint_id is None
    assert s.sdxl_refiner_steps == 10
    assert s.sdxl_refiner_strength == 0.25
    assert s.pnginfo_send_to_studio is True
    assert s.pnginfo_clear_after_apply is True


def test_saving_output_settings_round_trip():
    s = UserSettings(
        save_grid=True,
        save_sidecar_txt=True,
        filename_pattern="[seed]-[seq]",
        save_before_hires=True,
        save_interrupted=True,
        live_preview_decoder="vae",
        live_preview_title_progress=False,
        metadata_include_model_hash=False,
        metadata_include_vae_hash=False,
        metadata_include_lora_hashes=False,
        metadata_include_app_version=False,
        metadata_include_optimization_profile=False,
        optimization_profile_id="safe_eager_cuda",
        default_hr_upscaler="bicubic",
        sdxl_refiner_enabled=True,
        sdxl_refiner_checkpoint_id="sdxl-refiner",
        sdxl_refiner_steps=8,
        sdxl_refiner_strength=0.2,
        pnginfo_send_to_studio=False,
        pnginfo_clear_after_apply=False,
    )
    restored = UserSettings(**s.model_dump())
    assert restored.save_grid is True
    assert restored.save_sidecar_txt is True
    assert restored.filename_pattern == "[seed]-[seq]"
    assert restored.save_before_hires is True
    assert restored.save_interrupted is True
    assert restored.live_preview_decoder == "vae"
    assert restored.live_preview_title_progress is False
    assert restored.metadata_include_model_hash is False
    assert restored.metadata_include_vae_hash is False
    assert restored.metadata_include_lora_hashes is False
    assert restored.metadata_include_app_version is False
    assert restored.metadata_include_optimization_profile is False
    assert restored.optimization_profile_id == "safe_eager_cuda"
    assert restored.default_hr_upscaler == "bicubic"
    assert restored.sdxl_refiner_enabled is True
    assert restored.sdxl_refiner_checkpoint_id == "sdxl-refiner"
    assert restored.sdxl_refiner_steps == 8
    assert restored.sdxl_refiner_strength == 0.2
    assert restored.pnginfo_send_to_studio is False
    assert restored.pnginfo_clear_after_apply is False
