from __future__ import annotations

import pytest
from pathlib import Path

from aiwf.services.model_download_catalog import MODEL_DOWNLOAD_CATALOG, QUICK_START_BUNDLES
from aiwf.services.model_setup_manifest import (
    CONTROLNET_SETUP_CATALOG_KEYS,
    MODEL_SETUP_ROUTES,
    recommended_setup_bundle_key,
    setup_route_descriptor,
)


def test_setup_manifest_bundle_references_resolve_to_catalog_assets():
    catalog_keys = {entry.key for entry in MODEL_DOWNLOAD_CATALOG}

    for route in MODEL_SETUP_ROUTES:
        assert route.preflight_key
        assert route.bundle_key or route.setup_action or route.limitation
        if route.bundle_key:
            assert route.bundle_key in QUICK_START_BUNDLES

    assert all(key in catalog_keys for bundle in QUICK_START_BUNDLES.values() for key in bundle)
    catalog_entries = {entry.key: entry for entry in MODEL_DOWNLOAD_CATALOG}
    for family, keys in CONTROLNET_SETUP_CATALOG_KEYS.items():
        assert keys
        for key in keys:
            assert key in catalog_entries
            assert catalog_entries[key].category == "controlnet"
            assert key.startswith("cn15-" if family == "sd15" else "cnxl-")
    assert recommended_setup_bundle_key({"architecture": "flux_kontext", "title": "Flux Kontext"}) == "flux-kontext"
    assert recommended_setup_bundle_key({
        "architecture": "flux", "title": "Flux dev", "flux_conditioning_mode": "teacher",
    }) == "flux-components"
    assert recommended_setup_bundle_key({
        "architecture": "flux", "title": "Flux dev", "flux_conditioning_mode": "distillt5-control",
    }) == "flux-distillt5-control-components"
    assert recommended_setup_bundle_key({
        "architecture": "flux_fill", "title": "Flux Fill", "flux_conditioning_mode": "universal",
    }) is None
    universal_route = setup_route_descriptor({
        "architecture": "flux", "title": "Flux dev", "flux_conditioning_mode": "universal",
    })
    assert universal_route is not None
    assert universal_route["supportState"] == "runtime-dependent"
    assert universal_route["setupBundleKey"] is None
    assert universal_route["setupAction"] is None
    assert "does not currently provide a verified installation source" in universal_route["limitation"]
    assert "reviewed" not in universal_route["limitation"]
    assert setup_route_descriptor({
        "architecture": "flux", "title": "Flux dev", "flux_conditioning_mode": "unknown",
    }) is None
    assert recommended_setup_bundle_key({
        "architecture": "flux_kontext", "title": "Flux Kontext GGUF", "path": "F:/models/flux-kontext.gguf"
    }) == "flux-kontext-gguf-components"
    kontext = next(item for item in MODEL_DOWNLOAD_CATALOG if item.key == "flux-kontext-diffusers")
    assert kontext.repo_id == "black-forest-labs/FLUX.1-Kontext-dev"
    assert kontext.snapshot is True
    assert "accepting the Hugging Face access terms" in kontext.notes
    kontext_gguf = next(item for item in MODEL_DOWNLOAD_CATALOG if item.key == "flux-kontext-gguf-components")
    assert kontext_gguf.category == "flux_kontext_components"
    assert "transformer/**" not in kontext_gguf.snapshot_allow_patterns
    assert "transformer/config.json" in kontext_gguf.snapshot_allow_patterns


def test_setup_manifest_selects_variant_specific_bundles(tmp_path):
    assert recommended_setup_bundle_key({"architecture": "inpaint", "title": "SD 1.5 Inpaint"}) == "sd-components"
    assert recommended_setup_bundle_key({"architecture": "sdxl_inpaint", "title": "SDXL Inpaint"}) == "sdxl-components"
    assert recommended_setup_bundle_key({"architecture": "sdxl_refiner", "title": "SDXL Refiner"}) == "sdxl-components"
    assert recommended_setup_bundle_key({"architecture": "sd35", "title": "SD 3.5 Large"}) == "sd35-components"
    assert "hf-sdxl-refiner-singlefile-config" in QUICK_START_BUNDLES["sdxl"]
    assert QUICK_START_BUNDLES["sd-components"] == ["hf-vae-mse", "hf-sd15-singlefile-config", "hf-sd15-inpaint-singlefile-config"]
    assert "hf-sd15-pruned" not in QUICK_START_BUNDLES["sd-components"]
    assert "hf-sdxl-base" not in QUICK_START_BUNDLES["sdxl-components"]
    assert "hf-sdxl-refiner" not in QUICK_START_BUNDLES["sdxl-components"]
    assert QUICK_START_BUNDLES["sd35-components"] == ["hf-sd35-singlefile-config"]
    assert recommended_setup_bundle_key({"architecture": "sdxl_unknown", "title": "Unknown SDXL variant"}) is None
    assert recommended_setup_bundle_key({"architecture": "flux2_klein", "title": "Flux.2 Klein 4B"}) == "flux2-4b-components"
    flux2_base = tmp_path / "FLUX.2-klein-base-4B"
    flux2_base.mkdir()
    (flux2_base / "model_index.json").write_text('{"_class_name":"Flux2KleinPipeline"}', encoding="utf-8")
    assert recommended_setup_bundle_key({
        "architecture": "flux2_klein", "title": "Flux.2 Klein Base 4B", "path": str(flux2_base),
    }) == "flux2-4b-base"
    assert recommended_setup_bundle_key({"architecture": "flux2_klein", "title": "Flux.2 Klein 9B"}) == "flux2-9b-components"
    assert recommended_setup_bundle_key({"architecture": "flux2_klein", "title": "Flux.2 Klein"}) is None
    assert recommended_setup_bundle_key({
        "architecture": "flux2_klein",
        "title": "Flux.2 Klein",
        "filename": "flux2-klein.safetensors",
        "path": "C:/workspace/4b/archive/flux2-klein.safetensors",
    }) is None
    assert recommended_setup_bundle_key({
        "architecture": "flux2_klein",
        "title": "Flux.2 Klein",
        "filename": "flux2-klein-9b.safetensors",
        "path": "C:/workspace/4b/archive/flux2-klein-9b.safetensors",
    }) == "flux2-9b-components"
    assert recommended_setup_bundle_key({"architecture": "flux2", "title": "Flux.2 9B"}) is None
    assert recommended_setup_bundle_key({"architecture": "flux2", "title": "Flux.2 4B"}) is None
    full_4b = tmp_path / "FLUX.2-klein-4B"
    full_4b.mkdir()
    (full_4b / "model_index.json").write_text('{"_class_name":"Flux2KleinPipeline"}', encoding="utf-8")
    assert recommended_setup_bundle_key({
        "architecture": "flux2_klein", "title": "Flux.2 Klein 4B", "path": str(full_4b),
    }) == "flux2"
    transformer = full_4b / "transformer"
    transformer.mkdir()
    (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    assert recommended_setup_bundle_key({
        "architecture": "flux2_klein", "title": "Flux.2 Klein 4B", "path": str(full_4b),
    }) == "flux2"
    full_4b_route = setup_route_descriptor({
        "architecture": "flux2_klein", "title": "Flux.2 Klein 4B", "path": str(full_4b),
    })
    assert full_4b_route is not None
    assert full_4b_route["routeKey"] == "pro.image.flux2.4b-full-snapshot"
    assert full_4b_route["setupBundleKey"] == "flux2"
    qwen21 = tmp_path / "Qwen-Image-2.1"
    qwen21.mkdir()
    (qwen21 / "model_index.json").write_text('{"_class_name":"QwenImage21Pipeline"}', encoding="utf-8")
    assert recommended_setup_bundle_key({
        "architecture": "qwen_image", "title": "Qwen Image 2.1", "path": str(qwen21),
    }) is None
    qwen21_route = setup_route_descriptor({
        "architecture": "qwen_image", "title": "Qwen Image 2.1", "path": str(qwen21),
    })
    assert qwen21_route is not None
    assert qwen21_route["routeKey"] == "pro.image.qwen-2.1"
    assert qwen21_route["supportState"] == "runtime-dependent"
    assert "QwenImage21Pipeline" in qwen21_route["limitation"]
    qwen21_available = setup_route_descriptor({
        "architecture": "qwen_image", "title": "Qwen Image 2.1", "path": str(qwen21),
        "qwen_image_21_pipeline_available": True,
    })
    qwen21_unavailable = setup_route_descriptor({
        "architecture": "qwen_image", "title": "Qwen Image 2.1", "path": str(qwen21),
        "qwen_image_21_pipeline_available": False,
    })
    assert qwen21_available is not None
    assert qwen21_available["supportState"] == "supported-when-folder-installed"
    assert qwen21_unavailable is not None
    assert qwen21_unavailable["supportState"] == "blocked-runtime"
    assert recommended_setup_bundle_key({"architecture": "sana_video", "title": "Sana Video 480p"}) == "sana-video"
    assert recommended_setup_bundle_key({"architecture": "sana_video", "title": "Sana Video 720p"}) == "sana-video-720p"
    sana_720p_route = next(route for route in MODEL_SETUP_ROUTES if route.route_key == "pro.video.sana-video.720p")
    assert sana_720p_route.support_state == "supported"
    assert sana_720p_route.bundle_key == "sana-video-720p"
    sana_720p_bundle = QUICK_START_BUNDLES["sana-video-720p"]
    assert sana_720p_bundle == ["sana-video-2b-720p-diffusers"]
    sana_720p_asset = next(item for item in MODEL_DOWNLOAD_CATALOG if item.key == sana_720p_bundle[0])
    assert sana_720p_asset.repo_id.endswith("SANA-Video_2B_720p_diffusers")
    assert sana_720p_asset.snapshot is True
    wan_snapshot = tmp_path / "wan-ti2v-diffusers"
    wan_snapshot.mkdir()
    (wan_snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    assert recommended_setup_bundle_key({
        "architecture": "wan",
        "title": "Wan 2.2 TI2V 5B Diffusers",
        "path": str(wan_snapshot),
    }) == "wan-ti2v-diffusers"
    assert recommended_setup_bundle_key({
        "architecture": "wan",
        "filename": "wan2.2_ti2v_5b_fp16.safetensors",
        "path": str(tmp_path / "wan2.2_ti2v_5b_fp16.safetensors"),
    }) == "wan-ti2v-support"
    wan_standalone = setup_route_descriptor({
        "architecture": "wan",
        "filename": "wan2.2_ti2v_5b_fp16.safetensors",
        "path": str(tmp_path / "wan2.2_ti2v_5b_fp16.safetensors"),
    })
    assert wan_standalone is not None
    assert wan_standalone["routeKey"] == "pro.video.wan.ti2v-standalone-safetensors"
    assert wan_standalone["setupBundleKey"] == "wan-ti2v-support"
    assert "shared UMT5/text stack and the required Wan 2.2 48-channel VAE" in wan_standalone["limitation"]
    assert "does not download or replace it" in wan_standalone["limitation"]
    assert QUICK_START_BUNDLES[wan_standalone["setupBundleKey"]] == ["wan-vae-22", "wan-ti2v-components"]
    assert "wan-ti2v-diffusers-5b" not in QUICK_START_BUNDLES[wan_standalone["setupBundleKey"]]
    wan_ti2v_vae = next(entry for entry in MODEL_DOWNLOAD_CATALOG if entry.key == "wan-vae-22")
    assert wan_ti2v_vae.filename.endswith("wan2.2_vae.safetensors")
    assert "48-channel" in wan_ti2v_vae.title
    assert recommended_setup_bundle_key({
        "architecture": "wan",
        "title": "Wan 2.2 I2V High Noise",
        "filename": "Wan2.2-I2V-A14B-480P-HighNoise-Q4_K_M.gguf",
        "path": str(tmp_path / "Wan2.2-I2V-A14B-480P-HighNoise-Q4_K_M.gguf"),
        }) == "wan-14b-components"
    assert recommended_setup_bundle_key({
        "architecture": "wan",
        "filename": "wan1.3_unknown.gguf",
        "path": str(tmp_path / "wan1.3_unknown.gguf"),
    }) is None
    assert recommended_setup_bundle_key({
        "architecture": "wan",
        "title": "Wan 2.2 I2V A14B High Noise FP8",
        "path": str(tmp_path / "wan-high_noise_fp8.safetensors"),
    }) == "wan-14b-components"
    assert recommended_setup_bundle_key({
        "architecture": "wan",
        "title": "Wan 2.2 I2V A14B High Noise FP8",
        "path": str(tmp_path / "wan-high_noise_fp8.gguf"),
    }) is None
    assert recommended_setup_bundle_key({
        "architecture": "wan",
        "title": "Wan 2.2 I2V A14B High Noise Q5_K_M",
        "filename": "wan2.2_i2v_high_noise_14B_Q5_K_M.gguf",
        "path": str(tmp_path / "wan2.2_i2v_high_noise_14B_Q5_K_M.gguf"),
        "route_key": "pro.video.wan.gguf-pair",
    }) == "wan-14b-components"
    assert recommended_setup_bundle_key({
        "architecture": "wan",
        "filename": "Wan2.2-TI2V-A14B-HighNoise-Q4_K_M.gguf",
        "path": str(tmp_path / "Wan2.2-TI2V-A14B-HighNoise-Q4_K_M.gguf"),
    }) is None


def test_audio_setup_actions_are_represented_without_catalog_bundles():
    audio_routes = {route.route_key: route for route in MODEL_SETUP_ROUTES if route.modality == "audio"}

    assert audio_routes["pro.audio.minimum"].setup_action == "POST /api/pro/audio/setup/minimum"
    assert audio_routes["pro.audio.musicgen.medium"].setup_action == "POST /api/pro/audio/setup/musicgen/medium"
    assert audio_routes["pro.audio.mmaudio.large-44k-v2"].setup_action == "POST /api/pro/audio/setup/mmaudio/large_44k_v2"
    assert all(route.bundle_key is None for route in audio_routes.values())


def test_wan_fp8_pair_bundle_uses_shared_text_stack_and_a14b_vae():
    from aiwf.services.model_download_catalog import MODEL_DOWNLOAD_CATALOG, QUICK_START_BUNDLES

    route = next(route for route in MODEL_SETUP_ROUTES if route.route_key == "pro.video.wan.fp8-pair")
    bundle = QUICK_START_BUNDLES[route.bundle_key]
    entries = {item.key: item for item in MODEL_DOWNLOAD_CATALOG if item.key in bundle}

    assert set(bundle) == {"wan-ti2v-components", "wan-vae-21"}
    assert "14B high/low route uses this same shared text stack" in entries["wan-ti2v-components"].notes
    assert "wan_2.1_vae.safetensors" in entries["wan-vae-21"].filename


def test_standalone_wan_ti2v_bundle_places_the_vae_selected_by_fast_5b(tmp_path):
    from aiwf.core.config.settings import RuntimeFlags, UserSettings
    from aiwf.services.model_download import ModelDownloadService
    from aiwf.services.wan import WanService

    bundle = QUICK_START_BUNDLES["wan-ti2v-support"]
    assert set(bundle) == {"wan-vae-22", "wan-ti2v-components"}
    catalog = {entry.key: entry for entry in MODEL_DOWNLOAD_CATALOG}
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    downloads = ModelDownloadService(flags)
    vae_entry = catalog["wan-vae-22"]
    installed_vae = downloads.destination_for(vae_entry.category, vae_entry.filename)
    assert installed_vae == flags.resolved_models_dir() / "VAE" / "wan2.2_vae.safetensors"

    components_entry = catalog["wan-ti2v-components"]
    components = downloads.snapshot_destination_for(components_entry.category, components_entry.repo_id)
    assert components == flags.resolved_models_dir() / "wan" / "Diffusers" / "Wan2.2-TI2V-5B-Diffusers"
    (components / "text_encoder").mkdir(parents=True)
    (components / "tokenizer").mkdir()
    (components / "scheduler").mkdir()
    (components / "model_index.json").write_text('{"_class_name":"WanPipeline"}', encoding="utf-8")
    (components / "text_encoder" / "config.json").write_text('{"hidden_size":8}', encoding="utf-8")
    (components / "text_encoder" / "model.safetensors").write_bytes(b"fixture encoder")
    (components / "tokenizer" / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    (components / "scheduler" / "scheduler_config.json").write_text('{"_class_name":"FlowMatchEulerDiscreteScheduler"}', encoding="utf-8")

    installed_vae.parent.mkdir(parents=True, exist_ok=True)
    installed_vae.write_bytes(b"fixture VAE")
    wan = WanService(flags, UserSettings())
    assert wan.find_components_base() == str(components.resolve())
    assert wan.preferred_vae("fast_5b") == "wan2.2_vae.safetensors"


def test_ltx_general_picker_route_is_registered_as_supported():
    route = next(route for route in MODEL_SETUP_ROUTES if route.route_key == "pro.video.ltx")

    assert route.support_state == "supported-when-folder-installed"
    assert route.bundle_key is None
    assert "exact selected pipeline" in route.limitation
    assert "no single install bundle" in route.limitation
    assert "one-stage" not in route.limitation
    assert recommended_setup_bundle_key({"architecture": "ltx", "title": "LTX Video"}) is None


def test_ltx_dedicated_pipeline_variants_select_exact_setup_bundles():
    assert recommended_setup_bundle_key({"architecture": "ltx", "pipeline": "diffusers_2b"}) == "ltx-2b"
    assert recommended_setup_bundle_key({"architecture": "ltx", "pipeline": "distilled"}) == "ltx23"
    assert recommended_setup_bundle_key({"architecture": "ltx", "pipeline": "one_stage"}) == "ltx23-one-stage"
    fp8_path = "F:/models/ltx/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"
    assert recommended_setup_bundle_key({
        "architecture": "ltx", "pipeline": "one_stage", "path": fp8_path,
    }) == "ltx23-one-stage-fp8"

    routes = {route.route_key: route for route in MODEL_SETUP_ROUTES}
    one_stage = routes["pro.video.ltx.one-stage"]
    assert one_stage.support_state == "blocked-runtime"
    assert "access violation 3221225477" in one_stage.limitation
    assert "stage files" in one_stage.limitation


def test_setup_available_or_settings_configured_routes_expose_their_real_actions():
    routes = {route.route_key: route for route in MODEL_SETUP_ROUTES}
    nunchaku = routes["pro.image.qwen-nunchaku"]
    assert nunchaku.support_state == "setup-available"
    assert nunchaku.bundle_key == "qwen-nunchaku"
    assert nunchaku.setup_action == "POST /api/pro/engines/qwen_nunchaku/install"
    assert recommended_setup_bundle_key({"architecture": "qwen_image_nunchaku", "title": "Qwen Nunchaku"}) == "qwen-nunchaku"
    assert routes["pro.image.anima"].support_state == "unsupported"
    assert routes["pro.image.onnx"].setup_action.startswith("Configure Settings.onnx_model_dir")
    assert recommended_setup_bundle_key({"architecture": "anima", "title": "Anima"}) is None


def test_setup_route_descriptor_exposes_safe_route_and_support_contract():
    descriptor = setup_route_descriptor({
        "architecture": "flux",
        "title": "Flux Dev",
        "path": "F:/models/flux-dev.safetensors",
    })

    assert descriptor == {
        "routeKey": "pro.image.flux-teacher",
        "modality": "image",
        "supportState": "supported",
        "preflightKey": "flux_conditioning",
        "setupBundleKey": "flux-components",
        "setupAction": None,
        "limitation": None,
    }
    assert "path" not in descriptor
    assert setup_route_descriptor({"architecture": "unknown", "title": "Mystery"}) is None


def test_krea2_setup_descriptor_distinguishes_folder_route_from_split_assets():
    descriptor = setup_route_descriptor({"architecture": "krea2", "title": "Krea 2 Turbo"})
    raw_descriptor = setup_route_descriptor({"architecture": "krea2", "title": "Krea 2 Raw"})

    assert descriptor is not None
    assert descriptor["supportState"] == "runtime-dependent"
    assert descriptor["setupBundleKey"] == "krea2"
    assert raw_descriptor is not None
    assert raw_descriptor["supportState"] == "runtime-dependent"
    assert "only when the installed runtime exposes Krea2Pipeline" in descriptor["limitation"]
    assert "split-file Krea 2 assets remain unsupported" in descriptor["limitation"]


@pytest.mark.parametrize(
    ("architecture", "folder", "title", "expected_bundle"),
    [
        ("krea2", "Krea-2-Raw", "Krea 2 Raw", "krea2-raw"),
        ("krea2", "Krea-2-Turbo", "Krea 2 Turbo", "krea2"),
        ("qwen_image", "Qwen-Image", "Qwen Image", "qwen-image-original"),
        ("qwen_image", "Qwen-Image-2512", "Qwen Image 2512", "qwen-image"),
    ],
)
def test_variant_folder_setup_routes_recommend_matching_bundles(
    tmp_path: Path, architecture: str, folder: str, title: str, expected_bundle: str
):
    snapshot = tmp_path / folder
    snapshot.mkdir()
    (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    assert recommended_setup_bundle_key({
        "architecture": architecture,
        "title": title,
        "path": str(snapshot),
    }) == expected_bundle


def test_krea_split_file_does_not_recommend_incompatible_diffusers_folder_bundle(tmp_path):
    split = tmp_path / "krea2_raw_fp8_scaled.safetensors"
    split.write_bytes(b"split")
    assert recommended_setup_bundle_key({
        "architecture": "krea2", "title": "Krea 2 Raw FP8", "path": str(split),
    }) is None


@pytest.mark.parametrize(
    ("architecture", "title", "expected_bundle"),
    [
        ("krea2", "Krea 2 Raw", "krea2-raw"),
        ("qwen_image", "Qwen Image", "qwen-image-original"),
        ("qwen_image", "Qwen Image 2512", "qwen-image"),
    ],
)
def test_incomplete_variant_names_still_recommend_the_matching_bundle(architecture, title, expected_bundle):
    assert recommended_setup_bundle_key({"architecture": architecture, "title": title}) == expected_bundle


def test_sana_variants_recommend_matching_install_bundles():
    assert recommended_setup_bundle_key({
        "architecture": "sana", "title": "Sana Sprint 0.6B 1024px",
    }) == "sana"
    assert recommended_setup_bundle_key({
        "architecture": "sana", "title": "Sana Sprint 1.6B 1024px",
    }) == "sana-sprint-16b"
    assert recommended_setup_bundle_key({
        "architecture": "sana", "title": "Sana 1.6B 1024px BF16",
    }) == "sana-16b"


def test_setup_route_descriptor_accepts_explicit_routes_without_catalog_identity():
    descriptor = setup_route_descriptor({"routeKey": "pro.audio.musicgen.medium"})
    assert descriptor is not None
    assert descriptor["modality"] == "audio"
    assert descriptor["preflightKey"] == "musicgen"
    assert descriptor["setupAction"] == "POST /api/pro/audio/setup/musicgen/medium"


@pytest.mark.parametrize(
    ("architecture", "title", "bundle"),
    [
        ("ltx", "LTX distilled", "ltx23"),
        ("ltx", "LTX one_stage", "ltx23-one-stage"),
    ],
)
def test_ltx_worker_setup_is_a_structured_route_action(architecture, title, bundle):
    descriptor = setup_route_descriptor({"architecture": architecture, "title": title})

    assert descriptor is not None
    assert descriptor["setupBundleKey"] == bundle
    assert descriptor["setupAction"] == "POST /api/pro/engines/ltx/install"


def test_ltx_one_stage_support_label_tracks_selected_checkpoint_precision(tmp_path: Path):
    fp8 = setup_route_descriptor({
        "routeKey": "pro.video.ltx.one-stage",
        "path": str(tmp_path / "ltx-2.3-22b-dev-fp8.safetensors"),
    })
    bf16 = setup_route_descriptor({
        "routeKey": "pro.video.ltx.one-stage",
        "path": str(tmp_path / "ltx-2.3-22b-dev.safetensors"),
    })

    assert fp8 is not None and fp8["supportState"] == "supported-when-folder-installed"
    assert fp8["setupBundleKey"] == "ltx23-one-stage-fp8"
    assert recommended_setup_bundle_key({
        "routeKey": "pro.video.ltx.one-stage", "path": str(tmp_path / "ltx-2.3-22b-dev-fp8.safetensors"),
    }) == "ltx23-one-stage-fp8"
    assert "selected FP8" in fp8["limitation"]
    assert bf16 is not None and bf16["supportState"] == "blocked-runtime"
    assert "22B BF16 one-stage worker is blocked on Windows" in bf16["limitation"]


def test_ltx_2b_setup_descriptor_discloses_text_only_generation():
    descriptor = setup_route_descriptor({
        "architecture": "ltx",
        "title": "LTX 0.9.5 Diffusers 2B",
    })

    assert descriptor is not None
    assert "text-to-video only" in descriptor["limitation"]
