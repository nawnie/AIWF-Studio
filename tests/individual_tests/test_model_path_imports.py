from __future__ import annotations

from pathlib import Path

from aiwf.services.model_path_imports import (
    ImportedModelPaths,
    import_automatic1111_paths,
    import_comfyui_paths,
    merge_imported_path_text,
)


def test_import_automatic1111_paths_adds_models_and_embeddings(tmp_path: Path):
    root = tmp_path / "stable-diffusion-webui"
    (root / "models" / "Stable-diffusion").mkdir(parents=True)
    (root / "embeddings").mkdir(parents=True)

    imported = import_automatic1111_paths(str(root))

    assert (root / "models").resolve() in imported.extra_model_dirs
    assert (root / "embeddings").resolve() in imported.extra_model_dirs
    assert (root / "models" / "Stable-diffusion").resolve() in imported.extra_ckpt_dirs


def test_import_comfyui_paths_adds_models_and_checkpoints(tmp_path: Path):
    root = tmp_path / "ComfyUI"
    (root / "models" / "checkpoints").mkdir(parents=True)

    imported = import_comfyui_paths(str(root))

    assert imported.extra_model_dirs == [(root / "models").resolve()]
    assert imported.extra_ckpt_dirs == [(root / "models" / "checkpoints").resolve()]


def test_merge_imported_path_text_deduplicates_existing_entries(tmp_path: Path):
    existing_models = str((tmp_path / "models").resolve())
    existing_ckpts = str((tmp_path / "ckpts").resolve())
    (tmp_path / "models").mkdir()
    (tmp_path / "ckpts").mkdir()
    (tmp_path / "embeddings").mkdir()

    imported_models = [(tmp_path / "models").resolve(), (tmp_path / "embeddings").resolve()]
    imported_ckpts = [(tmp_path / "ckpts").resolve()]

    model_text, ckpt_text = merge_imported_path_text(
        existing_models,
        existing_ckpts,
        ImportedModelPaths(
            source="test",
            root=tmp_path,
            extra_model_dirs=imported_models,
            extra_ckpt_dirs=imported_ckpts,
            summary="",
        ),
    )

    assert model_text.splitlines() == [str((tmp_path / "models").resolve()), str((tmp_path / "embeddings").resolve())]
    assert ckpt_text.splitlines() == [str((tmp_path / "ckpts").resolve())]
