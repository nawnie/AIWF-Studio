from __future__ import annotations

from uuid import uuid4

from PIL import Image

from aiwf.core.domain.generation import GenerationMode, GenerationRequest, GenerationResult, JobRecord, JobState
from aiwf.services.image_lab import (
    ImageLabService,
    build_plot_axes,
    image_maturity_matrix,
    parse_axis_values,
    variation_requests,
)


class FakeGeneration:
    def __init__(self):
        self.submitted = []

    def submit(self, request, init_images=None, mask_images=None):
        self.submitted.append((request, init_images, mask_images))
        image = Image.new("RGB", (16, 16), (len(self.submitted) * 20, 0, 0))
        return JobRecord(
            request=request,
            state=JobState.COMPLETED,
            result=GenerationResult(
                job_id=uuid4(),
                images=[image],
                seeds=[request.seed],
                infotexts=[f"seed={request.seed}"],
                mode=request.mode,
            ),
        )


def test_maturity_matrix_has_ready_core_image_routes():
    matrix = image_maturity_matrix()
    routes = {route.route: route for route in matrix.routes}

    for route in ("txt2img", "img2img", "inpaint", "hires-refiner", "controlnet", "xyz-plot"):
        assert routes[route].score >= matrix.target_score
        assert routes[route].benchmark_kind

    assert routes["segment-inpaint"].score >= matrix.target_score
    assert routes["segment-inpaint"].status == "ready"
    assert routes["flux-txt2img"].score >= matrix.target_score
    assert routes["flux-txt2img"].status == "ready"
    assert "txt2img-only" in routes["flux-txt2img"].notes[0]


def test_axis_parser_normalizes_aliases_and_values():
    assert parse_axis_values("1, 2.5, true, euler_a") == [1, 2.5, True, "euler_a"]

    axes = build_plot_axes((("cfg", "6, 7"), ("denoise", "0.4;0.6"), ("", "")))

    assert [axis.field for axis in axes] == ["cfg_scale", "denoising_strength"]
    assert axes[0].values == [6, 7]
    assert axes[1].values == [0.4, 0.6]


def test_variation_requests_increment_seed_and_denoise():
    base = GenerationRequest(seed=10, denoising_strength=0.25)

    requests = variation_requests(base, count=3, seed_step=5, denoise_step=0.1)

    assert [request.seed for request in requests] == [10, 15, 20]
    assert [request.denoising_strength for request in requests] == [0.25, 0.35, 0.45]


def test_batch_inpaint_preserves_order_and_mask_pairing():
    generation = FakeGeneration()
    service = ImageLabService(generation)
    request = GenerationRequest(mode=GenerationMode.INPAINT, seed=77)
    images = [Image.new("RGB", (16, 16), "white"), Image.new("RGB", (16, 16), "blue")]
    masks = [Image.new("L", (16, 16), 255)]

    result = service.run_batch(request, images, masks=masks)

    assert len(result.images) == 2
    assert result.labels == ["1/2", "2/2"]
    assert result.seeds == [77, 77]
    assert result.grid is not None
    assert generation.submitted[0][1] == [images[0]]
    assert generation.submitted[1][1] == [images[1]]
    assert generation.submitted[0][2] == [masks[0]]
    assert generation.submitted[1][2] == [masks[0]]


def test_loopback_chains_previous_output_and_increments_seed():
    generation = FakeGeneration()
    service = ImageLabService(generation)
    request = GenerationRequest(mode=GenerationMode.IMG2IMG, seed=5, denoising_strength=0.8)
    source = Image.new("RGB", (16, 16), "white")

    result = service.run_loopback(request, source, iterations=3, denoise_decay=0.5, seed_mode="increment")

    assert len(result.images) == 3
    assert [call[0].seed for call in generation.submitted] == [5, 6, 7]
    assert [call[0].denoising_strength for call in generation.submitted] == [0.8, 0.4, 0.2]
    assert generation.submitted[0][1] == [source]
    assert generation.submitted[1][1] == [result.images[0]]
    assert generation.submitted[2][1] == [result.images[1]]
