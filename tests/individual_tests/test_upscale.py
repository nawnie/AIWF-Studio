import torch
from PIL import Image

from aiwf.core.domain.enhance import EnhanceModel, EnhanceModelKind, UpscaleOptions
from aiwf.infrastructure.enhance.upscale import upscale_image


class _DummyUpscaleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class _DummyDescriptor:
    def __init__(self):
        self.model = _DummyUpscaleNet()
        self.device = self.model.weight.device
        self.dtype = self.model.weight.dtype


def test_upscale_image_accepts_spandrel_style_descriptor():
    image = Image.new("RGB", (24, 24), (120, 80, 40))
    descriptor = _DummyDescriptor()
    model_info = EnhanceModel(
        id="dummy-upscale",
        title="Dummy",
        filename="dummy.pth",
        path="dummy.pth",
        kind=EnhanceModelKind.UPSCALER,
        architecture="ESRGAN",
        scale=1,
    )
    options = UpscaleOptions(model_id=model_info.id, scale=1.0, tile_size=0, tile_overlap=0)

    result = upscale_image(image, descriptor, model_info=model_info, options=options)

    assert result.size == image.size
    assert result.getpixel((0, 0)) == image.getpixel((0, 0))
