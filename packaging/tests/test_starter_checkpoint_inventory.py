"""Check that first-run model selection excludes interrupted checkpoint files."""
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.diffusers.checkpoints import scan_from_flags
from aiwf.infrastructure.model_inventory import get_model_inventory, invalidate_model_inventory_cache
from aiwf.services.native_image_backends import _aiwf_image_checkpoints, _supported_checkpoints


def miniature_sdxl_header():
    """A tiny structural fixture, not a model suitable for generation."""
    tensors = {
        "model.diffusion_model.input_blocks.0.0.weight": {
            "dtype": "U8", "shape": [1, 4, 1, 1], "data_offsets": [0, 4]
        },
        "conditioner.embedders.1.model.ln_final.weight": {
            "dtype": "U8", "shape": [1], "data_offsets": [4, 5]
        },
    }
    header = json.dumps(tensors).encode("utf-8")
    return struct.pack("<Q", len(header)) + header


class StarterCheckpointInventoryTests(unittest.TestCase):
    def test_only_complete_checkpoint_reaches_cpp_model_picker(self):
        with tempfile.TemporaryDirectory(
            prefix="aes-checkpoint-inventory-", dir=os.environ.get("AES_STUDIO_TEST_TMP") or None
        ) as folder:
            root = Path(folder)
            checkpoints = root / "models" / "Stable-diffusion"
            checkpoints.mkdir(parents=True)
            (checkpoints / "complete.safetensors").write_bytes(miniature_sdxl_header() + bytes(5))
            (checkpoints / "incomplete.safetensors").write_bytes(miniature_sdxl_header())
            flags = RuntimeFlags(
                _env_file=None, data_dir=root, models_dir=root / "models", ckpt_dir=checkpoints,
                output_dir=root / "outputs", extra_model_dirs=[], extra_ckpt_dirs=[]
            )
            try:
                inventory = get_model_inventory(flags, force_rescan=True, allow_torch_load=False)
                rows = {item.filename: item.family for item in inventory}
                self.assertEqual(rows.get("complete.safetensors"), "checkpoint")
                self.assertEqual(rows.get("incomplete.safetensors"), "invalid_asset")
                scanned = scan_from_flags(flags, allow_torch_load=False)

                class Backend:
                    def list_checkpoints(self):
                        return scanned

                eligible = _supported_checkpoints(Backend())
                self.assertEqual([(item.filename, item.architecture) for item in eligible],
                                 [("complete.safetensors", "sdxl")])
            finally:
                invalidate_model_inventory_cache()

    def test_cpp_picker_accepts_recognized_formats_without_loading_weights(self):
        with tempfile.TemporaryDirectory(prefix='aes-checkpoint-formats-',dir=os.environ.get('AES_STUDIO_TEST_TMP') or None) as folder:
            root=Path(folder); checkpoints=root/'models'/'Stable-diffusion'; checkpoints.mkdir(parents=True)
            (checkpoints/'sdxl-safe.safetensors').write_bytes(miniature_sdxl_header()+bytes(5))
            for ext in ('.ckpt','.pt','.pth'):
                (checkpoints/('sdxl-legacy'+ext)).write_bytes(b'synthetic legacy fixture')
            (checkpoints/'sdxl-quant.gguf').write_bytes(struct.pack('<4sIQQ',b'GGUF',3,1,0)+b'x')
            (checkpoints/'sdxl-broken.gguf').write_bytes(b'not a GGUF')
            with zipfile.ZipFile(checkpoints/'sdxl-script.pth','w') as archive:
                archive.writestr('script/code/__torch__/module.py','pass')
            flags=RuntimeFlags(_env_file=None,data_dir=root,models_dir=root/'models',ckpt_dir=checkpoints,
                               output_dir=root/'outputs',extra_model_dirs=[],extra_ckpt_dirs=[])
            try:
                inventory=get_model_inventory(flags,force_rescan=True,allow_torch_load=False)
                self.assertEqual(next(x.family for x in inventory if x.filename=='sdxl-broken.gguf'),'invalid_asset')
                scanned=scan_from_flags(flags,allow_torch_load=False)
                class Backend:
                    def list_checkpoints(self): return scanned
                names={item.filename for item in _supported_checkpoints(Backend())}
                self.assertEqual(names,{'sdxl-safe.safetensors','sdxl-legacy.ckpt','sdxl-legacy.pt','sdxl-legacy.pth','sdxl-quant.gguf'})
                python_names={Path(checkpoint.path).name for _,checkpoint in _aiwf_image_checkpoints(flags,'python')}
                self.assertNotIn('sdxl-quant.gguf',python_names)
                self.assertIn('sdxl-legacy.pth',python_names)
            finally:
                invalidate_model_inventory_cache()


if __name__ == "__main__":
    unittest.main()
