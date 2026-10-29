"""Check that installer staging rejects absent or altered offline wheels."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "stage_payload.py"
SPEC = importlib.util.spec_from_file_location("aes_stage_payload", MODULE_PATH)
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


class WheelValidationTests(unittest.TestCase):
    def test_helper_architecture_must_match(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("AES_STUDIO_TEST_TMP") or None) as directory:
            path = Path(directory) / "helper.dll"
            dos = bytearray(64)
            dos[:2] = b"MZ"
            struct.pack_into("<I", dos, 0x3C, 64)
            path.write_bytes(dos + b"PE\0\0" + struct.pack("<H", 0x014c))
            stage.require_pe_machine(path, 0x014c, "NSIS bridge")
            with self.assertRaisesRegex(SystemExit, "architecture"):
                stage.require_pe_machine(path, 0x8664, "x64 helper")

    def test_locked_wheel_is_required_and_verified(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("AES_STUDIO_TEST_TMP") or None) as directory:
            root = Path(directory)
            packaging = root / "packaging"
            native = root / "native" / "AIWF.Desktop"
            wheels = root / "wheels"
            for folder in (packaging, native, wheels):
                folder.mkdir(parents=True)
            (native / "engine-api-requirements.txt").write_text("example==1.0\n", encoding="utf-8")
            wheel = wheels / "example-1.0-py3-none-any.whl"
            wheel.write_bytes(b"verified wheel")
            (packaging / "wheel-provenance.json").write_text(json.dumps({
                "schema": "aes-wheel-provenance-v1", "files": [{
                    "filename": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()
                }]
            }), encoding="utf-8")
            with mock.patch.object(stage, "ROOT", root):
                self.assertEqual(stage.verified_wheels(wheels), [wheel])
                wheel.write_bytes(b"altered wheel")
                with self.assertRaisesRegex(SystemExit, "SHA-256 mismatch"):
                    stage.verified_wheels(wheels)
                wheel.unlink()
                with self.assertRaisesRegex(SystemExit, "missing"):
                    stage.verified_wheels(wheels)


if __name__ == "__main__":
    unittest.main()
