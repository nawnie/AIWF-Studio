"""A file-discovery check must not promise that generation has run."""
import os
from pathlib import Path
import tempfile
import unittest

from aiwf.core.domain.models import Checkpoint
from aiwf.services.native_image_backends import StableDiffusionCppImageJobs


class CppReadinessTests(unittest.TestCase):
    def test_found_model_reports_preliminary_checks_not_generation(self):
        with tempfile.TemporaryDirectory(
            prefix="aes-readiness-", dir=os.environ.get("AES_STUDIO_TEST_TMP") or None
        ) as folder:
            path = Path(folder) / "sample.safetensors"
            path.write_bytes(b"fixture only; no model weights")
            checkpoint = Checkpoint(id="sample", title="Sample", filename=path.name,
                                    path=str(path), architecture="sdxl")

            class Backend:
                executable = "synthetic-sd-cli"

                def list_checkpoints(self):
                    return [checkpoint]

                def probe_cli(self):
                    return True, ""

            jobs = StableDiffusionCppImageJobs(flags=object(), backend_factory=lambda _: Backend())
            status = jobs.readiness()
            self.assertTrue(status["available"])
            self.assertEqual(status["runtime_info"]["cli_probe"], "passed")
            self.assertEqual(len(status["models"]), 1)
            self.assertIn("does not load a model", status["reason"])
            self.assertIn("driver compatibility", status["reason"])
            self.assertIn("generation", status["reason"])
            self.assertNotIn("Ready for text-to-image", status["reason"])


if __name__ == "__main__":
    unittest.main()
