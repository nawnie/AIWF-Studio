"""Keep the optional Python backend compatible with the bundled API lock."""
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DependencyLockTests(unittest.TestCase):
    def test_optional_backend_resolves_every_bundled_api_version(self):
        requirements = ROOT / "native" / "AIWF.Desktop" / "engine-api-requirements.txt"
        lock = tomllib.loads((ROOT / "dependencies" / "windows-py312" / "uv.lock").read_text(encoding="utf-8"))
        versions = {}
        for package in lock["package"]:
            name = package["name"].lower().replace("_", "-")
            versions.setdefault(name, set()).add(package["version"])
        for raw in requirements.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            name, separator, version = line.partition("==")
            self.assertEqual(separator, "==", line)
            with self.subTest(package=name):
                self.assertIn(version, versions.get(name.lower().replace("_", "-"), set()))


if __name__ == "__main__":
    unittest.main()
