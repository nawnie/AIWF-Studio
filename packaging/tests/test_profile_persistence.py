"""Exercise AES folder settings and project data across installation roots."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from aiwf.core.config.launch import launch_settings_path
from aiwf.services.setup_settings import describe_setup, save_folders
from aiwf.services.unified_bridge import BridgeConfig, ProjectLedger

MODULE = Path(__file__).resolve().parents[1] / "setup_settings.py"
spec = importlib.util.spec_from_file_location("aes_profile_setup", MODULE)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class ProfilePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix="aes-profile-persistence-", dir=os.environ.get("AES_STUDIO_TEST_TMP") or None
        )
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.old = self.base / "old install"
        self.new = self.base / "new install"
        self.profile = self.base / "AES profile"
        for folder in (self.old, self.new, self.profile):
            folder.mkdir()

    def profile_environment(self):
        return mock.patch.dict(os.environ, {"AES_STUDIO_USE_PROFILE_DATA": "1",
                                           "AIWF_STUDIO_DATA_DIR": str(self.profile)})

    def test_legacy_mode_keeps_install_local_paths(self):
        with mock.patch.dict(os.environ, {"AES_STUDIO_USE_PROFILE_DATA": "0",
                                          "AIWF_STUDIO_DATA_DIR": str(self.profile)}):
            self.assertEqual(launch_settings_path(self.old), self.old / "launch.json")
            self.assertEqual(BridgeConfig.from_environment(self.old).projects_dir,
                             self.old / "_local" / "unified-projects")

    def test_folders_and_projects_survive_a_new_install_root(self):
        installer.ensure_profile_launch(self.profile)
        models = self.base / "custom models"
        outputs = self.base / "custom outputs"
        with self.profile_environment():
            saved = save_folders(self.old, {"models_dir": str(models),
                                            "ckpt_dir": str(models / "Stable-diffusion"),
                                            "output_dir": str(outputs)}, create_missing=True)
            self.assertTrue(saved["launch_file_exists"])
            first = ProjectLedger(BridgeConfig.from_environment(self.old).projects_dir)
            project = first.create("Persistent test project")
            after = describe_setup(self.new)
            second = ProjectLedger(BridgeConfig.from_environment(self.new).projects_dir)
            self.assertEqual(Path(after["launch_file"]), self.profile / "launch.json")
            self.assertEqual(Path(after["folders"]["models_dir"]["saved"]), models)
            self.assertEqual(Path(after["folders"]["output_dir"]["saved"]), outputs)
            self.assertEqual(second.list()[0]["project_id"], project["project_id"])
            self.assertFalse((self.old / "launch.json").exists())
            self.assertFalse((self.new / "_local" / "unified-projects").exists())

    def test_profile_mode_rejects_missing_or_relative_profile(self):
        for value in ("", "relative-path"):
            with self.subTest(value=value), mock.patch.dict(os.environ,
                    {"AES_STUDIO_USE_PROFILE_DATA": "1", "AIWF_STUDIO_DATA_DIR": value}):
                with self.assertRaises(ValueError):
                    launch_settings_path(self.old)

    def test_legacy_migration_copies_settings_and_projects_without_deleting_source(self):
        launch = b'{"models_dir":"custom"}\n'
        record = b'{"project_id":"aiwfp-0123456789abcdef","name":"test"}\n'
        (self.old / "launch.json").write_bytes(launch)
        source = self.old / "_local" / "unified-projects"
        source.mkdir(parents=True)
        (source / "aiwfp-0123456789abcdef.json").write_bytes(record)
        self.assertEqual(installer.migrate_legacy_profile(self.old, self.profile), 2)
        self.assertEqual((self.profile / "launch.json").read_bytes(), launch)
        self.assertEqual((self.profile / "_local" / "unified-projects" /
                          "aiwfp-0123456789abcdef.json").read_bytes(), record)
        self.assertEqual((self.old / "launch.json").read_bytes(), launch)
        self.assertEqual(installer.migrate_legacy_profile(self.old, self.profile), 0)

    def test_migration_conflict_does_not_replace_existing_profile(self):
        (self.old / "launch.json").write_bytes(b'{"models_dir":"old"}')
        profile_launch = self.profile / "launch.json"
        profile_launch.write_bytes(b'{"models_dir":"new"}')
        source = self.old / "_local" / "unified-projects"
        source.mkdir(parents=True)
        (source / "aiwfp-0123456789abcdef.json").write_bytes(b"test")
        with self.assertRaisesRegex(ValueError, "both copies were preserved"):
            installer.migrate_legacy_profile(self.old, self.profile)
        self.assertEqual(profile_launch.read_bytes(), b'{"models_dir":"new"}')
        self.assertFalse((self.profile / "_local" / "unified-projects").exists())

    def test_existing_profile_launch_is_not_replaced_with_defaults(self):
        launch = self.profile / "launch.json"
        original = b'{"models_dir":"user choice"}\n'
        launch.write_bytes(original)
        self.assertEqual(installer.ensure_profile_launch(self.profile), "preserved")
        self.assertEqual(launch.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
