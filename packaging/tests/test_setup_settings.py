"""Verify installer backend choices preserve the rest of an existing AES profile."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


MODULE = Path(__file__).resolve().parents[1] / "setup_settings.py"
spec = importlib.util.spec_from_file_location("aes_setup_settings", MODULE)
setup_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_settings)


class SetupSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix="aes-setup-settings-", dir=os.environ.get("AES_STUDIO_TEST_TMP") or None
        )
        self.addCleanup(self.temp.cleanup)
        self.profile = Path(self.temp.name)
        self.settings = self.profile / "settings.json"

    def test_new_profile_honors_each_backend_choice(self):
        for backend, enabled, selected in (
            ("cpp", ["cpp"], "cpp"),
            ("both", ["python", "cpp"], "cpp"),
        ):
            with self.subTest(backend=backend):
                profile = self.profile / backend
                profile.mkdir()
                self.assertEqual(setup_settings.configure_backend(profile, backend, "preserve"), "created")
                data = json.loads((profile / "settings.json").read_text(encoding="utf-8"))
                self.assertEqual(data["enabled_image_backends"], enabled)
                self.assertEqual(data["image_backend_id"], selected)
                self.assertFalse(data["setup_completed"])

    def test_installer_rejects_python_only_choice_without_modifying_profile(self):
        original = json.dumps({"schema": setup_settings.SCHEMA, "user_data": "kept"}).encode()
        self.settings.write_bytes(original)
        with self.assertRaises(ValueError):
            setup_settings.configure_backend(self.profile, "python", "apply")
        self.assertEqual(self.settings.read_bytes(), original)

    def test_interactive_reinstall_changes_only_backend_fields(self):
        original = {"schema": setup_settings.SCHEMA, "setup_completed": True,
                    "enabled_image_backends": ["python"], "image_backend_id": "python",
                    "user_data": {"name": "kept", "value": 42}}
        self.settings.write_text(json.dumps(original), encoding="utf-8")
        self.assertEqual(setup_settings.configure_backend(self.profile, "both", "apply"), "updated")
        actual = json.loads(self.settings.read_text(encoding="utf-8"))
        expected = dict(original, enabled_image_backends=["python", "cpp"], image_backend_id="cpp")
        self.assertEqual(actual, expected)

    def test_silent_reinstall_keeps_existing_bytes_even_if_invalid(self):
        original = b"not a settings document\r\n"
        self.settings.write_bytes(original)
        self.assertEqual(setup_settings.configure_backend(self.profile, "cpp", "preserve"), "preserved")
        self.assertEqual(self.settings.read_bytes(), original)

    def test_interactive_reinstall_rejects_invalid_or_unrecognized_settings(self):
        for original in (b"{broken", b"[]", b'{"schema":"different"}'):
            with self.subTest(original=original):
                self.settings.write_bytes(original)
                with self.assertRaises(ValueError):
                    setup_settings.configure_backend(self.profile, "cpp", "apply")
                self.assertEqual(self.settings.read_bytes(), original)
                self.assertEqual(list(self.profile.glob(".aes-setup-*.tmp")), [])

    def test_failed_replace_keeps_original_and_removes_temporary_file(self):
        original = json.dumps({"schema": setup_settings.SCHEMA, "user_data": "keep"}).encode()
        self.settings.write_bytes(original)
        with mock.patch.object(setup_settings.os, "replace", side_effect=OSError("simulated failure")):
            with self.assertRaises(OSError):
                setup_settings.configure_backend(self.profile, "cpp", "apply")
        self.assertEqual(self.settings.read_bytes(), original)
        self.assertEqual(list(self.profile.glob(".aes-setup-*.tmp")), [])

    def test_concurrent_profile_change_is_not_overwritten(self):
        original = json.dumps({"schema": setup_settings.SCHEMA, "user_data": "before"}).encode()
        later = json.dumps({"schema": setup_settings.SCHEMA, "user_data": "later"}).encode()
        self.settings.write_bytes(original)
        with mock.patch.object(setup_settings.os, "fsync", side_effect=lambda fd: self.settings.write_bytes(later)):
            with self.assertRaisesRegex(ValueError, "changed during installation"):
                setup_settings.configure_backend(self.profile, "cpp", "apply")
        self.assertEqual(self.settings.read_bytes(), later)
        self.assertEqual(list(self.profile.glob(".aes-setup-*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
