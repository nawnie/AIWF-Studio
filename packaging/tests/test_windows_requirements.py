"""Exercise the installer's real NSIS prerequisite logic and OS probes."""
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest


COMPILER = os.environ.get("AES_NSIS_COMPILER") or shutil.which("makensis")
REQUIREMENTS = Path(__file__).resolve().parents[1] / "SystemRequirements.nsh"


@unittest.skipUnless(os.name == "nt" and COMPILER, "NSIS compiler required")
class InstallerRequirementsTests(unittest.TestCase):
    def test_requirement_decisions_and_current_windows_probe(self):
        with tempfile.TemporaryDirectory(
            prefix="aes-nsis-requirements-", dir=os.environ.get("AES_STUDIO_TEST_TMP") or None
        ) as folder:
            root = Path(folder)
            exe, script, results = root / "check.exe", root / "check.nsi", root / "results.txt"
            content = r'''Unicode true
RequestExecutionLevel user
ManifestSupportedOS all
!include "LogicLib.nsh"
!include "x64.nsh"
!include "WinVer.nsh"
!include "@REQUIREMENTS@"
OutFile "@EXE@"
Section
  FileOpen $9 "@RESULTS@" w
  !insertmacro AES_RequirementStatus $2 0 19041
  FileWrite $9 "x86=$2$\r$\n"
  !insertmacro AES_RequirementStatus $2 1 19040
  FileWrite $9 "old=$2$\r$\n"
  !insertmacro AES_RequirementStatus $2 1 19041
  FileWrite $9 "minimum=$2$\r$\n"
  !insertmacro AES_RequirementStatus $2 1 26300
  FileWrite $9 "newer=$2$\r$\n"
  StrCpy $0 0
  ${If} ${RunningX64}
    StrCpy $0 1
  ${EndIf}
  ${WinVerGetBuild} $1
  !insertmacro AES_RequirementStatus $2 $0 $1
  FileWrite $9 "current64=$0$\r$\ncurrentbuild=$1$\r$\ncurrent=$2$\r$\n"
  FileClose $9
SectionEnd
'''
            content = content.replace("@REQUIREMENTS@", str(REQUIREMENTS))
            content = content.replace("@EXE@", str(exe)).replace("@RESULTS@", str(results))
            script.write_text(content, encoding="utf-8")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.run([COMPILER, "/V1", str(script)], check=True, capture_output=True, creationflags=flags)
            subprocess.run([str(exe), "/S"], check=True, creationflags=flags)
            observed = dict(line.split("=", 1) for line in results.read_text(encoding="utf-8").splitlines())
            self.assertEqual(observed["x86"], "architecture")
            self.assertEqual(observed["old"], "version")
            self.assertEqual(observed["minimum"], "ok")
            self.assertEqual(observed["newer"], "ok")
            if struct.calcsize("P") == 8:
                self.assertEqual(observed["current64"], "1")
            self.assertEqual(int(observed["currentbuild"]), sys.getwindowsversion().build)
            self.assertEqual(observed["current"], "ok")


if __name__ == "__main__":
    unittest.main()
