import hashlib, os, subprocess, tempfile, unittest
from pathlib import Path
SCRATCH=os.environ.get('AES_STUDIO_TEST_TMP') or None
WORKER=Path(__file__).resolve().parents[1]/'uninstall_worker.exe'
def digest(data): return hashlib.sha256(data).hexdigest()
def manifest(root, rows):
    (root/'owned-files.tsv').write_text('AES-STUDIO-OWNED-V1\n'+''.join(h+'\t'+p+'\n' for p,h in rows),encoding='utf-8',newline='')
class UninstallerSafetyTests(unittest.TestCase):
    def test_owned_only_unchanged_files_are_removed(self):
        with tempfile.TemporaryDirectory(prefix='aes-uninstall-test-',dir=SCRATCH) as folder:
            root=Path(folder); (root/'app').mkdir()
            (root/'app'/'unchanged.bin').write_bytes(b'owned')
            (root/'app'/'changed.bin').write_bytes(b'user modified')
            (root/'app'/'user.txt').write_bytes(b'user file')
            manifest(root,[('app/unchanged.bin',digest(b'owned')),('app/changed.bin',digest(b'original'))])
            pending=root.parent/(root.name+'-pending.txt')
            try:
                r=subprocess.run([str(WORKER),str(root),str(pending)])
                self.assertEqual(r.returncode,0)
                self.assertFalse((root/'app'/'unchanged.bin').exists())
                self.assertEqual((root/'app'/'changed.bin').read_bytes(),b'user modified')
                self.assertEqual((root/'app'/'user.txt').read_bytes(),b'user file')
                self.assertEqual(pending.read_text(encoding='utf-16-le'),'')
            finally: pending.unlink(missing_ok=True)
    def test_unicode_paths_are_removed_correctly(self):
        with tempfile.TemporaryDirectory(prefix='aes-uninstall-test-',dir=SCRATCH) as folder:
            root=Path(folder); (root/'模型').mkdir(); target=root/'模型'/'démo.bin';target.write_bytes(b'data')
            manifest(root,[('模型/démo.bin',digest(b'data'))]); pending=root.parent/(root.name+'-pending.txt')
            try:
                r=subprocess.run([str(WORKER),str(root),str(pending)])
                self.assertEqual(r.returncode,0)
                self.assertFalse(target.exists())
            finally: pending.unlink(missing_ok=True)

    def test_empty_runtime_scaffolding_is_removed(self):
        with tempfile.TemporaryDirectory(prefix='aes-uninstall-empty-',dir=SCRATCH) as folder:
            root=Path(folder)
            for name in ('runtime/python/libs', 'venv/Include', 'venv/Scripts'):
                (root/name).mkdir(parents=True,exist_ok=True)
            for name in ('runtime/python/python.exe', 'venv/Scripts/python.exe'):
                (root/name).write_bytes(b'owned')
            manifest(root,[(name,digest(b'owned')) for name in ('runtime/python/python.exe','venv/Scripts/python.exe')])
            pending=root.parent/(root.name+'-pending.txt')
            try:
                r=subprocess.run([str(WORKER),str(root),str(pending)])
                self.assertEqual(r.returncode,0)
                self.assertFalse((root/'runtime').exists())
                self.assertFalse((root/'venv').exists())
                self.assertEqual(pending.read_text(encoding='utf-16-le'),'')
            finally: pending.unlink(missing_ok=True)

    def test_nonempty_runtime_scaffolding_is_preserved(self):
        with tempfile.TemporaryDirectory(prefix='aes-uninstall-kept-',dir=SCRATCH) as folder:
            root=Path(folder)
            for name in ('runtime/python/libs','venv/Include','venv/Scripts'):
                (root/name).mkdir(parents=True,exist_ok=True)
            (root/'runtime/python/libs/user.txt').write_bytes(b'user runtime data')
            (root/'venv/Include/user.h').write_bytes(b'user header')
            for name in ('runtime/python/python.exe','venv/Scripts/python.exe'):
                (root/name).write_bytes(b'owned')
            manifest(root,[(name,digest(b'owned')) for name in ('runtime/python/python.exe','venv/Scripts/python.exe')])
            pending=root.parent/(root.name+'-pending.txt')
            try:
                r=subprocess.run([str(WORKER),str(root),str(pending)])
                self.assertEqual(r.returncode,0)
                self.assertEqual((root/'runtime/python/libs/user.txt').read_bytes(),b'user runtime data')
                self.assertEqual((root/'venv/Include/user.h').read_bytes(),b'user header')
            finally: pending.unlink(missing_ok=True)

    @unittest.skipUnless(os.name == 'nt', 'Windows directory junctions required')
    def test_user_junction_and_its_target_are_preserved(self):
        with tempfile.TemporaryDirectory(prefix='aes-uninstall-junction-',dir=SCRATCH) as folder:
            base=Path(folder); root=base/'app'; outside=base/'outside'
            (root/'owned').mkdir(parents=True); outside.mkdir()
            (outside/'owned.txt').write_bytes(b'owned')
            (outside/'user.txt').write_bytes(b'user data')
            link=root/'owned'/'junction'
            create=subprocess.run(['cmd.exe','/d','/c','mklink','/J',str(link),str(outside)],capture_output=True,text=True)
            self.assertEqual(create.returncode,0,create.stdout+create.stderr)
            try:
                manifest(root,[('owned/junction/owned.txt',digest(b'owned'))])
                pending=base/'pending.txt'
                r=subprocess.run([str(WORKER),str(root),str(pending)])
                self.assertEqual(r.returncode,0)
                self.assertTrue(link.exists(), 'uninstall removed the user junction')
                self.assertTrue(link.lstat().st_file_attributes & 0x400)
                self.assertEqual((outside/'owned.txt').read_bytes(),b'owned')
                self.assertEqual((outside/'user.txt').read_bytes(),b'user data')
            finally:
                # Remove only the synthetic junction itself before the temporary fixture.
                if link.exists() and link.lstat().st_file_attributes & 0x400:
                    os.rmdir(link)

    def test_invalid_manifest_fails_before_removing_any_file(self):
        with tempfile.TemporaryDirectory(prefix='aes-uninstall-test-',dir=SCRATCH) as folder:
            root=Path(folder); (root/'owned.bin').write_bytes(b'owned')
            manifest(root,[('owned.bin',digest(b'owned')),('../outside.bin',digest(b'other'))])
            pending=root.parent/(root.name+'-pending.txt')
            r=subprocess.run([str(WORKER),str(root),str(pending)])
            self.assertNotEqual(r.returncode,0)
            self.assertEqual((root/'owned.bin').read_bytes(),b'owned')
            self.assertFalse(pending.exists())
    def test_locked_file_is_deferred_until_worker_exits(self):
        with tempfile.TemporaryDirectory(prefix='aes-uninstall-test-',dir=SCRATCH) as folder:
            root=Path(folder); file=root/'locked.bin';file.write_bytes(b'locked')
            manifest(root,[('locked.bin',digest(b'locked'))]); pending=root.parent/(root.name+'-pending.txt')
            try:
                with file.open('rb'):
                    r=subprocess.run([str(WORKER),str(root),str(pending)])
                    self.assertEqual(r.returncode,0)
                    self.assertTrue(file.exists())
                    self.assertEqual(pending.read_text(encoding='utf-16-le').strip(),str(file))
                file.unlink()
            finally: pending.unlink(missing_ok=True)
if __name__=='__main__': unittest.main(verbosity=2)
