import io, json, os, struct, tempfile, threading, time, unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request
from email.message import Message
from aiwf.services import aes_downloads as dl

SCRATCH=os.environ.get('AES_STUDIO_TEST_TMP') or None

def fixture():
    h=json.dumps({'tensor': {'dtype':'U8','shape':[1],'data_offsets':[0,1]}}).encode()
    return struct.pack('<Q',len(h))+h+b'\x01'

class FakeResponse:
    def __init__(self, data):
        self.data=io.BytesIO(data); self.headers={'Content-Length':str(len(data))}
    def __enter__(self): return self
    def __exit__(self, *_): self.data.close()
    def read(self, n): return self.data.read(n)

class FakeOpener:
    def __init__(self, data): self.data=data; self.requests=[]
    def open(self, request, timeout):
        self.requests.append(request)
        return FakeResponse(self.data)

class Flags:
    def __init__(self, root): self.root=root
    def resolved_ckpt_dir(self): return self.root

def done(manager, job):
    for _ in range(100):
        state=manager.status(job['id'])
        if state['state']!='running': return state
        time.sleep(.02)
    raise AssertionError('Download worker did not finish')

class DownloadSafetyTests(unittest.TestCase):
    def test_only_https_provider_redirects_and_authentication_stripping(self):
        self.assertTrue(dl._trusted_url('huggingface','https://cas-bridge.xethub.hf.co/a'))
        for url in ('http://huggingface.co/a','https://huggingface.co.evil.example/a','https://127.0.0.1/a','https://evil.example/a'):
            self.assertFalse(dl._trusted_url('huggingface',url),url)
        req=Request('https://huggingface.co/a',headers={'Authorization':'Bearer fake-test-only'})
        handler=dl._ProviderRedirect('huggingface')
        same=handler.redirect_request(req,None,302,'Found',Message(),'https://huggingface.co/b')
        other=handler.redirect_request(req,None,302,'Found',Message(),'https://cdn-lfs.huggingface.co/b')
        self.assertEqual(same.get_header('Authorization'),'Bearer fake-test-only')
        self.assertIsNone(other.get_header('Authorization'))
        with self.assertRaises(dl.DownloadProblem): handler.redirect_request(req,None,302,'Found',Message(),'https://evil.example/b')
    def test_both_providers_use_saved_keys_and_preserve_existing_files(self):
        with tempfile.TemporaryDirectory(prefix='aes-download-test-',dir=SCRATCH) as folder:
            root=Path(folder); sentinel=root/'.someone-elses.partial'; sentinel.write_bytes(b'keep')
            for source, reference, filename in [('huggingface','example/model','first.safetensors'),('civitai','12345','second.safetensors')]:
                fake=FakeOpener(fixture()); manager=dl.ModelDownloads(Flags(root))
                with patch.object(dl, 'read_provider_key', return_value='synthetic-not-a-secret'), patch.object(dl.urllib.request,'build_opener',return_value=fake):
                    state=done(manager,manager.start(source,reference,filename))
                self.assertEqual(state['state'],'completed',state)
                self.assertEqual((root/filename).read_bytes(),fixture())
                self.assertNotIn('synthetic-not-a-secret',json.dumps(state))
                self.assertEqual(fake.requests[0].get_header('Authorization'),'Bearer synthetic-not-a-secret')
                with self.assertRaises(dl.DownloadProblem):manager.start(source,reference,filename)
            self.assertEqual(sentinel.read_bytes(),b'keep')
    def test_invalid_file_and_wrong_official_hash_are_not_installed(self):
        with tempfile.TemporaryDirectory(prefix='aes-download-test-',dir=SCRATCH) as folder:
            root=Path(folder)
            for reference,filename,data in [('example/model','broken.safetensors',b'not a checkpoint'),(dl.SDXL_REPO,dl.SDXL_FILE,fixture())]:
                manager=dl.ModelDownloads(Flags(root))
                with patch.object(dl,'read_provider_key',return_value=''),patch.object(dl.urllib.request,'build_opener',return_value=FakeOpener(data)):
                    state=done(manager,manager.start('huggingface',reference,filename))
                self.assertEqual(state['state'],'failed',state)
                self.assertFalse((root/filename).exists())
                self.assertFalse(list(root.glob('.aes-download-*.partial')))
    def test_cancellation_keeps_unrelated_files_and_removes_its_partial(self):
        with tempfile.TemporaryDirectory(prefix='aes-download-test-',dir=SCRATCH) as folder:
            root=Path(folder); keep=root/'other.partial'; keep.write_bytes(b'keep')
            manager=dl.ModelDownloads(Flags(root)); ident='synthetic-job'
            manager.jobs[ident]={'state':'running','message':'','bytes_received':0,'bytes_total':0}
            event=threading.Event(); event.set()
            with patch.object(dl,'read_provider_key',return_value=''),patch.object(dl.urllib.request,'build_opener',return_value=FakeOpener(fixture())):
                manager._run(ident,'huggingface','https://huggingface.co/example/model/resolve/main/a.safetensors',root/'a.safetensors',None,event)
            self.assertEqual(manager.status(ident)['state'],'cancelled')
            self.assertEqual(keep.read_bytes(),b'keep')
            self.assertFalse((root/'a.safetensors').exists())
            self.assertFalse(list(root.glob('.aes-download-*.partial')))

    def test_invalid_requests_never_start_a_worker(self):
        manager=dl.ModelDownloads(Flags(SCRATCH))
        for source,reference,filename in [('huggingface','https://evil.example/x','x.safetensors'),('civitai','abc','x.safetensors'),('civitai','123','../x.safetensors'),('huggingface','owner/repo','program.exe')]:
            with self.assertRaises(dl.DownloadProblem):manager.start(source,reference,filename)
        self.assertEqual(manager.jobs,{})

    def test_supported_model_formats_are_saved_without_executing_them(self):
        with tempfile.TemporaryDirectory(prefix='aes-download-formats-',dir=SCRATCH) as folder:
            root=Path(folder)
            examples={
                'sample.safetensors': fixture(),
                'sample.ckpt': b'legacy checkpoint fixture',
                'sample.pt': b'legacy pt fixture',
                'sample.pth': b'legacy pth fixture',
                'sample.gguf': struct.pack('<4sIQQ',b'GGUF',3,1,0)+b'x',
            }
            for filename,data in examples.items():
                manager=dl.ModelDownloads(Flags(root))
                with patch.object(dl,'read_provider_key',return_value=''),patch.object(dl.urllib.request,'build_opener',return_value=FakeOpener(data)):
                    state=done(manager,manager.start('huggingface','example/model',filename))
                self.assertEqual(state['state'],'completed',state)
                self.assertEqual((root/filename).read_bytes(),data)
                if filename.endswith(('.ckpt','.pt','.pth')):
                    self.assertIn('can execute code',state['message'])
                if filename.endswith('.gguf'):
                    self.assertIn('Only the GGUF header',state['message'])

    def test_invalid_gguf_and_empty_legacy_file_are_not_installed(self):
        with tempfile.TemporaryDirectory(prefix='aes-download-invalid-formats-',dir=SCRATCH) as folder:
            root=Path(folder)
            for filename,data in [('bad.gguf',b'not a GGUF file'),('empty.ckpt',b'')]:
                manager=dl.ModelDownloads(Flags(root))
                with patch.object(dl,'read_provider_key',return_value=''),patch.object(dl.urllib.request,'build_opener',return_value=FakeOpener(data)):
                    state=done(manager,manager.start('civitai','123',filename))
                self.assertEqual(state['state'],'failed',state)
                self.assertFalse((root/filename).exists())
                self.assertFalse(list(root.glob('.aes-download-*.partial')))

if __name__=='__main__':unittest.main(verbosity=2)
