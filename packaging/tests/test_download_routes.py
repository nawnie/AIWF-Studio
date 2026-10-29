import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from aiwf.web.unified_api import build_unified_router

class RouteTests(unittest.TestCase):
    def client(self, host):
        class Flags:
            data_dir=Path(tempfile.gettempdir())
            def resolved_ckpt_dir(self): return self.data_dir
        app=FastAPI();app.include_router(build_unified_router(SimpleNamespace(flags=Flags())))
        return TestClient(app,client=(host,12345))
    def test_local_post_dispatches_and_rejects_invalid_input(self):
        client=self.client('127.0.0.1')
        body={'source':'huggingface','reference':'owner/model','filename':'x.safetensors'}
        with patch('aiwf.services.aes_downloads.ModelDownloads.start',return_value={'id':'test','state':'running'}) as start:
            response=client.post('/api/pro/unified/checkpoint-downloads',json=body)
        self.assertEqual(response.status_code,200)
        start.assert_called_once_with('huggingface','owner/model','x.safetensors')
        response=client.post('/api/pro/unified/checkpoint-downloads',json={'source':'civitai','reference':'../bad','filename':'x.safetensors'})
        self.assertEqual(response.status_code,422)
    def test_remote_post_is_forbidden(self):
        client=self.client('203.0.113.9')
        response=client.post('/api/pro/unified/checkpoint-downloads',json={'source':'civitai','reference':'123','filename':'x.safetensors'})
        self.assertEqual(response.status_code,403)
    def test_active_job_can_be_reattached(self):
        with patch('aiwf.services.aes_downloads.ModelDownloads.active',return_value={'id':'test','state':'running'}):
            response=self.client('127.0.0.1').get('/api/pro/unified/checkpoint-downloads')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['job']['id'],'test')
    def test_unknown_job_is_not_found(self):
        self.assertEqual(self.client('127.0.0.1').get('/api/pro/unified/checkpoint-downloads/absent').status_code,404)
if __name__=='__main__': unittest.main(verbosity=2)
