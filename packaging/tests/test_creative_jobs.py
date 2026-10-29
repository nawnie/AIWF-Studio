import io
import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from aiwf.services.creative_jobs import CreativeJobs, _PRESETS, _graph
from aiwf.services.unified_bridge import BridgeError
from aiwf.web.unified_api import build_unified_router


class CreativeJobsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('AES_STUDIO_TEST_TMP'))
        self.root = Path(self.temp.name)
        self.events = []
        self.queue = []
        preset = _PRESETS['qwen21-edit']
        self.info = {node['class_type']: {'input': {'required': {}}} for node in _graph(preset).values()}
        for _, (_, _, node, field) in preset['fields'].items():
            self.info[node]['input']['required'][field] = [[field + '-file']]
        def respond(request):
            self.events.append((request.method, request.url.path, request.content))
            if request.url.path == '/object_info': return httpx.Response(200, json=self.info)
            if request.url.path == '/upload/image':
                # Extract the generated unique name from the multipart body.
                import re
                match = re.search(rb'filename="(aes-studio-[0-9a-f]{32}\.png)"', request.content)
                return httpx.Response(200, json={'name':match.group(1).decode(),'subfolder':''})
            if request.url.path == '/prompt':
                self.queue = [[1,'prompt-own'], [2,'prompt-other']]
                return httpx.Response(200, json={'prompt_id':'prompt-own'})
            if request.url.path == '/queue' and request.method == 'GET':
                return httpx.Response(200, json={'queue_running':[], 'queue_pending':self.queue})
            if request.url.path == '/queue' and request.method == 'POST':
                payload = json.loads(request.content)
                self.queue = [v for v in self.queue if v[1] not in payload['delete']]
                return httpx.Response(200, json={})
            if request.url.path == '/view': return httpx.Response(200, content=b'png bytes')
            return httpx.Response(404)
        self.transport=httpx.MockTransport(respond)
        self.runner=CreativeJobs(client=lambda slow: httpx.Client(base_url='http://127.0.0.1:8188', transport=self.transport), comfy_url='http://127.0.0.1:8188')
        self.selections={'model':'unet_name-file','encoder':'clip_name-file','vae':'vae_name-file'}
        self.image=self.root/'input.png'
        Image.new('RGB',(8,8),'red').save(self.image)

    def tearDown(self): self.temp.cleanup()

    def test_missing_node_and_unlisted_component_refuse_before_queue(self):
        self.info.pop('QwenImage21Cache')
        with self.assertRaises(BridgeError) as missing:
            self.runner.start_creative(name='qwen21-edit',prompt='edit',selections=self.selections,input_path=str(self.image),output_root=self.root)
        self.assertEqual(missing.exception.code,'missing_nodes')
        self.info['QwenImage21Cache']={'input':{'required':{}}}
        with self.assertRaises(BridgeError) as bad:
            self.runner.start_creative(name='qwen21-edit',prompt='edit',selections={**self.selections,'model':'not-listed'},input_path=str(self.image),output_root=self.root)
        self.assertEqual(bad.exception.code,'invalid_selection')
        self.assertFalse(any(path in ('/upload/image','/prompt') for _,path,_ in self.events))

    def test_valid_user_image_is_uploaded_and_only_own_pending_job_cancelled(self):
        with patch.object(self.runner,'_watch_guarded',return_value=None):
            job=self.runner.start_creative(name='qwen21-edit',prompt='make sky blue',selections=self.selections,input_path=str(self.image),output_root=self.root)['job']
        sent=json.loads(next(body for method,path,body in self.events if path=='/prompt'))['prompt']
        self.assertEqual(sent['459:474']['inputs']['prompt'],'make sky blue')
        self.assertTrue(sent['470']['inputs']['image'].startswith('aes-studio-'))
        result=self.runner.cancel(job['job_id'])['job']
        self.assertEqual(result['state'],'cancelled')
        self.assertEqual(self.queue,[[2,'prompt-other']])

    def test_running_job_never_sends_a_global_interrupt(self):
        with patch.object(self.runner,'_watch_guarded',return_value=None):
            job=self.runner.start_creative(name='qwen21-edit',prompt='edit',selections=self.selections,input_path=str(self.image),output_root=self.root)['job']
        def running(request):
            self.events.append((request.method,request.url.path,request.content))
            if request.url.path=='/queue': return httpx.Response(200,json={'queue_running':[[1,'prompt-own']],'queue_pending':[[2,'prompt-other']]})
            return httpx.Response(404)
        self.transport=httpx.MockTransport(running)
        with self.assertRaises(BridgeError) as result: self.runner.cancel(job['job_id'])
        self.assertEqual(result.exception.code,'already_running')
        self.assertFalse(any(path=='/interrupt' for _,path,_ in self.events))

    def test_rejects_non_image_and_invalid_output_paths(self):
        bad=self.root/'bad.png';bad.write_text('not an image')
        with self.assertRaises(BridgeError):
            self.runner.start_creative(name='qwen21-edit',prompt='edit',selections=self.selections,input_path=str(bad),output_root=self.root)
        with httpx.Client(base_url='http://127.0.0.1:8188',transport=self.transport) as client:
            job={'output_root':self.root,'preset':'qwen21-edit'}
            for ref in [{'filename':'../other.png'}, {'filename':'x.png','subfolder':'../outside'}, {'filename':'x.png','type':'input'}]:
                with self.assertRaises(ValueError): self.runner._copy_output(job,ref,client)

    def test_collision_does_not_delete_existing_output(self):
        from datetime import datetime
        folder=self.root/'creative-models'/datetime.now().strftime('%Y-%m-%d');folder.mkdir(parents=True)
        existing=folder/'qwen21-edit-aaaaaaaaaaaa.png';existing.write_bytes(b'keep')
        with patch('aiwf.services.creative_jobs.uuid.uuid4',return_value=SimpleNamespace(hex='a'*32)):
            with httpx.Client(base_url='http://127.0.0.1:8188',transport=self.transport) as client:
                with self.assertRaises(FileExistsError):self.runner._copy_output({'output_root':self.root,'preset':'qwen21-edit'},{'filename':'x.png'},client)
        self.assertEqual(existing.read_bytes(),b'keep')

    def test_moss_queues_with_standard_comfy_loaders(self):
        preset = _PRESETS['moss-image']
        self.info = {node['class_type']: {'input': {'required': {}}} for node in _graph(preset).values()}
        for _, (_, _, node, field) in preset['fields'].items():
            self.info[node]['input']['required'][field] = [[field + '-file']]
        self.assertNotIn('AESModelFamilyLoader', self.info)
        choices = {'model':'unet_name-file', 'encoder':'clip_name-file', 'vae':'vae_name-file'}
        with patch.object(self.runner, '_watch_guarded', return_value=None):
            self.runner.start_creative(name='moss-image', prompt='blue mug', selections=choices,
                                       input_path=None, output_root=self.root)
        graph = json.loads(next(body for method,path,body in self.events if path == '/prompt'))['prompt']
        self.assertEqual(graph['1']['inputs']['unet_name'], choices['model'])
        self.assertEqual(graph['8']['inputs']['clip_name'], choices['encoder'])
        self.assertEqual(graph['8']['inputs']['type'], 'moss_image')
        self.assertEqual(graph['9']['inputs']['vae_name'], choices['vae'])
        self.assertEqual(graph['2']['inputs']['clip'], ['8', 0])
        self.assertEqual(graph['6']['inputs']['vae'], ['9', 0])

    def test_remote_cannot_access_creative_inventory(self):
        flags=SimpleNamespace(data_dir=self.root)
        app=FastAPI();app.include_router(build_unified_router(SimpleNamespace(flags=flags)))
        client=TestClient(app,client=('203.0.113.10',1234))
        response=client.get('/api/pro/unified/creative/inventory',params={'preset':'moss-image'})
        self.assertEqual(response.status_code,403)

if __name__=='__main__': unittest.main()
