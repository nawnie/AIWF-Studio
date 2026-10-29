import json
import unittest
from email.message import Message
from unittest.mock import patch
from urllib.request import Request
from aiwf.infrastructure.download import stream
from aiwf.core.config.settings import UserSettings

class ProviderSafetyTests(unittest.TestCase):
    def test_hugging_face_token_is_only_sent_to_https_huggingface(self):
        with patch.object(stream, 'read_provider_key', return_value='synthetic-test-key'):
            self.assertEqual(stream._auth_headers('https://huggingface.co/org/model'), {'Authorization':'Bearer synthetic-test-key'})
            for url in ('http://huggingface.co/path', 'https://evil.example/path', 'https://huggingface.co.evil.example/path', 'https://civitai.com/path'):
                self.assertEqual(stream._auth_headers(url), {}, url)
    def test_redirect_strips_authentication_across_origin_and_keeps_same_origin(self):
        handler=stream._SafeRedirect(); headers=Message()
        def redirected(url):
            req=Request('https://huggingface.co/a', headers={'Authorization':'Bearer synthetic-test-key'})
            return handler.redirect_request(req, None, 302, 'Found', headers, url)
        self.assertEqual(redirected('https://huggingface.co/b').get_header('Authorization'), 'Bearer synthetic-test-key')
        self.assertIsNone(redirected('https://unrelated.example/b').get_header('Authorization'))
        self.assertIsNone(redirected('http://huggingface.co/b').get_header('Authorization'))
    def test_serialized_settings_exclude_both_provider_keys(self):
        settings=UserSettings(huggingface_token='synthetic-hf',civitai_token='synthetic-civitai')
        text=settings.model_dump_json(exclude={'huggingface_token','civitai_token'})
        self.assertNotIn('synthetic-hf',text)
        self.assertNotIn('synthetic-civitai',text)
        self.assertNotIn('huggingface_token',json.loads(text))
        self.assertNotIn('civitai_token',json.loads(text))
    def test_only_named_provider_targets_can_be_read(self):
        from aiwf.credentials import read_provider_key
        self.assertEqual(read_provider_key('not-a-provider'), '')

if __name__ == '__main__': unittest.main(verbosity=2)
