import requests
import hashlib
import time
import uuid
from typing import List, Dict
from .base import *

@register_translator('Youdao')
class YoudaoTranslator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'api_key': '',
        'app_secret': '',
    }

    @property
    def api_key(self) -> str:
        return self.params['api_key']

    @property
    def app_secret(self) -> str:
        return self.params['app_secret']

    def _setup_translator(self):
        self.lang_map['简体中文'] = 'zh-CHS'
        self.lang_map['English'] = 'en'
        self.lang_map['日本語'] = 'ja'
        self.lang_map['한국어'] = 'ko'
        # Add more language mappings as needed

    def generate_input(self, query: str) -> str:
        if len(query) > 20:
            input_str = query[:10] + str(len(query)) + query[-10:]
        else:
            input_str = query
        return input_str

    def generate_sign(self, query: str, salt: str, curtime: str) -> str:
        input_str = self.generate_input(query)
        sign_str = self.api_key + input_str + salt + curtime + self.app_secret
        hash_algorithm = hashlib.sha256()
        hash_algorithm.update(sign_str.encode('utf-8'))
        return hash_algorithm.hexdigest()

    def _translate(self, src_list: List[str]) -> List[str]:
        url = "https://openapi.youdao.com/api"
        results = []
        for query in src_list:
            salt = str(uuid.uuid4())
            curtime = str(int(time.time()))
            sign = self.generate_sign(query, salt, curtime)

            payload = {
                'q': query,
                'from': self.lang_map[self.lang_source],
                'to': self.lang_map[self.lang_target],
                'appKey': self.api_key,
                'salt': salt,
                'sign': sign,
                'signType': 'v3',
                'curtime': curtime,
            }

            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }

            response = requests.post(url, data=payload, headers=headers)
            response_data = response.json()

            if 'translation' in response_data:
                results.append(response_data['translation'][0])
            else:
                results.append('')

        return results

