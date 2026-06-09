from .base import *

import random
import hashlib

@register_translator('Baidu')
class BaiduTranslator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'token': '',
        'appId': '',
        'delay': 0.0
    }
    @staticmethod
    def get_json(from_lang, to_lang, query_text,BAIDU_APP_ID,BAIDU_SECRET_KEY):
        salt = random.randint(32768, 65536)
        sign = BAIDU_APP_ID + query_text + str(salt) + BAIDU_SECRET_KEY
        sign = sign.encode('utf-8')
        m1 = hashlib.md5()
        m1.update(sign)
        sign = m1.hexdigest()
        payload = {
            "appid": BAIDU_APP_ID,
            "q": query_text,
            "from": from_lang,
            "to": to_lang,
            "salt":str(salt),
            "sign":sign
        }
        return payload

    def _setup_translator(self):
        self.lang_map['简体中文'] = 'zh'
        self.lang_map['繁體中文'] = 'cht'
        self.lang_map['日本語'] = 'jp'
        self.lang_map['English'] = 'en'  
    
    def _translate(self, src_list: List[str]) -> List[str]:

        n_queries = []
        query_split_sizes = []
        for query in src_list:
            batch = query.split('\n')
            query_split_sizes.append(len(batch))
            n_queries.extend(batch)
        token = self.params['token']
        appId = self.params['appId']
        if token == '' or token is None:
            raise MissingTranslatorParams('token')
        if appId == '' or appId is None:
            raise MissingTranslatorParams('appId')
        
        payload = self.get_json(self.lang_map[self.lang_source], self.lang_map[self.lang_target], '\n'.join(n_queries),appId,token)
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }

        response = requests.request("POST", 'https://fanyi-api.baidu.com/api/trans/vip/translate', data=payload, headers=headers)
        result = json.loads(response.text)
        result_list = []
        if "trans_result" not in result:
            raise MissingTranslatorParams(f'Baidu returned invalid response: {result}\nAre the API keys set correctly?')
        for ret in result["trans_result"]:
            for v in ret["dst"].split('\n'):
                result_list.append(v)

        # Join queries that had \n back together
        translations = []
        i = 0
        for size in query_split_sizes:
            translations.append('\n'.join(result_list[i:i+size]))
            i += size

        return translations