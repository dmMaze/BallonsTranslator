from .base import *

@register_translator('Caiyun')
class CaiyunTranslator(BaseTranslator):

    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'token': '',
        'delay': 0.0
    }

    def _setup_translator(self):
        self.lang_map['简体中文'] = 'zh'
        self.lang_map['繁體中文'] = 'zh'
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'en'  
        
    def _translate(self, src_list: List[str]) -> List[str]:

        url = "http://api.interpreter.caiyunai.com/v1/translator"
        token = self.params['token']
        if token == '' or token is None:
            raise MissingTranslatorParams('token')

        direction = self.lang_map[self.lang_source] + '2' + self.lang_map[self.lang_target]

        payload = {
            "source": src_list,
            "trans_type": direction,
            "request_id": "demo",
            "detect": True,
        }

        headers = {
            "content-type": "application/json",
            "x-authorization": "token " + token,
        }

        response = requests.request("POST", url, data=json.dumps(payload), headers=headers)
        translations = json.loads(response.text)["target"]

        return translations