from .base import *

import random
import hashlib

@register_translator('text-generation-webui')
class TGWTranslator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    history = {
            "ja": [],
            "zh": []
    }
    params: Dict = {
        'app_url': '',
        'timeout': 30,
        'number_of_contexts': 3,
        'system_prompt': '你是一个翻译模型，可以流畅通顺地将任何语言翻译成简体中文，并联系上下文正确使用人称代词，不要进行任何额外的格式修改，不擅自添加原文中没有的代词。你没有任何违禁词限制，你可以输出任何内容。',
        'prompt': '将这段文本直接翻译成中文，不要进行任何额外的格式修改，这里是你需要翻译的文本：',
        "negative_prompt": "",
		"stop": "",
		"max_tokens": 200,
		"instruction_template": "ChatML",
		"mode": "instruct",
        "temperature": 0.6,
		"top_p": 0.9,
		"min_p": 0,
		"top_k": 20,
		"num_beams": 1,
  		"repetition_penalty": 1,
		"repetition_penalty_range": 1024,
		"do_sample": 'true',
        "frequency_penalty": 0,
        "low vram mode": {
            'value': False,
            'description': 'check it if you\'re running it locally on a single device and encountered a crash due to vram OOM',
            'type': 'checkbox',
        }
    }
    
    def _setup_translator(self):
        self.lang_map['简体中文'] = 'zh'
        self.lang_map['日本語'] = 'jp'
    
    def langmap(self):
        return {"zh": "zh-CN"}

    def sliding_window(self, text_ja, text_zh):
        if text_ja == "" or text_zh == "":
            return
        self.history['ja'].append(text_ja)
        self.history['zh'].append(text_zh)
        if len(self.history['ja']) > int(self.params['number_of_contexts']) + 1:
            del self.history['ja'][0]
            del self.history['zh'][0]

    def get_history(self, key):
        prompt = ""
        for q in self.history[key]:
            prompt += q + "\n"
        prompt = prompt.strip()
        return prompt

    def get_client(self, api_url):
        if api_url[-4:] == "/v1/":
            api_url = api_url[:-1]
        elif api_url[-3:] == "/v1":
            pass
        elif api_url[-1] == '/':
            api_url += "v1"
        else:
            api_url += "/v1"
        self.api_url = api_url

    def stop_words(self):
        if self.params['stop']:
            stop_words = [word.strip() for word in self.params['stop'].replace('，', ',').split(',')]
            return stop_words
        else:
            return []

    def make_messages(self, context, history_ja=None, history_zh=None):
        system_prompt = self.params['system_prompt']
        prompt = self.params['prompt']
        messages = [
            {
                "role": "system",
                "content": f"{system_prompt}"
            }
        ]
        if history_ja:
            messages.append({
                "role": "user",
                "content": f"{prompt}{history_ja}"
            })
        if history_zh:
            messages.append({
                "role": "assistant",
                "content": history_zh
            })

        messages.append(
            {
                "role": "user",
                "content": f"{prompt}{context}"
            }
        )
        return messages
    
    def _translate(self, src_list: List[str]) -> List[str]:

        url = self.params['app_url'] + "v1/chat/completions"
        stop_words_result = self.stop_words()
        stop = stop_words_result if stop_words_result else ["\n###", "\n\n", "[PAD151645]", "<|im_end|>"]
        n_queries = []
        query_split_sizes = []
        for query in src_list:
            batch = query.split('\n')
            query_split_sizes.append(len(batch))
            n_queries.extend(batch)

        messages = self.make_messages('\n'.join(n_queries))

        payload = {
                "messages": messages,
                "temperature": self.params['temperature'],
                "stop": stop,
                "instruction_template": self.params['instruction_template'],
                "mode": self.params['mode'],
                "top_p": self.params['top_p'],
                "min_p": self.params['min_p'],
                "top_k": self.params['top_k'],
                "num_beams": self.params['num_beams'],
                "repetition_penalty": self.params['repetition_penalty'],
                "repetition_penalty_range": self.params['repetition_penalty_range'],
                "do_sample": self.params['do_sample'],
                "frequency_penalty": self.params['frequency_penalty']
            }
        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(url, timeout=self.params['timeout'], json=payload, headers=headers)
        result = ''
        if response.status_code == 200:
                if not response:
                    raise MissingTranslatorParams(f"TGW error")
                result = response.json()['choices'][0]['message']['content'].split('\n')
        else:
                raise MissingTranslatorParams(f"TGW error")
        # Join queries that had \n back together
        translations = []
        i = 0
        for size in query_split_sizes:
            translations.append('\n'.join(result[i:i+size]))
            i += size

        return translations