# stealt & modified from https://github.com/zyddnys/manga-image-translator/blob/main/manga_translator/translators/chatgpt.py

import re
import time
from typing import List, Dict, Union
import yaml
import traceback
import inspect

import openai

from .base import BaseTranslator, register_translator


OPENAPI_V1_API = int(openai.__version__.split('.')[0]) >= 1


class InvalidNumTranslations(Exception):
    pass

@register_translator('ChatGPT')
class GPTTranslator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'api key': '',
        'model': {
            'type': 'selector',
            'options': [
                'gpt-4o',
                'gpt-4-turbo',
                'gpt3',
                'gpt35-turbo',
                'gpt4',
            ],
            'value': 'gpt-4o'
        },
        'override model': '',
        'prompt template': {
            'type': 'editor',
            'value': 'Please help me to translate the following text from a manga to {to_lang} (if it\'s already in {to_lang} or looks like gibberish you have to output it as it is instead):\n',
        },
        'chat system template': {
            'type': 'editor',
            'value': 'You are a professional translation engine, please translate the text into a colloquial, elegant and fluent content, without referencing machine translations. You must only translate the text content, never interpret it. If there\'s any issue in the text, output the text as is.\nTranslate to {to_lang}.',
        },
        
        'chat sample': {
            'type': 'editor',
            'value': 
'''日本語-简体中文:
    source:
        - 二人のちゅーを 目撃した ぼっちちゃん
        - ふたりさん
        - 大好きなお友達には あいさつ代わりに ちゅーするんだって
        - アイス あげた
        - 喜多ちゃんとは どどど どういった ご関係なのでしようか...
        - テレビで見た！
    target:
        - 小孤独目击了两人的接吻
        - 二里酱
        - 我听说人们会把亲吻作为与喜爱的朋友打招呼的方式
        - 我给了她冰激凌
        - 喜多酱和你是怎么样的关系啊...
        - 我在电视上看到的！'''
        },
        'invalid repeat count': 2,
        'max requests per minute': 20,
        'delay': 0.3,
        'max tokens': 4096,
        'temperature': 0.5,
        'top p': 1.,
        # 'return prompt': False,
        'retry attempts': 5,
        'retry timeout': 15,
        '3rd party api url': '',
        'frequency penalty': 0.0,
        'presence penalty': 0.0,
        'low vram mode': {
            'value': False,
            'description': 'check it if you\'re running it locally on a single device and encountered a crash due to vram OOM',
            'type': 'checkbox',
        }
    }

    def _setup_translator(self):
        self.lang_map['简体中文'] = 'Simplified Chinese'
        self.lang_map['繁體中文'] = 'Traditional Chinese'
        self.lang_map['日本語'] = 'Japanese'
        self.lang_map['English'] = 'English'
        self.lang_map['한국어'] = 'Korean'
        self.lang_map['Tiếng Việt'] = 'Vietnamese'
        self.lang_map['čeština'] = 'Czech'
        self.lang_map['Français'] = 'French'
        self.lang_map['Deutsch'] = 'German'
        self.lang_map['magyar nyelv'] = 'Hungarian'
        self.lang_map['Italiano'] = 'Italian'
        self.lang_map['Polski'] = 'Polish'
        self.lang_map['Português'] = 'Portuguese'
        self.lang_map['limba română'] = 'Romanian'
        self.lang_map['русский язык'] = 'Russian'
        self.lang_map['Español'] = 'Spanish'
        self.lang_map['Türk dili'] = 'Turkish'
        self.lang_map['украї́нська мо́ва'] = 'Ukrainian'
        self.lang_map['Thai'] = 'Thai'
        self.lang_map['Arabic'] = 'Arabic'
        self.lang_map['Malayalam'] = 'Malayalam'
        self.lang_map['Tamil'] = 'Tamil'
        self.lang_map['Hindi'] = 'Hindi'

        self.token_count = 0
        self.token_count_last = 0
    
    @property
    def model(self) -> str:
        return self.params['model']['value']

    @property
    def temperature(self) -> float:
        return self.params['temperature']
    
    @property
    def max_tokens(self) -> int:
        return self.params['max tokens']
    
    @property
    def top_p(self) -> float:
        return self.params['top p']
    
    @property
    def retry_attempts(self) -> int:
        return self.params['retry attempts']
    
    @property
    def retry_timeout(self) -> int:
        return self.params['retry timeout']
    
    @property
    def chat_system_template(self) -> str:
        to_lang = self.lang_map[self.lang_target]
        return self.params['chat system template']['value'].format(to_lang=to_lang)
    
    @property
    def chat_sample(self):

        if self.model == 'gpt3':
            return None

        samples = self.params['chat sample']['value']
        try: 
            samples = yaml.load(self.params['chat sample']['value'], Loader=yaml.FullLoader)
        except:
            self.logger.error(f'failed to load parse sample: {samples}')
            samples = {}
        src_tgt = self.lang_source + '-' + self.lang_target
        if src_tgt in samples:
            src_list = samples[src_tgt]['source']
            tgt_list = samples[src_tgt]['target']
            src_queries = ''
            tgt_queries = ''
            for i, (src, tgt) in enumerate(zip(src_list, tgt_list)):
                src_queries += f'\n<|{i+1}|>{src}'
                tgt_queries += f'\n<|{i+1}|>{tgt}'
            src_queries = src_queries.lstrip()
            tgt_queries = tgt_queries.lstrip()
            return [src_queries, tgt_queries]
        else:
            return None

    def _assemble_prompts(self, queries: List[str], from_lang: str = None, to_lang: str = None, max_tokens = None):
        if from_lang is None:
            from_lang = self.lang_map[self.lang_source]
        if to_lang is None:
            to_lang = self.lang_map[self.lang_target]
            
        prompt = ''

        if max_tokens is None:
            max_tokens = self.max_tokens
        # return_prompt = self.params['return prompt']
        prompt_template = self.params['prompt template']['value'].format(to_lang=to_lang).rstrip()
        prompt += prompt_template

        i_offset = 0
        num_src = 0
        for i, query in enumerate(queries):
            prompt += f'\n<|{i+1-i_offset}|>{query}'
            num_src += 1
            # If prompt is growing too large and theres still a lot of text left
            # split off the rest of the queries into new prompts.
            # 1 token = ~4 characters according to https://platform.openai.com/tokenizer
            # TODO: potentially add summarizations from special requests as context information
            if max_tokens * 2 and len(''.join(queries[i+1:])) > max_tokens:
                # if return_prompt:
                #     prompt += '\n<|1|>'
                yield prompt.lstrip(), num_src
                prompt = prompt_template
                # Restart counting at 1
                i_offset = i + 1
                num_src = 0

        # if return_prompt:
        #     prompt += '\n<|1|>'
        yield prompt.lstrip(), num_src

    def _format_prompt_log(self, to_lang: str, prompt: str) -> str:
        chat_sample = self.chat_sample
        if self.model != 'gpt3' and chat_sample is not None:
            return '\n'.join([
                'System:',
                self.chat_system_template,
                'User:',
                chat_sample[0],
                'Assistant:',
                chat_sample[1],
                'User:',
                prompt,
            ])
        else:
            return '\n'.join([
                'System:',
                self.chat_system_template,
                'User:',
                prompt,
            ])

    def _translate(self, src_list: List[str]) -> List[str]:
        translations = []
        # self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')
        from_lang = self.lang_map[self.lang_source]
        to_lang = self.lang_map[self.lang_target]
        queries = src_list
        # return_prompt = self.params['return prompt']
        chat_sample = self.chat_sample
        for prompt, num_src in self._assemble_prompts(queries, from_lang, to_lang):
            retry_attempt = 0
            while True:
                try:
                    response = self._request_translation(prompt, chat_sample)
                    new_translations = re.split(r'<\|\d+\|>', response)[-num_src:]
                    if len(new_translations) != num_src:
                        # https://github.com/dmMaze/BallonsTranslator/issues/379
                        _tr2 = re.sub(r'<\|\d+\|>', '', response)
                        _tr2 = _tr2.split('\n')
                        if len(_tr2) == num_src:
                            new_translations = _tr2
                        else:
                            raise InvalidNumTranslations
                    break
                except InvalidNumTranslations:
                    retry_attempt += 1
                    message = f'number of translations does not match to source:\nprompt:\n    {prompt}\ntranslations:\n  {new_translations}\nopenai response:\n  {response}'
                    if retry_attempt >= self.retry_attempts:
                        self.logger.error(message)
                        new_translations = [''] * num_src
                        break
                    self.logger.warning(message + '\n' + f'Restarting request. Attempt: {retry_attempt}')

                except Exception as e:
                    retry_attempt += 1
                    if retry_attempt >= self.retry_attempts:
                        new_translations = [''] * num_src
                        break
                    self.logger.warning(f'Translation failed due to {e}. Attempt: {retry_attempt}, sleep for {self.retry_timeout} secs...')
                    self.logger.error(f'Request traceback: ', traceback.format_exc())
                    time.sleep(self.retry_timeout)
                    # time.sleep(self.retry_timeout)
            # if return_prompt:
            #     new_translations = new_translations[:-1]

            # if chat_sample is not None:
            #     new_translations = new_translations[1:]
            translations.extend([t.strip() for t in new_translations])

        if self.token_count_last:
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')

        return translations

    def _request_translation_gpt3(self, prompt: str) -> str:

        if OPENAPI_V1_API:
            openai_completions_create = openai.completions.create
        else:
            openai_completions_create = openai.Completion.create

        response = openai_completions_create(
            model='text-davinci-003',
            prompt=prompt,
            max_tokens=self.max_tokens // 2, # Assuming that half of the tokens are used for the query
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=float(self.params['frequency penalty']),
            presence_penalty=float(self.params['presence penalty'])
        )

        if OPENAPI_V1_API:
            if response.usage is not None:
                self.token_count += response.usage.total_tokens
                self.token_count_last = response.usage.total_tokens
        else:
            self.token_count += response.usage['total_tokens']
            self.token_count_last = response.usage['total_tokens']
        return response.choices[0].text
    
    def _request_translation_with_chat_sample(self, prompt: str, model: str, chat_sample: List) -> str:
        messages = [
            {'role': 'system', 'content': self.chat_system_template},
            {'role': 'user', 'content': prompt},
        ]

        if chat_sample is not None:
            messages.insert(1, {'role': 'user', 'content': chat_sample[0]})
            messages.insert(2, {'role': 'assistant', 'content': chat_sample[1]})

        func_args = {
            'model': model,
            'messages': messages,
            'temperature': self.temperature,
            'top_p': self.top_p,
        }
        max_tokens = self.max_tokens // 2 # Assuming that half of the tokens are used for the query
        func_parameters = inspect.signature(openai.chat.completions.create).parameters
        if 'max_completion_tokens' in func_parameters:
            func_args['max_completion_tokens'] = max_tokens
        else:
            func_args['max_tokens'] = max_tokens
        if 'presence_penalty' in func_parameters:
            func_args['presence_penalty'] = self.params['presence penalty']
            func_args['frequency_penalty'] = self.params['frequency penalty']

        if OPENAPI_V1_API:
            openai_chatcompletions_create = openai.chat.completions.create
        else:
            openai_chatcompletions_create = openai.ChatCompletion.create

        response = openai_chatcompletions_create(**func_args)

        if OPENAPI_V1_API:
            if response.usage is not None:
                self.token_count += response.usage.total_tokens
                self.token_count_last = response.usage.total_tokens
        else:
            self.token_count += response.usage['total_tokens']
            self.token_count_last = response.usage['total_tokens']
        for choice in response.choices:
            if OPENAPI_V1_API:
                return choice.message.content
            else:
                if 'text' in choice:
                    return choice.text

        # If no response with text is found, return the first response's content (which may be empty)
        return response.choices[0].message.content

    @property
    def api_url(self):
        url = self.params['3rd party api url'].strip()
        if not url:
            return None
        
        # 对于小于v1.0.0版本的openai包，末尾的斜杠会导致请求失败，因此弹出警告
        if url.endswith('v1/'):
            if not OPENAPI_V1_API:
                self.logger.warning(f"The OpenAI package version you are using is outdated. Please remove the trailing slash after 'v1' in the URL: {url}")

        # 检查是否包含"/v1"
        if '/v1' not in url:
            self.logger.warning(f"API URL does not contain '/v1': {url}, please ensure it's the correct URL.")
        
        return url

    def _request_translation(self, prompt, chat_sample: List):

        self.logger.debug(f'chatgpt prompt: \n {prompt}' )

        openai.api_key = self.params['api key'].strip()
        base_url = self.api_url
        if OPENAPI_V1_API:
            openai.base_url = base_url
        else:
            if base_url is None:
                base_url = 'https://api.openai.com/v1'
            openai.api_base = base_url
        
        override_model = self.params['override model'].strip()
        if override_model != '':
            model: str = override_model
        else:
            model:str = self.model
            if model == 'gpt3':
                return self._request_translation_gpt3(prompt)
            elif model == 'gpt35-turbo':
                model = 'gpt-3.5-turbo'
            elif model == 'gpt4':
                model = 'gpt-4'

        return self._request_translation_with_chat_sample(prompt, model, chat_sample)