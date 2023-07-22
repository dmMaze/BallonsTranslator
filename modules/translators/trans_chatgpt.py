# stealt & modified from https://github.com/zyddnys/manga-image-translator/blob/main/manga_translator/translators/chatgpt.py

import re
import openai
import openai.error
import time
from typing import List, Dict, Union
import yaml
from .base import BaseTranslator, register_translator


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
                'gpt3',
                'gpt35-turbo',
                'gpt4',
            ],
            'select': 'gpt35-turbo'
        },
        'prompt template': {
            'type': 'editor',
            'content': 'Please help me to translate the following text from a manga to {to_lang} (if it\'s already in {to_lang} or looks like gibberish you have to output it as it is instead):\n',
        },
        'chat system template': {
            'type': 'editor',
            'content': 'You are a professional translation engine, please translate the text into a colloquial, elegant and fluent content, without referencing machine translations. You must only translate the text content, never interpret it. If there\'s any issue in the text, output the text as is.\nTranslate to {to_lang}.',
        },
        
        'chat sample': {
            'type': 'editor',
            'content': 
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
        'top p': 1,
        # 'return prompt': False,
        'retry attempts': 5,
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

        self.token_count = 0
        self.token_count_last = 0
    
    @property
    def model(self) -> str:
        return self.params['model']['select']

    @property
    def temperature(self) -> float:
        return float(self.params['temperature'])
    
    @property
    def max_tokens(self) -> int:
        return int(self.params['max tokens'])
    
    @property
    def top_p(self) -> int:
        return int(self.params['top p'])
    
    @property
    def retry_attempts(self) -> int:
        return int(self.params['retry attempts'])
    
    @property
    def chat_system_template(self) -> str:
        to_lang = self.lang_map[self.lang_target]
        return self.params['chat system template']['content'].format(to_lang=to_lang)
    
    @property
    def chat_sample(self):

        if self.model == 'gpt3':
            return None

        samples = self.params['chat sample']['content']
        try: 
            samples = yaml.load(self.params['chat sample']['content'], Loader=yaml.FullLoader)
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

    def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        prompt = ''
        max_tokens = self.max_tokens
        # return_prompt = self.params['return prompt']
        prompt_template = self.params['prompt template']['content'].format(to_lang=to_lang).rstrip()
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
        for prompt, num_src in self._assemble_prompts(from_lang, to_lang, queries):
            ratelimit_attempt = 0
            retry_attempt = 0
            while True:
                try:
                    response = self._request_translation(prompt, chat_sample)
                    new_translations = re.split(r'<\|\d+\|>', response)[-num_src:]
                    if len(new_translations) != num_src:
                        raise InvalidNumTranslations
                    break
                except openai.error.RateLimitError: # Server returned ratelimit response
                    ratelimit_attempt += 1
                    if ratelimit_attempt >= 3:
                        new_translations = [''] * num_src
                        break
                    self.logger.warn(f'Restarting request due to ratelimiting by openai servers. Attempt: {ratelimit_attempt}')
                    time.sleep(2)
                except openai.error.APIError: # Server returned 500 error (probably server load)
                    retry_attempt += 1
                    if retry_attempt >= self.retry_attempts:
                        self.logger.error('OpenAI encountered a server error, possibly due to high server load. Use a different translator or try again later.')
                        new_translations = [''] * num_src
                        break
                    self.logger.warn(f'Restarting request due to a server error. Attempt: {retry_attempt}')
                    time.sleep(1)
                except openai.error.ServiceUnavailableError:
                    retry_attempt += 1
                    if retry_attempt >= self.retry_attempts:
                        self.logger.error('OpenAI encountered a server error, possibly due to high server load. Use a different translator or try again later.')
                        new_translations = [''] * num_src
                        break
                    self.logger.warn(f'Restarting request due to a server error. Attempt: {retry_attempt}')
                    time.sleep(2)
                except InvalidNumTranslations:
                    retry_attempt += 1
                    message = f'number of translations does not match to source:\nprompt:\n    {prompt}\ntranslations:\n  {new_translations}\nopenai response:\n  {response}'
                    if retry_attempt >= self.retry_attempts:
                        self.logger.error(message)
                        new_translations = [''] * num_src
                        break
                    self.logger.warn(message + '\n' + f'Restarting request. Attempt: {retry_attempt}')

            # if return_prompt:
            #     new_translations = new_translations[:-1]

            # if chat_sample is not None:
            #     new_translations = new_translations[1:]
            translations.extend([t.strip() for t in new_translations])

        if self.token_count_last:
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')

        return translations

    def _request_translation_gpt3(self, prompt: str) -> str:
        response = openai.Completion.create(
            model='text-davinci-003',
            prompt=prompt,
            max_tokens=self.max_tokens // 2, # Assuming that half of the tokens are used for the query
            temperature=self.temperature,
            top_p=self.top_p,
        )
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

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=self.max_tokens // 2,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        self.token_count += response.usage['total_tokens']
        self.token_count_last = response.usage['total_tokens']
        for choice in response.choices:
            if 'text' in choice:
                return choice.text

        # If no response with text is found, return the first response's content (which may be empty)
        return response.choices[0].message.content
    
    def _request_translation(self, prompt, chat_sample: List):
        openai.api_key = self.params['api key']
        model = self.model
        if model == 'gpt3':
            return self._request_translation_gpt3(prompt)
        elif model == 'gpt35-turbo':
            return self._request_translation_with_chat_sample(prompt, 'gpt-3.5-turbo-0613', chat_sample)
        elif model == 'gpt4':
            return self._request_translation_with_chat_sample(prompt, 'gpt-4-0613', chat_sample)
        else:
            raise Exception(f'Invalid GPT model: {model}')