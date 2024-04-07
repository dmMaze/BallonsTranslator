# stealt & modified from https://github.com/zyddnys/manga-image-translator/blob/main/manga_translator/translators/chatgpt.py

from http import client
import re
import time
from token import OP
from typing import List, Dict, Union
import traceback
import time

import openai

from .base import BaseTranslator, register_translator
from utils.error_handling import create_error_dialog

OPENAPI_V1_API = int(openai.__version__.split('.')[0]) >= 1


class InvalidNumTranslations(Exception):
    pass


@register_translator('Sakura')
class SakuraTranslator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'api baseurl': 'http://127.0.0.1:8080',
        'retry attempts': 1,
        'timeout': 999,
        'max tokens': 1024,
    }

    @property
    def max_tokens(self) -> int:
        return int(self.params['max tokens'])

    @property
    def timeout(self) -> int:
        return int(self.params['timeout'])

    @property
    def retry_attempts(self) -> int:
        return int(self.params['retry attempts'])

    @property
    def api_base_raw(self) -> str:
        return self.params['api baseurl']
    
    @property
    def api_base(self) -> str:
        url = self.api_base_raw
        if url.endswith('/'):
            url = url[:-1]
        if not url.endswith('/v1'):
            url += '/v1'
        return url


    _CHAT_SYSTEM_TEMPLATE = (
        '你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。'
    )

    def _setup_translator(self):
        self.lang_map['简体中文'] = 'Simplified Chinese'
        self.lang_map['日本語'] = 'Japanese'
        self.temperature = 0.3
        self.top_p = 0.3
        self.frequency_penalty = 0.0
        self._current_style = "normal"

    def detect_and_remove_extra_repeats(self, s: str, threshold: int = 10, remove_all=True):
        """
        检测字符串中是否有任何模式连续重复出现超过阈值，并在去除多余重复后返回新字符串。
        保留一个模式的重复。

        :param s: str - 待检测的字符串。
        :param threshold: int - 连续重复模式出现的最小次数，默认为2。
        :return: tuple - (bool, str)，第一个元素表示是否有重复，第二个元素是处理后的字符串。
        """

        repeated = False
        for pattern_length in range(1, len(s) // 2 + 1):
            i = 0
            while i < len(s) - pattern_length:
                pattern = s[i:i + pattern_length]
                count = 1
                j = i + pattern_length
                while j <= len(s) - pattern_length:
                    if s[j:j + pattern_length] == pattern:
                        count += 1
                        j += pattern_length
                    else:
                        break
                if count >= threshold:
                    repeated = True
                    # 保留一个模式的重复
                    if remove_all:
                        s = s[:i + pattern_length] + s[j:]
                    break
                i += 1
            if repeated:
                break
        return repeated, s

    def _format_prompt_log(self, prompt: str) -> str:
        return '\n'.join([
            'System:',
            self._CHAT_SYSTEM_TEMPLATE,
            'User:',
            '将下面的日文文本翻译成中文：',
            prompt,
        ])

    # str 通过/n转换为list
    def _split_text(self, text: str) -> list:
        if isinstance(text, list):
            return text
        return text.split('\n')

    def check_align(self, queries: List[str], response: str) -> bool:
        """
        检查原始文本（queries）与翻译后的文本（response）是否保持相同的行数。

        :param queries: 原始文本的列表。
        :param response: 翻译后的文本，可能是一个字符串。
        :return: 两者行数是否相同的布尔值。
        """
        # 确保response是列表形式
        translated_texts = self._split_text(
            response) if isinstance(response, str) else response

        # 日志记录，而不是直接打印
        self.logger.debug(
            f"原始文本行数: {len(queries)}, 翻译文本行数: {len(translated_texts)}")

        # 检查行数是否匹配
        is_aligned = len(queries) == len(translated_texts)
        if not is_aligned:
            self.logger.warning(
                f"原始文本与翻译文本的行数不匹配。原始文本行数: {len(queries)}, 翻译文本行数: {len(translated_texts)}")

        return is_aligned

    def _delete_quotation_mark(self, texts: List[str]) -> List[str]:
        print(texts)
        new_texts = []
        for text in texts:
            text = text.strip('「」')
            new_texts.append(text)
        return new_texts

    def _translate(self, src_list):

        queries = src_list
        translations = []
        self.logger.debug(
            f'Temperature: {self.temperature}, TopP: {self.top_p}')
        self.logger.debug(f'Queries: {queries}')
        text_prompt = '\n'.join(queries)
        self.logger.debug('-- Sakura Prompt --\n' +
                          self._format_prompt_log(text_prompt) + '\n\n')
        queries = [re.sub(r'[\U00010000-\U0010ffff]', '', query)
                   for query in queries]
        queries = [re.sub(r'❤', '♥', query) for query in queries]
        queries = [f'「{query}」' for query in queries]
        response = self._handle_translation_request(queries)
        self.logger.debug('-- Sakura Response --\n' + response + '\n\n')
        response = response.strip()
        rep_flag = self.detect_and_remove_extra_repeats(response)[0]
        if rep_flag:
            for i in range(self.retry_attempts):
                if self.detect_and_remove_extra_repeats(queries)[0]:
                    self.logger.warning('Queries have repeats.')
                    break
                self.logger.warning(
                    f'Re-translated because of model degradation, {i} times.')
                self._set_gpt_style("precise")
                self.logger.debug(
                    f'Temperature: {self.temperature}, TopP: {self.top_p}')
                response = self._handle_translation_request(queries)
                rep_flag = self.detect_and_remove_extra_repeats(response)[0]
                if not rep_flag:
                    break
            if rep_flag:
                self.logger.warning(
                    'Model degradation, try to translate single line.')
                for query in queries:
                    response = self._handle_translation_request(query)
                    translations.append(response)
                    rep_flag = self.detect_and_remove_extra_repeats(response)[
                        0]
                    if rep_flag:
                        self.logger.warning(
                            'Model degradation, fill original text')
                        return self._delete_quotation_mark(queries)
                return self._delete_quotation_mark(translations)
        align_flag = self.check_align(queries, response)
        if not align_flag:
            for i in range(self.retry_attempts):
                self.logger.warning(
                    f'Re-translated because of a mismatch in the number of lines, {i} times.')
                self._set_gpt_style("precise")
                self.logger.debug(
                    f'Temperature: {self.temperature}, TopP: {self.top_p}')
                response = self._handle_translation_request(queries)
                align_flag = self.check_align(queries, response)
                if align_flag:
                    break
            if not align_flag:
                self.logger.warning(
                    'Mismatch in the number of lines, try to translate single line.')
                for query in queries:
                    print(query)
                    response = self._handle_translation_request(query)
                    translations.append(response)
                    print(translations)
                align_flag = self.check_align(queries, translations)
                if not align_flag:
                    self.logger.warning(
                        'Mismatch in the number of lines, fill original text')
                    return self._delete_quotation_mark(queries)
                return self._delete_quotation_mark(translations)
        translations = self._split_text(response)
        if isinstance(translations, list):
            return self._delete_quotation_mark(translations)
        translations = self._split_text(response)
        return self._delete_quotation_mark(translations)

    def _handle_translation_request(self, prompt):
        server_error_attempt = 0
        if OPENAPI_V1_API:
            while True:
                try:
                    response = self._request_translation(prompt)
                    break
                except openai.APIError as e:
                    server_error_attempt += 1
                    if server_error_attempt >= self.retry_attempts:
                        create_error_dialog(e, 'Sakura translation failed. Return original text.', exception_type='SakuraTranslator')
                        return '\n'.join(prompt)
                    self.logger.warn(
                        f'Restarting request due to a server error. Attempt: {server_error_attempt}')
                    time.sleep(1)
                except openai.APIConnectionError as e:
                    server_error_attempt += 1
                    if server_error_attempt >= self.retry_attempts:
                        create_error_dialog(e, 'Sakura translation failed. Return original text.', exception_type='SakuraTranslator')
                        return '\n'.join(prompt)
                    self.logger.warn(
                        f'Restarting request due to a server connection error.Current API baseurl is "{self.api_base}" Attempt: {server_error_attempt}')
                    time.sleep(1)
                except FileNotFoundError:
                    self.logger.warn(
                        'Restarting request due to FileNotFoundError.')
                    time.sleep(30)
            return response
        else:
            self.logger.warning("You are utilizing an outdated version of the OpenAI Python API.")
            while True:
                try:
                    response = self._request_translation(prompt)
                    break
                except openai.error.APIError as e:
                    server_error_attempt += 1
                    if server_error_attempt >= self.retry_attempts:
                        create_error_dialog(e, 'Sakura translation failed. Return original text.', exception_type='SakuraTranslator')
                        return '\n'.join(prompt)
                    self.logger.warn(
                        f'Restarting request due to a server error. Attempt: {server_error_attempt}')
                    time.sleep(1)
                except openai.error.APIConnectionError as e:
                    server_error_attempt += 1
                    if server_error_attempt >= self.retry_attempts:
                        create_error_dialog(e, 'Sakura translation failed. Return original text.', exception_type='SakuraTranslator')
                        return '\n'.join(prompt)
                    self.logger.warn(
                        f'Restarting request due to a server connection error.Current API baseurl is "{self.api_base}" Attempt: {server_error_attempt}')
                    time.sleep(1)
                except FileNotFoundError:
                    self.logger.warn(
                        'Restarting request due to FileNotFoundError.')
                    time.sleep(30)
            return response

    def _request_translation(self, input_text_list):
        if isinstance(input_text_list, list):
            raw_text = "\n".join(input_text_list)
        else:
            raw_text = input_text_list
        extra_query = {
            'do_sample': False,
            'num_beams': 1,
            'repetition_penalty': 1.0,
        }
        if OPENAPI_V1_API:
            client = openai.Client(
                api_key="sk-114514",
                base_url=self.api_base
            )
            response = client.chat.completions.create(
                model="sukinishiro",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
                    },
                    {
                        "role": "user",
                        "content": f"将下面的日文文本翻译成中文：{raw_text}"
                    }
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                frequency_penalty=self.frequency_penalty,
                seed=-1,
                extra_query=extra_query,
            )
        else:
            openai.api_base = self.api_base
            openai.api_key = "sk-114514"
            response = openai.ChatCompletion.create(
                model="sukinishiro",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
                    },
                    {
                        "role": "user",
                        "content": f"将下面的日文文本翻译成中文：{raw_text}"
                    }
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                frequency_penalty=self.frequency_penalty,
                seed=-1,
                extra_query=extra_query,
            )

        for choice in response.choices:
            if OPENAPI_V1_API:
                return choice.message.content
            else:
                if 'text' in choice:
                    return choice.text

        # If no response with text is found, return the first response's content (which may be empty)
        return response.choices[0].message.content

    def _set_gpt_style(self, style_name: str):
        if self._current_style == style_name:
            return
        self._current_style = style_name
        if style_name == "precise":
            temperature, top_p = 0.1, 0.3
            frequency_penalty = 0.0
        elif style_name == "normal":
            temperature, top_p = 0.3, 0.3
            frequency_penalty = 0.15

        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty