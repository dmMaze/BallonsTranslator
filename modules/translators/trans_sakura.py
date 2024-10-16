# 同步更新自manga-image-translator

import logging
import re
import time
from typing import List, Dict, Union, Callable
import time
import os
import json

import openai

from .base import BaseTranslator, register_translator

OPENAPI_V1_API = int(openai.__version__.split('.')[0]) >= 1


class InvalidNumTranslations(Exception):
    pass


class SakuraDict():
    """
    Sakura字典类，用于加载和管理Sakura字典。

    属性:
    --------
    logger : logging.Logger
        日志记录器对象
    dict_str : str
        字典内容字符串
    version : str
        Sakura字典版本号
    path : str
        字典文件路径

    方法:
    --------
    __init__(self, path: str, logger: logging.Logger, version: str = "0.9") -> None:
        初始化Sakura字典对象。
    load_dict(self, dic_path: str) -> None:
        根据字典类型加载字典。
    get_dict_str(self) -> str:
        获取字典内容字符串。
    save_dict_to_file(self, dic_path: str, dict_type: str = "sakura") -> None:
        将字典内容保存到文件。

    """

    def __init__(self, path: str, logger: logging.Logger, version: str = "0.9") -> None:
        """
        初始化Sakura字典对象。

        参数:
        --------
        path : str
            字典文件路径
        logger : logging.Logger
            日志记录器对象
        version : str, optional
            Sakura字典版本号，默认为"0.9"

        """
        self.logger = logger
        self.dict_str = ""
        self.version = version
        self.path = path

        if not path:
            return  # 如果路径为空，直接返回，不加载字典

        if not os.path.exists(path):
            if self.version != "0.9":
                self.logger.info(f"字典文件不存在: {path}\n 如果您不需要字典功能，请忽略此警告。")
            return

        if self.version == "1.0":
            try:
                self.load_dict(path)
            except Exception as e:
                self.logger.warning(f"载入字典失败: {e}")
        elif self.version == "0.9":
            pass
        else:
            self.logger.info("您当前选择了Sakura 0.9版本，暂不支持术语表")

    def load_dict(self, dic_path: str) -> None:
        """
        根据字典类型加载字典。

        参数:
        --------
        dic_path : str
            字典文件路径

        """
        if self.version == "0.9" or not dic_path:
            return

        dic_type = self._detect_type(dic_path)
        if dic_type == "galtransl":
            self._load_galtransl_dic(dic_path)
        elif dic_type == "sakura":
            self._load_sakura_dict(dic_path)
        elif dic_type == "json":
            self._load_json_dict(dic_path)
        else:
            self.logger.warning(f"未知的字典类型: {dic_path}")

        self.logger.debug(f"字典内容（转换后）: {self.dict_str[:100]}")

    def _load_galtransl_dic(self, dic_path: str) -> None:
        """
        加载Galtransl格式的字典。

        参数:
        --------
        dic_path : str
            字典文件路径

        """
        if self.version == "0.9":
            return

        with open(dic_path, encoding="utf8") as f:
            dic_lines = f.readlines()
        if not dic_lines:
            return
        dic_name = os.path.basename(dic_path)
        gpt_dict = []
        for line in dic_lines:
            if line.startswith(("\n", "\\\\", "//")):
                continue
            line = line.replace("    ", "\t")
            sp = line.rstrip("\r\n").split("\t")
            if len(sp) < 2:
                continue
            src, dst, *info = sp
            gpt_dict.append(
                {"src": src, "dst": dst, "info": info[0] if info else None})
        gpt_dict_text_list = [
            f"{gpt['src']}->{gpt['dst']}{' #' + gpt['info'] if gpt['info'] else ''}" for gpt in gpt_dict]
        self.dict_str = "\n".join(gpt_dict_text_list)
        self.logger.info(f"载入 Galtransl 字典: {dic_name} {len(gpt_dict)}普通词条")

    def _load_sakura_dict(self, dic_path: str) -> None:
        """
        加载Sakura格式的字典。

        参数:
        --------
        dic_path : str
            字典文件路径

        """
        if self.version == "0.9":
            return

        with open(dic_path, encoding="utf8") as f:
            dic_lines = f.readlines()
        if not dic_lines:
            return
        dic_name = os.path.basename(dic_path)
        gpt_dict_text_list = []
        for line in dic_lines:
            if line.startswith(("\n", "\\\\", "//")):
                continue
            sp = line.rstrip("\r\n").split("->")
            if len(sp) < 2:
                continue
            src, dst_info = sp
            dst_info_sp = dst_info.split("#")
            dst = dst_info_sp[0].strip()
            info = dst_info_sp[1].strip() if len(dst_info_sp) > 1 else None
            gpt_dict_text_list.append(
                f"{src}->{dst}{' #' + info if info else ''}")
        self.dict_str = "\n".join(gpt_dict_text_list)
        self.logger.info(
            f"载入标准Sakura字典: {dic_name} {len(gpt_dict_text_list)}普通词条")

    def _load_json_dict(self, dic_path: str) -> None:
        """
        加载JSON格式的字典。

        参数:
        --------
        dic_path : str
            字典文件路径

        """
        if self.version == "0.9":
            return

        with open(dic_path, encoding="utf8") as f:
            dic_json = json.load(f)
        if not dic_json:
            return
        dic_name = os.path.basename(dic_path)
        gpt_dict_text_list = []
        for item in dic_json:
            if not item:
                continue
            src = item.get("src", "")
            dst = item.get("dst", "")
            info = item.get("info", "")
            gpt_dict_text_list.append(
                f"{src}->{dst}{' #' + info if info else ''}")
        self.dict_str = "\n".join(gpt_dict_text_list)
        self.logger.info(f"载入JSON字典: {dic_name} {len(gpt_dict_text_list)}条记录")

    def _detect_type(self, dic_path: str) -> str:
        """
        检测字典文件的类型。

        参数:
        --------
        dic_path : str
            字典文件路径

        返回:
        --------
        str
            字典类型，可能的值有"galtransl"、"sakura"、"json"和"unknown"

        """
        if self.version == "0.9":
            return "unknown"

        with open(dic_path, encoding="utf8") as f:
            dic_lines = f.readlines()
        self.logger.debug(f"检测字典类型: {dic_path}")
        if not dic_lines:
            return "unknown"
        if dic_path.endswith(".json"):
            return "json"
        for line in dic_lines:
            if line.startswith(("\n", "\\\\", "//")):
                continue
            if "\t" in line or "    " in line:
                return "galtransl"
            elif "->" in line:
                return "sakura"
        return "unknown"

    def get_dict_str(self) -> str:
        """
        获取字典内容字符串。

        返回:
        --------
        str
            字典内容字符串

        """
        if self.version == "0.9" or not self.path:
            return ""

        if not self.dict_str:
            try:
                self.load_dict(self.path)
            except Exception as e:
                self.logger.warning(f"载入字典失败: {e}")
        return self.dict_str
    
    def get_dict_str_within_text(self, text: str, force_apply_dict: bool = False) -> str:
        """
        获取字典内容字符串，仅保留字典中出现的词条。

        参数:
        --------
        text : str
            待翻译文本

        返回:
        --------
        str
            字典内容字符串

        """
        if force_apply_dict:
            return self.get_dict_str()
        if self.version == "0.9" or not self.path:
            return ""

        if not self.dict_str:
            try:
                self.load_dict(self.path)
            except Exception as e:
                self.logger.warning(f"载入字典失败: {e}")
                return ""

        # 初始化一个空列表用于存储匹配的字典行
        matched_dict_lines = []

        # 遍历字典中的每一行
        for line in self.dict_str.splitlines():
            if '->' in line:
                src = line.split('->')[0]
                # 检查 src 是否在输入文本中
                # self.logger.debug(f"检查字典原文{src}是否在文本{text}中")
                if src in text:
                    # self.logger.debug(f"匹配到字典行: {line}")
                    matched_dict_lines.append(line)

        # 将匹配的字典行拼接成一个字符串并返回
        return '\n'.join(matched_dict_lines)

    def dict_to_json(self) -> str:
        """
        将字典内容转换为JSON格式。

        返回:
        --------
        str
            字典内容的JSON格式字符串

        """
        if self.version == "0.9" or not self.path:
            return ""

        if not self.dict_str:
            try:
                self.load_dict(self.path)
            except Exception as e:
                self.logger.warning(f"载入字典失败: {e}")
        dict_json = []
        for line in self.dict_str.split("\n"):
            if not line:
                continue
            sp = line.split("->")
            if len(sp) < 2:
                continue
            src, dst_info = sp
            dst_info_sp = dst_info.split("#")
            dst = dst_info_sp[0].strip()
            info = dst_info_sp[1].strip() if len(dst_info_sp) > 1 else None
            dict_json.append({"src": src, "dst": dst, "info": info})
        return json.dumps(dict_json, ensure_ascii=False, indent=4)

    def save_dict_to_file(self, dic_path: str, dict_type: str = "sakura") -> None:
        """
        将字典内容保存到文件。

        参数:
        --------
        dic_path : str
            字典文件保存路径
        dict_type : str, optional
            字典类型，可选值有"sakura"、"galtransl"和"json"，默认为"sakura"

        """
        if self.version == "0.9" or not self.path:
            return

        if dict_type == "sakura":
            with open(dic_path, "w", encoding="utf8") as f:
                f.write(self.dict_str)
        elif dict_type == "galtransl":
            with open(dic_path, "w", encoding="utf8") as f:
                f.write(self.dict_str.replace(
                    "->", "    ").replace(" #", "    "))
        elif dict_type == "json":
            json_data = self.dict_to_json()
            with open(dic_path, "w", encoding="utf8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
        else:
            self.logger.warning(f"未知的字典类型: {dict_type}")

@register_translator('Sakura')
class SakuraTranslator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'low vram mode': {
            'value': True,
            'description': 'check it if you\'re running it locally on a single device and encountered a crash due to vram OOM',
            'type': 'checkbox',
        },
        'api baseurl': 'http://127.0.0.1:8080/v1',
        'dict path': '',
        'version': {
            'type': 'selector',
            'options': [
                '0.9',
                '1.0',
                'galtransl-v1'
            ],
            'value': '0.9'
        },
        'retry attempts': 3,
        'timeout': 999,
        'max tokens': 1024,
        'repeat detect threshold': 20,
        'force apply dict': {
            'value': False,
            'description': 'Force apply the dictionary regardless of whether the terms appear in the original text \n DO NOT CHECK THIS IF YOU ARE NOT SURE WHAT IT MEANS',
            'type': 'checkbox',
        },
    }

    _CHAT_SYSTEM_TEMPLATE_009 = (
        '你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。'
    )
    _CHAT_SYSTEM_TEMPLATE_100 = (
        '你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。'
    )

    _CHAT_SYSTEM_TEMPLATE_GALTRANSL_V1 = (
        '你是一个视觉小说翻译模型，可以通顺地使用给定的术语表以指定的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，注意不要混淆使役态和被动态的主语和宾语，不要擅自添加原文中没有的代词，也不要擅自增加或减少换行。'
    )

    @property 
    def timeout(self) -> int:
        return self.params['timeout']
    
    @property
    def retry_attempts(self) -> int:
        return self.params['retry attempts']
    
    @property
    def repeat_detect_threshold(self) -> int:
        return self.params['repeat detect threshold']

    @property
    def max_tokens(self) -> int:
        return self.params['max tokens']

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

    @property
    def sakura_version(self) -> str:
        return self.params['version']['value']

    @property
    def dict_path(self) -> str:
        return self.params['dict path']

    @property
    def force_apply_dict(self) -> bool:
        return self.params['force apply dict']['value']

    def _setup_translator(self):
        self.lang_map['简体中文'] = 'Simplified Chinese'
        self.lang_map['日本語'] = 'Japanese'
        self.temperature = 0.1
        self.top_p = 0.3
        self.frequency_penalty = 0.05
        self._current_style = "precise"
        self._emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]')
        self._heart_pattern = re.compile(r'❤')
        sakura_version = self.sakura_version if self.sakura_version!= 'galtransl-v1' else '1.0'
        self.sakura_dict = SakuraDict(
            self.dict_path, self.logger, sakura_version)
        self.logger.info(f'当前选择的Sakura版本: {self.sakura_version}')

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if param_key == 'dict path' or param_key == 'version':
            self.set_dict_path(self.params['dict path'])

    def set_dict_path(self, path: str):
        self.params['dict path'] = path
        self.sakura_dict = SakuraDict(path, self.logger, self.sakura_version)
        self.logger.debug(f'更新Sakura字典路径为: {path}')

    @staticmethod
    def enlarge_small_kana(text, ignore=''):
        """将小写平假名或片假名转换为普通大小

        参数
        ----------
        text : str
            全角平假名或片假名字符串。
        ignore : str, 可选
            转换时要忽略的字符。

        返回
        ------
        str
            平假名或片假名字符串，小写假名已转换为大写

        示例
        --------
        >>> print(enlarge_small_kana('さくらきょうこ'))
        さくらきようこ
        >>> print(enlarge_small_kana('キュゥべえ'))
        キユウべえ
        """
        SMALL_KANA = list('ぁぃぅぇぉゃゅょっァィゥェォヵヶャュョッ')
        SMALL_KANA_NORMALIZED = list('あいうえおやゆよつアイウエオカケヤユヨツ')
        SMALL_KANA2BIG_KANA = dict(
            zip(map(ord, SMALL_KANA), SMALL_KANA_NORMALIZED))

        def _exclude_ignorechar(ignore, conv_map):
            for character in map(ord, ignore):
                del conv_map[character]
            return conv_map

        def _convert(text, conv_map):
            return text.translate(conv_map)

        def _translate(text, ignore, conv_map):
            if ignore:
                _conv_map = _exclude_ignorechar(ignore, conv_map.copy())
                return _convert(text, _conv_map)
            return _convert(text, conv_map)

        return _translate(text, ignore, SMALL_KANA2BIG_KANA)

    def detect_and_calculate_repeats(self, s: str, threshold: int = 20, remove_all=True) -> Union[bool, str, int, str, int]:
        """
        检测文本中是否存在重复模式,并计算重复次数。
        返回值: (是否重复, 去除重复后的文本, 重复次数, 重复模式, 实际阈值)
        """

        # 初始化标记重复模式的变量
        repeated = False
        longest_pattern = ''  # 存储最长的重复模式
        longest_count = 0     # 存储最长模式的重复次数
        counts = []           # 存储所有找到的重复次数

        # 遍历所有可能的模式长度，从1到字符串长度的一半
        for pattern_length in range(1, len(s) // 2 + 1):
            # 构建正则表达式模式，匹配指定长度的重复模式
            pattern = re.compile(r'(.{%d})\1+' % pattern_length)

            # 查找所有匹配的重复模式
            for match in re.finditer(pattern, s):
                current_pattern = match.group(1)  # 当前找到的重复模式
                current_count = len(match.group(0)) // pattern_length  # 计算重复次数
                counts.append(current_count)  # 将当前模式的重复次数添加到 counts 列表

                # 如果当前模式的重复次数达到或超过阈值
                if current_count >= threshold:
                    self.logger.warning(f"检测到重复模式: {current_pattern}，重复次数: {current_count}")
                    repeated = True  # 标记检测到重复模式

                    # 如果当前模式的重复次数大于最长的重复次数
                    if current_count > longest_count:
                        longest_count = current_count  # 更新最长的重复次数
                        longest_pattern = current_pattern  # 更新最长的重复模式

                    # 如果需要移除所有重复模式
                    if remove_all:
                        s = s[:match.start()] + s[match.end():]  # 从字符串中移除重复模式
                    break  # 跳出当前循环，检查下一个模式长度

            if repeated:
                break  # 如果已经检测到重复模式，跳出外层循环

        # 计算实际阈值，取默认阈值和所有找到的重复次数的最大众数中的最大值
        actual_threshold = max(threshold, max(counts, default=0))

        # 返回检测结果，包括是否重复、去除重复后的文本、重复次数、重复模式和实际阈值
        return repeated, s, longest_count, longest_pattern, actual_threshold

    def _format_prompt_log(self, prompt: str) -> str:
        gpt_dict_raw_text = self.sakura_dict.get_dict_str_within_text(prompt, self.force_apply_dict)
        prompt_009 = '\n'.join([
            'System:',
            self._CHAT_SYSTEM_TEMPLATE_009,
            'User:',
            '将下面的日文文本翻译成中文：',
            prompt,
        ])
        prompt_100 = '\n'.join([
            'System:',
            self._CHAT_SYSTEM_TEMPLATE_100,
            'User:',
            "根据以下术语表（可以为空）：",
            gpt_dict_raw_text,
            "将下面的日文文本根据对应关系和备注翻译成中文：",
            prompt,
        ])
        prompt_galtransl_v1 = '\n'.join([
            'System:',
            self._CHAT_SYSTEM_TEMPLATE_GALTRANSL_V1,
            'User:',
            "根据以下术语表：",
            gpt_dict_raw_text,
            "将下面的日文文本根据上述术语表的对应关系和注释翻译成中文：",
            prompt,
        ])
        if self.sakura_version == '0.9':
            return prompt_009
        elif self.sakura_version == '1.0':
            return prompt_100
        else:
            return prompt_galtransl_v1

    def _split_text(self, text: str) -> List[str]:
        """
        将字符串按换行符分割为列表。
        """
        if isinstance(text, list):
            return text
        return text.split('\n')

    def _preprocess_queries(self, queries: List[str]) -> List[str]:
        """
        预处理查询文本,去除emoji,替换特殊字符,并添加「」标记。
        """
        queries = [self.enlarge_small_kana(query) for query in queries]
        queries = [self._emoji_pattern.sub('', query) for query in queries]
        queries = [self._heart_pattern.sub('♥', query) for query in queries]
        queries = [f'「{query}」' for query in queries]
        self.logger.debug(f'预处理后的查询文本：{queries}')
        return queries

    def _check_translation_quality(self, queries: List[str], response: str) -> List[str]:
        """
        检查翻译结果的质量,包括重复和行数对齐问题,如果存在问题则尝试重新翻译或返回原始文本。
        """
        def _retry_translation(queries: List[str], check_func: Callable[[str], bool], error_message: str) -> str:
            styles = ["precise", "normal", "aggressive", ]
            for i in range(self.retry_attempts):
                self._set_gpt_style(styles[i])
                self.logger.warning(
                    f'{error_message} 尝试次数: {i + 1}。当前参数风格：{self._current_style}。')
                response = self._handle_translation_request(queries)
                if not check_func(response):
                    return response
            return None

        # 检查请求内容是否含有超过默认阈值的重复内容
        if self.detect_and_calculate_repeats(''.join(queries), self.repeat_detect_threshold)[0]:
            self.logger.warning(
                f'请求内容本身含有超过默认阈值{self.repeat_detect_threshold}的重复内容。')

        # 根据译文众数和默认阈值计算实际阈值
        actual_threshold = max(max(self.detect_and_calculate_repeats(
            query)[4] for query in queries), self.repeat_detect_threshold)

        if self.detect_and_calculate_repeats(response, actual_threshold)[0]:
            response = _retry_translation(queries, lambda r: self.detect_and_calculate_repeats(
                r, actual_threshold)[0], f'检测到大量重复内容（当前阈值：{actual_threshold}），疑似模型退化，重新翻译。')
            if response is None:
                self.logger.warning(
                    f'疑似模型退化，尝试{self.retry_attempts}次仍未解决，进行单行翻译。')
                return self._translate_single_lines(queries)

        if not self.check_align(queries, response):
            response = _retry_translation(queries, lambda r: not self.check_align(
                queries, r), '因为检测到原文与译文行数不匹配，重新翻译。')
            if response is None:
                self.logger.warning(
                    f'原文与译文行数不匹配，尝试{self.retry_attempts}次仍未解决，进行单行翻译。')
                return self._translate_single_lines(queries)

        return self._split_text(response)

    def _translate_single_lines(self, queries: List[str]) -> List[str]:
        """
        逐行翻译查询文本。
        """
        translations = []
        for query in queries:
            response = self._handle_translation_request(query)
            if self.detect_and_calculate_repeats(response)[0]:
                self.logger.warning(f"单行翻译结果存在重复内容: {response}，返回原文。")
                translations.append(query)
            else:
                translations.append(response)
        return translations

    def check_align(self, queries: List[str], response: str) -> bool:
        """
        检查原始文本和翻译结果的行数是否对齐。
        """
        translations = self._split_text(response)
        is_aligned = len(queries) == len(translations)
        if not is_aligned:
            self.logger.warning(
                f"行数不匹配 - 原文行数: {len(queries)}，译文行数： {len(translations)}")
        return is_aligned

    def _delete_quotation_mark(self, texts: List[str]) -> List[str]:
        """
        删除文本中的「」标记。
        """
        new_texts = []
        for text in texts:
            text = text.strip('「」')
            new_texts.append(text)
        return new_texts

    def _translate(self, src_list) -> List[str]:
        self.logger.debug(
            f'Temperature: {self.temperature}, TopP: {self.top_p}')
        self.logger.debug(f'原文： {src_list}')
        text_prompt = '\n'.join(src_list)
        self.logger.debug('-- Sakura Prompt --\n' +
                          self._format_prompt_log(text_prompt) + '\n\n')

        # 预处理查询文本
        queries = self._preprocess_queries(src_list)

        # 发送翻译请求
        response = self._handle_translation_request(queries)
        self.logger.debug('-- Sakura Response --\n' + response + '\n\n')

        # 检查翻译结果是否存在重复或行数不匹配的问题
        translations = self._check_translation_quality(queries, response)

        return self._delete_quotation_mark(translations)

    def _handle_translation_request(self, prompt):
        ratelimit_attempt = 0
        server_error_attempt = 0
        timeout_attempt = 0
        while True:
            if OPENAPI_V1_API:
                try:
                    response = self._request_translation(prompt)
                    break
                except openai.RateLimitError:
                    ratelimit_attempt += 1
                    if ratelimit_attempt >= self.retry_attempts:
                        raise
                    self.logger.warning(
                        f'Sakura因被限速而进行重试。尝试次数： {ratelimit_attempt}')
                    time.sleep(2)
                except openai.APIError as e:
                    server_error_attempt += 1
                    if server_error_attempt >= self.retry_attempts:
                        self.logger.warning(e)
                        self.logger.warning('Sakura翻译失败。返回原始文本。')
                        return '\n'.join(prompt)
                    self.logger.warning(
                        f'Sakura因服务器错误而进行重试。 当前API baseurl为"{self.api_base}"，尝试次数： {server_error_attempt}, 错误信息： {e}')
                    time.sleep(1)
                except FileNotFoundError:
                    self.logger.warning(
                        'Sakura因文件不存在而进行重试。')
                    time.sleep(30)
                except TimeoutError:
                    timeout_attempt += 1
                    if timeout_attempt >= self.retry_attempts:
                        raise Exception('Sakura超时。')
                    self.logger.warning(
                        f'Sakura因超时而进行重试。尝试次数： {timeout_attempt}')
            else:
                try:
                    response = self._request_translation(prompt)
                    break
                except openai.error.RateLimitError:
                    ratelimit_attempt += 1
                    if ratelimit_attempt >= self.retry_attempts:
                        raise
                    self.logger.warning(
                        f'Sakura因被限速而进行重试。尝试次数： {ratelimit_attempt}')
                    time.sleep(2)
                except openai.error.APIError as e:
                    server_error_attempt += 1
                    if server_error_attempt >= self.retry_attempts:
                        self.logger.warning(
                            e, 'Sakura翻译失败。返回原始文本。')
                        return '\n'.join(prompt)
                    self.logger.warning(
                        f'Sakura因服务器错误而进行重试，请检查Sakura是否已经启动，API baseurl是否正确，并关闭一切代理软件后重试。\n 当前API baseurl为"{self.api_base}"，尝试次数： {server_error_attempt}, 错误信息： {e}')
                    time.sleep(1)
                except openai.error.APIConnectionError as e:
                    server_error_attempt += 1
                    if server_error_attempt >= self.retry_attempts:
                        self.logger.warning(
                            e, 'Sakura翻译失败。返回原始文本。')
                        return '\n'.join(prompt)
                    self.logger.warning(
                        f'Sakura因服务器连接错误而进行重试，请检查Sakura是否已经启动，API baseurl是否正确，并关闭一切代理软件后重试。\n 当前API baseurl为"{self.api_base}"，尝试次数： {server_error_attempt}, 错误信息： {e}')
                    time.sleep(1)
                except FileNotFoundError:
                    self.logger.warning(
                        'Sakura因文件不存在而进行重试。')
                    time.sleep(30)
                except TimeoutError:
                    timeout_attempt += 1
                    if timeout_attempt >= self.retry_attempts:
                        raise Exception('Sakura超时。')
                    self.logger.warning(
                        f'Sakura因超时而进行重试。尝试次数： {timeout_attempt}')

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
        gpt_dict_raw_text = self.sakura_dict.get_dict_str_within_text(raw_text, self.force_apply_dict)
        if self.sakura_version == "0.9" or gpt_dict_raw_text == "":
            messages = [
                {
                    "role": "system",
                    "content": f"{self._CHAT_SYSTEM_TEMPLATE_009}"
                },
                {
                    "role": "user",
                    "content": f"将下面的日文文本翻译成中文：{raw_text}"
                }
            ]
        elif self.sakura_version == "1.0":
            messages = [
                {
                    "role": "system",
                    "content": f"{self._CHAT_SYSTEM_TEMPLATE_100}"
                },
                {
                    "role": "user",
                    "content": f"根据以下术语表（可以为空）：\n{gpt_dict_raw_text}\n将下面的日文文本根据对应关系和备注翻译成中文：{raw_text}"
                }
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": f"{self._CHAT_SYSTEM_TEMPLATE_GALTRANSL_V1}"
                },
                {
                    "role": "user",
                    "content": f"根据以下术语表：\n{gpt_dict_raw_text}\n将下面的日文文本根据上述术语表的对应关系和注释翻译成中文：{raw_text}"
                }
            ]
        if OPENAPI_V1_API:
            client = openai.Client(
                api_key="sk-114514",
                base_url=self.api_base
            )
            response = client.chat.completions.create(
                model="sukinishiro",
                messages=messages,
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
                messages=messages,
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

        return response.choices[0].message.content

    def _set_gpt_style(self, style_name: str):
        """
        设置GPT的生成风格。
        """
        if self._current_style == style_name:
            return
        self._current_style = style_name
        if style_name == "precise":
            temperature, top_p = 0.1, 0.3
            frequency_penalty = 0.05
        elif style_name == "normal":
            temperature, top_p = 0.3, 0.3
            frequency_penalty = 0.2
        elif style_name == "aggressive":
            temperature, top_p = 0.3, 0.3
            frequency_penalty = 0.3

        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty