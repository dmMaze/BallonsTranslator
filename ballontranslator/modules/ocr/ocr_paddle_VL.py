import numpy as np
import json
import cv2
import requests
import base64
import time
from typing import List, Any

from .base import register_OCR, OCRBase
from ballontranslator.utils.message import create_info_dialog


@register_OCR('paddle_vl')
class OCRPaddleVL(OCRBase):
    params = {
        'server_url': {
            'value': 'http://127.0.0.1:8080/layout-parsing',
            'display_name': 'Server URL'
        },
        'prettifyMarkdown': {
            'type': 'checkbox',
            'value': False,
            'display_name': 'Prettify Markdown'
        },
        'visualize': {'type': 'checkbox', 'value': False},
        'max retry times': 3,
        'retry interval': 1.0,
        'description': '本地部署的 Paddle OCR-VL 服务 (POST /layout-parsing)'
    }

    @property
    def server_url(self):
        val = self.params.get('server_url')
        # UI may wrap param as a dict like {'value': 'http://...', 'data_type': <class 'str'>}
        if isinstance(val, dict):
            return val.get('value') or val.get('text') or ''
        return val or ''

    @property
    def prettifyMarkdown(self):
        v = self.params.get('prettifyMarkdown')
        if isinstance(v, dict):
            return bool(v.get('value', False))
        return bool(v)

    @property
    def visualize(self):
        v = self.params.get('visualize')
        if isinstance(v, dict):
            return bool(v.get('value', False))
        return bool(v)

    @property
    def max_retry_times(self):
        return max(1, int(self.get_param_value('max retry times')))

    @property
    def retry_interval(self):
        return max(0.0, float(self.get_param_value('retry interval')))

    def __init__(self, **params) -> None:
        super().__init__(**params)

    def _extract_texts_from_pruned(self, pruned: Any) -> List[str]:
        texts: List[str] = []

        def walk(node: Any):
            if node is None:
                return
            if isinstance(node, dict):
                # common keys may include 'texts' or 'text'
                if 'texts' in node and isinstance(node['texts'], (list, str)):
                    if isinstance(node['texts'], list):
                        texts.append(''.join(node['texts']).strip())
                    else:
                        texts.append(str(node['texts']).strip())
                if 'text' in node and isinstance(node['text'], str):
                    texts.append(node['text'].strip())
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for it in node:
                    walk(it)
            elif isinstance(node, str):
                texts.append(node.strip())

        walk(pruned)
        # filter empties and deduplicate nearby
        return [t for t in texts if t]

    def _markdown_to_text(self, md: str) -> str:
        """
        简单地把 Markdown 转换为纯文本：
        - 移除图片语法 ![...](...)
        - 把链接 [text](url) -> text
        - 移除标题前导的 #
        - 移除强调符号 (*, _, **)
        - 移除行内代码和 HTML 标签
        - 合并连续空行并去除前后空白
        """
        if not md:
            return ''
        try:
            import re

            # remove image markdown
            md = re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', md)
            # replace links [text](url) -> text
            md = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', md)
            # remove heading markers at line starts
            md = re.sub(r'(?m)^\s{0,3}#{1,6}\s*', '', md)
            # remove bold/italic markers (*, _, **, __)
            md = re.sub(r'(\*\*|__)(.*?)\1', r'\2', md)
            md = re.sub(r'(\*|_)(.*?)\1', r'\2', md)
            # remove inline code backticks
            md = re.sub(r'`([^`]*)`', r'\1', md)
            # remove any remaining html tags
            md = re.sub(r'<[^>]+>', '', md)
            # normalize whitespace and remove multiple blank lines
            md = re.sub(r"\r\n|\r", "\n", md)
            md = re.sub(r"\n{2,}", "\n", md)
            md = md.strip()
            return md
        except Exception:
            return md

    def _request_ocr(self, payload: dict) -> str:
        try:
            resp = requests.post(self.server_url, json=payload, timeout=60)
        except Exception:
            self.logger.exception('请求本地 Paddle-VL 服务失败')
            raise

        if resp.status_code != 200:
            self.logger.error(f'Paddle-VL 请求失败，状态码：{resp.status_code}')
            raise ValueError(f'Paddle-VL 请求失败，状态码：{resp.status_code}')

        try:
            data = resp.json()
        except Exception:
            self.logger.exception('Paddle-VL 响应解析 JSON 失败')
            raise

        # Paddle 服务标准返回: { logId, errorCode, errorMsg, result }
        if 'errorCode' in data and data.get('errorCode', -1) != 0:
            msg = data.get('errorMsg', '')
            self.logger.error(f'Paddle-VL 返回错误：{msg}')
            raise ValueError(f'Paddle-VL 返回错误：{msg}')

        result = data.get('result', data)
        lprs = result.get('layoutParsingResults') or []
        if not lprs:
            # 没有 layoutParsingResults，则尝试直接从 result 中解析
            # 最后退回到将整个响应以字符串返回（用于调试）
            self.logger.debug('未找到 layoutParsingResults，返回完整响应文本')
            return json.dumps(result, ensure_ascii=False)

        first = lprs[0]
        # 优先使用 Markdown，但把 Markdown 清理为纯文本
        md_raw = first.get('markdown', {}).get('text') if isinstance(first.get('markdown'), dict) else None
        if md_raw:
            md_txt = self._markdown_to_text(md_raw)
            if md_txt:
                return md_txt

        # 否则尝试从 prunedResult 中抽取 texts 字段
        pruned = first.get('prunedResult')
        if pruned is not None:
            texts = self._extract_texts_from_pruned(pruned)
            if texts:
                # join and clean result to remove any possible markdown artifacts
                joined = '\n'.join(texts)
                return self._markdown_to_text(joined)

        # 最后退回到 outputImages 或 pruned 的 JSON 字符串
        return json.dumps(first, ensure_ascii=False)

    def ocr_img(self, img: np.ndarray, **kwargs) -> str:
        """
        将图片（单张或块）以 Base64 发送到本地 Paddle-VL 服务的 `/layout-parsing`。
        优先使用返回的 Markdown 文本；若无，则尝试从 prunedResult 中抽取文本。
        返回字符串（整块识别结果）。
        """
        try:
            image_bytes = cv2.imencode('.jpg', img)[1].tobytes()
        except Exception:
            self.logger.exception('图片编码失败')
            raise

        payload = {
            'file': base64.b64encode(image_bytes).decode('ascii'),
            'fileType': 1,
            'prettifyMarkdown': self.prettifyMarkdown,
            'visualize': self.visualize,
        }

        last_error = None
        attempts = self.max_retry_times
        for attempt in range(attempts):
            try:
                return self._request_ocr(payload)
            except Exception as e:
                last_error = e
                if attempt + 1 >= attempts:
                    break
                self.logger.warning(
                    f'Paddle-VL OCR failed, retrying {attempt + 1}/{attempts}: {e}'
                )
                time.sleep(self.retry_interval)

        raise RuntimeError(f'Paddle-VL OCR failed after {attempts} attempts: {last_error}') from last_error

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        # 当 server_url 等参数变动时，提示用户
        if param_key == 'server_url':
            create_info_dialog('Paddle-VL 服务地址已更新')
