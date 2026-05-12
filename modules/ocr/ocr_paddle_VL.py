import numpy as np
import json
import cv2
import requests
import base64
from typing import List, Any

from .base import register_OCR, OCRBase, TextBlock
from utils.message import create_error_dialog, create_info_dialog


@register_OCR('paddle_vl')
class OCRPaddleVL(OCRBase):
    params = {
        'server_url': 'http://127.0.0.1:8080/layout-parsing',
        'prettifyMarkdown': {'type': 'checkbox', 'value': False},
        'visualize': {'type': 'checkbox', 'value': False},
        'description': 'Locally deployed Paddle OCR-VL service (POST /layout-parsing)'
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

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.debug = False

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        """
        Crop each text block and send it to the local Paddle-VL service.
        This keeps compatibility with the existing block-level TextBlock workflow.
        """
        im_h, im_w = img.shape[:2]
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if y2 < im_h and x2 < im_w and x1 >= 0 and y1 >= 0 and x1 < x2 and y1 < y2:
                try:
                    crop = img[y1:y2, x1:x2]
                    blk.text = self.ocr(crop)
                except Exception as e:
                    self.logger.exception('Paddle-VL block recognition failed')
                    blk.text = ['']
            else:
                self.logger.warning('invalid textbbox to target img')
                blk.text = ['']

    def ocr_img(self, img: np.ndarray) -> str:
        self.logger.debug(f'ocr_img: {img.shape}')
        return self.ocr(img)

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
        Convert Markdown to plain text:
        - Remove image syntax ![...](...)
        - Convert links [text](url) -> text
        - Remove heading markers
        - Remove emphasis markers
        - Remove inline code and HTML tags
        - Collapse repeated blank lines and trim whitespace
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

    def ocr(self, img: np.ndarray) -> str:
        """
        Send an image or cropped block to the local Paddle-VL `/layout-parsing`
        endpoint as base64. Prefer returned Markdown text, then extract text
        from prunedResult as a fallback.
        """
        try:
            image_bytes = cv2.imencode('.jpg', img)[1].tobytes()
        except Exception as e:
            self.logger.exception('Image encoding failed')
            raise

        image_b64 = base64.b64encode(image_bytes).decode('ascii')

        payload = {
            'file': image_b64,
            'fileType': 1,
            'prettifyMarkdown': self.prettifyMarkdown,
            'visualize': self.visualize,
        }

        try:
            resp = requests.post(self.server_url, json=payload, timeout=60)
        except Exception as e:
            self.logger.exception('Request to local Paddle-VL service failed')
            raise

        if resp.status_code != 200:
            self.logger.error(f'Paddle-VL request failed with status code: {resp.status_code}')
            raise ValueError(f'Paddle-VL request failed with status code: {resp.status_code}')

        try:
            data = resp.json()
        except Exception:
            self.logger.exception('Failed to parse Paddle-VL JSON response')
            raise

        # Standard Paddle service response: { logId, errorCode, errorMsg, result }
        if 'errorCode' in data and data.get('errorCode', -1) != 0:
            msg = data.get('errorMsg', '')
            self.logger.error(f'Paddle-VL returned an error: {msg}')
            raise ValueError(f'Paddle-VL returned an error: {msg}')

        result = data.get('result', data)
        lprs = result.get('layoutParsingResults') or []
        if not lprs:
            self.logger.debug('No layoutParsingResults found; returning full response text')
            return json.dumps(result, ensure_ascii=False)

        first = lprs[0]
        md_raw = first.get('markdown', {}).get('text') if isinstance(first.get('markdown'), dict) else None
        if md_raw:
            md_txt = self._markdown_to_text(md_raw)
            if md_txt:
                return md_txt

        pruned = first.get('prunedResult')
        if pruned is not None:
            texts = self._extract_texts_from_pruned(pruned)
            if texts:
                # join and clean result to remove any possible markdown artifacts
                joined = '\n'.join(texts)
                return self._markdown_to_text(joined)

        return json.dumps(first, ensure_ascii=False)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == 'server_url':
            create_info_dialog('Paddle-VL service URL updated')
