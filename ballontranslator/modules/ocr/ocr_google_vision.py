import re
import time
import base64
import json
import cv2
import numpy as np
from typing import List
import httpx

from .base import register_OCR, OCRBase, TextBlock

import logging
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


@register_OCR('google_vision')
class OCRGoogleVisionAPI(OCRBase):
    params = {
        'api_key': '',
        'language_hints': {
            'value': '',
            'description': 'Language codes separated by commas (BCP-47)'
        },
        'proxy': {
            'value': '',
            'description': 'Proxy address (e.g., http(s)://user:password@host:port or socks4/5://user:password@host:port)'
        },
        'delay': 0.0,
        'newline_handling': {
            'type': 'selector',
            'options': [
                'preserve',
                'remove'
            ],
            'value': 'preserve',
            'description': 'Choose how to handle newline characters in OCR results'
        },
        'no_uppercase': {
            'type': 'checkbox',
            'value': False,
            'description': 'Convert text to lowercase except the first letter of each sentence'
        },
        'description': 'OCR using Google Vision API'
    }

    @property
    def request_delay(self):
        try:
            return float(self.get_param_value('delay'))
        except (ValueError, TypeError):
            return 1.0

    @property
    def language_hints(self):
        hints = self.get_param_value('language_hints')
        return [hint.strip() for hint in hints.split(",")] if hints else None

    @property
    def api_key(self):
        return self.get_param_value('api_key')

    @property
    def proxy(self):
        return self.get_param_value('proxy')

    @property
    def newline_handling(self):
        return self.get_param_value('newline_handling')

    @property
    def no_uppercase(self):
        return self.get_param_value('no_uppercase')

    def __init__(self, **params) -> None:
        if 'delay' in params:
            try:
                params['delay'] = float(params['delay'])
            except (ValueError, TypeError):
                params['delay'] = 1.0  
        super().__init__(**params)
        self.proxy_url = self.proxy  
        self.last_request_time = 0

    def send_to_google_vision(self, image_buffer: bytes):
        VISION_API_URL = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"

        image_content = base64.b64encode(image_buffer).decode("utf-8")

        request_body = {
            "requests": [
                {
                    "image": {
                        "content": image_content
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION"
                        }
                    ]
                }
            ]
        }

        if self.language_hints:
            request_body["requests"][0]["imageContext"] = {
                "languageHints": self.language_hints
            }

        headers = {
            "Content-Type": "application/json"
        }

        client_kwargs = {'headers': headers} 
        if self.proxy_url: 
            mounts = {}
            if self.proxy_url.startswith(('http://', 'https://', 'socks4://', 'socks5://')): 
                mounts["all://"] = httpx.HTTPTransport(proxy=self.proxy_url) 
            else:
                self.logger.warning("The proxy URL does not contain a schema (http://, https://, socks4://, socks5://). The proxy may not work.")
                mounts["all://"] = httpx.HTTPTransport(proxy=self.proxy_url) 
            client_kwargs['mounts'] = mounts 

        with httpx.Client(**client_kwargs) as client: 
            try:
                if self.debug_mode:
                    proxy_info = self.proxy_url if self.proxy_url else "No proxy"
                    self.logger.debug(f"Sending request to Google Vision API with proxy: {proxy_info}")

                response = client.post(VISION_API_URL, headers=headers, json=request_body)
                response.raise_for_status() 

                return response.json()
            except httpx.HTTPError as e:
                raise Exception(f"Error during request to Google Vision API: {e}")

    def extract_text_and_coordinates(self, annotations):
        text_with_coords = []
        for annotation in annotations:
            if 'description' in annotation:
                words = annotation.get('description', '').split()
                vertices = annotation.get('boundingPoly', {}).get('vertices', [])
                text_with_coords.append({
                    "text": annotation['description'],
                    "coordinates": [(v.get("x", 0), v.get("y", 0)) for v in vertices]
                })
        return text_with_coords

    def extract_full_text(self, response_json):
        try:
            return response_json['responses'][0]['fullTextAnnotation']['text']
        except (IndexError, KeyError, TypeError):
            return "Full text not found or not recognized"

    def process_image(self, image_buffer: bytes):
        response = self.send_to_google_vision(image_buffer)
        full_text = self.extract_full_text(response)

        return {
            'full_text': full_text,
            'language': response['responses'][0].get('language', 'und'),
            'text_with_coordinates': self.extract_text_and_coordinates(response.get("responses", [{}])[0].get("textAnnotations", []))
        }

    def format_ocr_result(self, result):
        formatted_result = {
            "language": result.get("language", ""),
            "full_text": result.get("full_text", ""),
            "text_with_coordinates": [
                f"{item['text']}: {item['coordinates']}"
                for item in result.get("text_with_coordinates", [])
            ]
        }
        return json.dumps(formatted_result, indent=4, ensure_ascii=False)

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        im_h, im_w = img.shape[:2]
        if self.debug_mode:
            self.logger.debug(f'Image dimensions: {im_h}x{im_w}')
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if self.debug_mode:
                self.logger.debug(f'Processing block: ({x1}, {y1}, {x2}, {y2})')
            if y2 < im_h and x2 < im_w and x1 >= 0 and y1 >= 0 and x1 < x2 and y1 < y2:
                cropped_img = img[y1:y2, x1:x2]
                if self.debug_mode:
                    self.logger.debug(f'Cropped image dimensions: {cropped_img.shape}')
                blk.text = self.ocr(cropped_img)
            else:
                if self.debug_mode:
                    self.logger.warning('Invalid text block coordinates')
                blk.text = ''

    def ocr_img(self, img: np.ndarray) -> str:
        return self.ocr(img)

    def ocr(self, img: np.ndarray) -> str:
        if self.debug_mode:
            self.logger.debug(f'Starting OCR on image of shape: {img.shape}')
        self._respect_delay()
        try:
            if img.size > 0:
                if self.debug_mode:
                    self.logger.debug(f'Input image size: {img.shape}')
                _, buffer = cv2.imencode('.jpg', img)
                result = self.process_image(buffer.tobytes())
                if self.debug_mode:
                    formatted_result = self.format_ocr_result(result)
                    self.logger.debug(f'OCR result: {formatted_result}')

                ignore_texts = [
                    'Full text not found or not recognized'
                ]
                if result['full_text'] in ignore_texts:
                    return ''
                full_text = result['full_text']
                if self.newline_handling == 'remove':
                    full_text = full_text.replace('\n', ' ')

                full_text = self._apply_punctuation_and_spacing(full_text)

                if self.no_uppercase:
                    full_text = self._apply_no_uppercase(full_text)

                return full_text
            else:
                if self.debug_mode:
                    self.logger.warning('Empty image for OCR')
                return ''
        except Exception as e:
            if self.debug_mode:
                self.logger.error(f"OCR error: {str(e)}")
            return ''

    def _apply_no_uppercase(self, text: str) -> str:
        def process_sentence(sentence):
            words = sentence.split()
            if not words:
                return ''
            processed = [words[0].capitalize()] + [word.lower() for word in words[1:]]
            return ' '.join(processed)

        sentences = re.split(r'(?<=[.!?…])\s+', text)
        processed_sentences = [process_sentence(sentence) for sentence in sentences]

        return ' '.join(processed_sentences)

    def _apply_punctuation_and_spacing(self, text: str) -> str:
        text = re.sub(r'\s+([,.!?…])', r'\1', text)
        text = re.sub(r'([,.!?…])(?!\s)(?![,.!?…])', r'\1 ', text)
        text = re.sub(r'([,.!?…])\s+([,.!?…])', r'\1\2', text)
        return text.strip()

    def _respect_delay(self):
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if self.debug_mode:
            self.logger.info(f'Time since last request: {time_since_last_request} seconds')

        if time_since_last_request < self.request_delay:
            sleep_time = self.request_delay - time_since_last_request
            if self.debug_mode:
                self.logger.info(f'Waiting {sleep_time} seconds before next request')
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def updateParam(self, param_key: str, param_content):
        if param_key == 'delay':
            try:
                param_content = float(param_content)
            except (ValueError, TypeError):
                param_content = 1.0
        super().updateParam(param_key, param_content)
        if param_key == 'proxy':
            self.proxy_url = param_content 