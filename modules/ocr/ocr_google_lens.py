import numpy as np
import time
import cv2
import re
from typing import List

import requests
from PIL import Image
import io
import time
import json5
import lxml.html
import http.cookiejar as cookielib

from .base import register_OCR, OCRBase, TextBlock

class LensCore:
    LENS_ENDPOINT = 'https://lens.google.com/v3/upload'
    SUPPORTED_MIMES = [
        'image/x-icon', 'image/bmp', 'image/jpeg',
        'image/png', 'image/tiff', 'image/webp', 'image/heic'
    ]
    HEADERS = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Origin': 'https://lens.google.com',
        'Referer': 'https://lens.google.com/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }

    def __init__(self):
        self.cookie_jar = cookielib.CookieJar()
        self.session = requests.Session()
        self.session.cookies = self.cookie_jar

    def scan_by_data(self, data, mime, dimensions):
        headers = self.HEADERS.copy()
        files = {
            'encoded_image': ('image.jpg', data, mime),
            'original_width': (None, str(dimensions[0])),
            'original_height': (None, str(dimensions[1])),
            'processed_image_dimensions': (None, f"{dimensions[0]},{dimensions[1]}")
        }
        response = self.session.post(self.LENS_ENDPOINT, headers=headers, files=files)
        if response.status_code != 200:
            raise Exception(f"Failed to upload image. Status code: {response.status_code}")
        
        tree = lxml.html.parse(io.StringIO(response.text))
        r = tree.xpath("//script[@class='ds:1']")
        return json5.loads(r[0].text[len("AF_initDataCallback("):-2])


class Lens(LensCore):
    @staticmethod
    def resize_image(image, max_size=(1000, 1000)):
        image.thumbnail(max_size)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue(), image.size

    def scan_by_file(self, file_path):
        with Image.open(file_path) as img:
            img_data, dimensions = self.resize_image(img)
        return self.scan_by_data(img_data, 'image/jpeg', dimensions)

    def scan_by_buffer(self, buffer):
        img = Image.open(io.BytesIO(buffer))
        img_data, dimensions = self.resize_image(img)
        return self.scan_by_data(img_data, 'image/jpeg', dimensions)


class LensAPI:
    def __init__(self):
        self.lens = Lens()

    @staticmethod
    def extract_full_text(data):
        try:
            text_data = data[3][4][0][0]
            if isinstance(text_data, list):
                return "\n".join(text_data)
            return text_data
        except (IndexError, TypeError):
            return "Full text not found(or Lens could not recognize it)"

    @staticmethod
    def extract_language(data):
        try:
            return data[3][3]
        except (IndexError, TypeError):
            return "Language not found in expected structure"

    def process_image(self, image_path=None, image_buffer=None):
        if image_path:
            result = self.lens.scan_by_file(image_path)
        elif image_buffer:
            result = self.lens.scan_by_buffer(image_buffer)
        else:
            raise ValueError("Either image_path or image_buffer must be provided")

        return {
            'full_text': self.extract_full_text(result['data']),
            'language': self.extract_language(result['data'])
        }


@register_OCR('google_lens')
class OCRLensAPI(OCRBase):
    params = {
        "delay": 1.0,
        'newline_handling': {
            'type': 'selector',
            'options': [
                'preserve',
                'remove'
            ],
            'value': 'preserve',
            'description': 'Choose how to handle newline characters in OCR result'
        },
        'no_uppercase': {
            'type': 'checkbox',
            'value': False,
            'description': 'Convert text to lowercase except the first letter of each sentence'
        },
        'description': 'OCR using Google Lens API'
    }
    
    @property
    def request_delay(self):
        return self.get_param_value('delay')
    
    @property
    def newline_handling(self):
        return self.get_param_value('newline_handling')
    
    @property
    def no_uppercase(self):
        return self.get_param_value('no_uppercase')

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.api = LensAPI()
        self.last_request_time = 0

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        im_h, im_w = img.shape[:2]
        if self.debug_mode:
            self.logger.info(f'Image size: {im_h}x{im_w}')
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if self.debug_mode:
                self.logger.info(f'Processing block: ({x1, y1, x2, y2})')
            if y2 < im_h and x2 < im_w and x1 > 0 and y1 > 0 and x1 < x2 and y1 < y2:
                cropped_img = img[y1:y2, x1:x2]
                if self.debug_mode:
                    self.logger.info(f'Cropped image size: {cropped_img.shape}')
                blk.text = self.ocr(cropped_img)
            else:
                if self.debug_mode:
                    self.logger.warning('Invalid text bbox to target image')
                blk.text = ['']

    def ocr_img(self, img: np.ndarray) -> str:
        if self.debug_mode:
            self.logger.debug(f'ocr_img: {img.shape}')
        return self.ocr(img)

    def ocr(self, img: np.ndarray) -> str:
        if self.debug_mode:
            self.logger.info(f'Starting OCR on image of shape: {img.shape}')
        self._respect_delay()
        try:
            if img.size > 0:  # Check if the image is not empty
                if self.debug_mode:
                    self.logger.info(f'Input image size: {img.shape}')
                _, buffer = cv2.imencode('.jpg', img)
                result = self.api.process_image(image_buffer=buffer.tobytes())
                if self.debug_mode:
                    self.logger.info(f'OCR result: {result}')
                # Check the result for the specified text
                ignore_texts = [
                    'Full text not found in expected structure',
                    'Full text not found(or Lens could not recognize it)'
                ]
                if result['full_text'] in ignore_texts:
                    return ''
                full_text = result['full_text']
                if self.newline_handling == 'remove':
                    full_text = full_text.replace('\n', ' ')
                
                # Apply punctuation and spacing rules
                full_text = self._apply_punctuation_and_spacing(full_text)

                if self.no_uppercase:
                    full_text = self._apply_no_uppercase(full_text)

                if isinstance(full_text, list):
                    return '\n'.join(full_text)
                else:
                    return full_text
            else:
                if self.debug_mode:
                    self.logger.warning('Empty image provided for OCR')
                return ''
        except Exception as e:
            if self.debug_mode:
                self.logger.error(f"OCR error: {str(e)}")
            return ''

    def _apply_no_uppercase(self, text: str) -> str:
        def process_sentence(sentence):
            # Split the sentence into words, preserving punctuation
            words = re.findall(r'\S+|\s+', sentence)
            processed_words = []
            for i, word in enumerate(words):
                if i == 0 or words[i-1].strip() in '.!?…':
                    processed_words.append(word.capitalize())
                else:
                    processed_words.append(word.lower())
            return ''.join(processed_words)

        # Split the text into sentences, preserving original spacing and punctuation
        sentences = re.split(r'(?<=[.!?…])', text)
        processed_sentences = [process_sentence(sentence) for sentence in sentences]
        
        return ''.join(processed_sentences)

    def _apply_punctuation_and_spacing(self, text: str) -> str:
        # Remove extra spaces before punctuation
        text = re.sub(r'\s+([,.!?…])', r'\1', text)
        
        # Ensure single space after punctuation, except for multiple punctuation marks
        text = re.sub(r'([,.!?…])(?!\s)(?![,.!?…])', r'\1 ', text)
        
        # Remove space between multiple punctuation marks
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
                self.logger.info(f'Sleeping for {sleep_time} seconds')
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
