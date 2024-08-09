import requests
from PIL import Image
import filetype
import io
import time
import json5
import lxml.html
import http.cookiejar as cookielib
import logging

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
        self.logger = logging.getLogger('LensCore')
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

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
        self.logger = logging.getLogger('LensAPI')
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

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

# Пример использования:
# api = LensAPI()
# result = api.process_image(image_path='path/to/your/image.jpg')
# print(result['full_text'])
# print(result['language'])
