import numpy as np
import cv2
from typing import Tuple, List
import requests
import base64

from .base import register_textdetectors, TextDetectorBase, TextBlock


@register_textdetectors('stariver_ocr')
class StariverDetector(TextDetectorBase):

    params = {
        'token': "Replace with your token",
        'expand_ratio': "0.01",
        "refine": {
            'type': 'checkbox',
            'value': True
        },
        "filtrate": {
            'type': 'checkbox',
            'value': True
        },
        "disable_skip_area": {
            'type': 'checkbox',
            'value': True
        },
        "detect_scale": "3",
        "merge_threshold": "2.0",
        "low_accuracy_mode": {
            'type': 'checkbox',
            'value': False
        },
        "force_expand":{
            'type': 'checkbox',
            'value': False
        },
        "font_size_offset": "0",
        "font_size_min(set to -1 to disable)": "-1",
        "font_size_max(set to -1 to disable)": "-1",
        'description': '星河云(团子翻译器) OCR 文字检测器'
    }

    @property
    def token(self):
        return self.params['token']

    @property
    def expand_ratio(self):
        return float(self.params['expand_ratio'])

    @property
    def refine(self):
        return self.params['refine']['value']

    @property
    def filtrate(self):
        return self.params['filtrate']['value']

    @property
    def disable_skip_area(self):
        return self.params['disable_skip_area']['value']

    @property
    def detect_scale(self):
        return int(self.params['detect_scale'])

    @property
    def merge_threshold(self):
        return float(self.params['merge_threshold'])

    @property
    def low_accuracy_mode(self):
        return self.params['low_accuracy_mode']['value']
        
    @property
    def force_expand(self):
        return self.params['force_expand']['value']
        
    @property
    def font_size_offset(self):
        return int(self.params['font_size_offset'])

    @property
    def font_size_min(self):
        return int(self.params['font_size_min(set to -1 to disable)'])

    @property
    def font_size_max(self):
        return int(self.params['font_size_max(set to -1 to disable)'])

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.url = 'https://dl.ap-sh.starivercs.cn/v2/manga_trans/advanced/manga_ocr'
        self.debug = False
        # self.name = 'StariverDetector'

    def adjust_font_size(self, original_font_size):
        new_font_size = original_font_size + self.font_size_offset
        if self.font_size_min != -1:
            new_font_size = max(new_font_size, self.font_size_min)
        if self.font_size_max != -1:
            new_font_size = min(new_font_size, self.font_size_max)
        return new_font_size

    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, List[TextBlock]]:
        if not self.token or self.token == 'Replace with your token':
            self.logger.error(f'token 没有设置。当前token：{self.token}')
            raise ValueError('token 没有设置。')
        img_encoded = cv2.imencode('.jpg', img)[1]
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        payload = {
            "token": self.token,
            "mask": True,
            "refine": self.refine,
            "filtrate": self.filtrate,
            "disable_skip_area": self.disable_skip_area,
            "detect_scale": self.detect_scale,
            "merge_threshold": self.merge_threshold,
            "low_accuracy_mode": self.low_accuracy_mode,
            "force_expand": self.force_expand,
            "image": img_base64
        }
        if self.debug:
            payload_log = {k: v for k, v in payload.items() if k != 'image'}
            self.logger.debug(f'请求参数：{payload_log}')
        response = requests.post(self.url, json=payload)
        if response.status_code != 200:
            self.logger.error(f'请求失败，状态码：{response.status_code}')
            if response.json().get('Code', -1) != 0:
                self.logger.error(f'错误信息：{response.json().get("Message", "")}')
                with open('stariver_ocr_error.txt', 'w', encoding='utf-8') as f:
                    f.write(response.text)
            raise ValueError('请求失败。')
        response_data = response.json()['Data']

        blk_list = []
        for block in response_data.get('text_block', []):
            xyxy = [int(min(coord[0] for coord in block['block_coordinate'].values())),
                    int(min(coord[1]
                        for coord in block['block_coordinate'].values())),
                    int(max(coord[0]
                        for coord in block['block_coordinate'].values())),
                    int(max(coord[1] for coord in block['block_coordinate'].values()))]
            lines = [np.array([[coord[pos][0], coord[pos][1]] for pos in ['upper_left', 'upper_right',
                              'lower_right', 'lower_left']], dtype=np.float32) for coord in block['coordinate']]
            texts = [text.replace('<skip>', '') for text in block.get('texts', [])]

            original_font_size = block.get('text_size', 0)

            font_size_recalculated = self.adjust_font_size(original_font_size)

            if self.debug:
                self.logger.debug(f'原始字体大小：{original_font_size}，修正后字体大小：{font_size_recalculated}')

            blk = TextBlock(
                xyxy=xyxy,
                lines=lines,
                language=block.get('language', 'unknown'),
                vertical=block.get('is_vertical', False),
                font_size=font_size_recalculated,

                text=texts,
                fg_colors=np.array(block.get('foreground_color', [
                                   0, 0, 0]), dtype=np.float32),
                bg_colors=np.array(block.get('background_color', [
                                   0, 0, 0]), dtype=np.float32)
            )
            blk_list.append(blk)
            if self.debug:
                self.logger.debug(f'检测到文本块：{blk.to_dict()}')

        mask = self._decode_base64_mask(response_data['mask'])
        mask = self.expand_mask(mask)
        return mask, blk_list

    @staticmethod
    def _decode_base64_mask(base64_str: str) -> np.ndarray:
        img_data = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        mask = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        return mask

    def expand_mask(self, mask: np.ndarray, expand_ratio: float = 0.01) -> np.ndarray:
        """
        在mask的原始部分上扩展mask，以便于提取更大的文字区域。
        :param mask: 输入的mask
        :param expand_ratio: 扩展比例，默认值为0.01
        :return: 扩展后的mask
        """

        if expand_ratio == 0:
            return mask
        
        # 确保mask是二值图像（只含0和255）
        mask = (mask > 0).astype(np.uint8) * 255

        # 获得图像的尺寸
        height, width = mask.shape

        # 计算kernel的大小（取图像尺寸的一部分，按比例expand_ratio）
        kernel_size = int(min(height, width) * expand_ratio)
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保kernel尺寸是奇数

        # 创建一个正方形的kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # 执行膨胀操作
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # 计算扩展后的mask
        dilated_mask = (dilated_mask > 0).astype(np.uint8) * 255

        return dilated_mask