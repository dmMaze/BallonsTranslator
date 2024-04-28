import cv2
import requests
import json
import base64
import numpy as np
from utils.textblock import TextBlock

class StariverOCR:
    def __init__(self, token):
        self.token = token
        self.url = 'https://dl.ap-sh.starivercs.cn/v2/manga_trans/advanced/manga_ocr'

    def ocr(self, img: np.ndarray):
        img = cv2.imencode('.png', img)[1]
        img_base64 = base64.b64encode(img).decode('utf-8')
        data = {
            "token": self.token,
            "mask": False,
            "refine": True,
            "filtrate": True,
            "disable_skip_area": True,
            "detect_scale": 3,
            "merge_threshold": 0.5,
            "low_accuracy_mode": False,
            "image": img_base64
        }
        response = requests.post(self.url, data=json.dumps(data))
        text_blocks = response.json()['Data']['text_block']
        texts = [text for block in text_blocks for text in block['texts']]
        return texts
    
    def ocr_verbose(self, img: np.ndarray):
        """
        测试用，返回mask和TextBlock列表
        """
        img_encoded = cv2.imencode('.jpg', img)[1]
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        payload = {
            "token": self.token,
            "mask": True,
            "refine": True,
            "filtrate": True,
            "disable_skip_area": True,
            "detect_scale": 3,
            "merge_threshold": 0.5,
            "low_accuracy_mode": False,
            "image": img_base64
        }
        response = requests.post(self.url, json=payload)
        response_data = response.json()['Data']

        blk_list = []
        for block in response_data.get('text_block', []):
            xyxy = [int(min(coord[0] for coord in block['block_coordinate'].values())),
                    int(min(coord[1] for coord in block['block_coordinate'].values())),
                    int(max(coord[0] for coord in block['block_coordinate'].values())),
                    int(max(coord[1] for coord in block['block_coordinate'].values()))]
            lines = [np.array([[coord[pos][0], coord[pos][1]] for pos in ['upper_left', 'upper_right', 'lower_right', 'lower_left']], dtype=np.float32) for coord in block['coordinate']]
            texts = block.get('texts', '')
            blk = TextBlock(
                xyxy=xyxy,
                lines=lines,
                language=block.get('language', 'unknown'),
                vertical=block.get('is_vertical', False),
                font_size=block.get('text_size', 0),
                distance=np.array([0, 0], dtype=np.float32),
                angle=0,
                vec=np.array([0, 0], dtype=np.float32),
                norm=0,
                merged=False,
                text=texts,
                fg_colors=np.array(block.get('foreground_color', [0, 0, 0]), dtype=np.float32),
                bg_colors=np.array(block.get('background_color', [0, 0, 0]), dtype=np.float32)
            )
            # print(blk.to_dict())
            blk_list.append(blk)
        
        mask = self._decode_base64_mask(response_data['mask'])
        return mask, blk_list

    @staticmethod
    def _decode_base64_mask(base64_str: str) -> np.ndarray:
        img_data = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        mask = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("Error decoding the mask.")
            return None
        return mask