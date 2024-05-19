import numpy as np
import json
import cv2
import requests
import base64
from typing import List

from .base import register_OCR, OCRBase, TextBlock


@register_OCR('stariver_ocr')
class OCRStariver(OCRBase):
    params = {
        'User': "填入你的用户名",
        'Password': "填入你的密码。请注意，密码会明文保存，请勿在公共电脑上使用",
        'force_refresh_token': {
            'type': 'checkbox',
            'value': False
        },
        "refine":{
            'type': 'checkbox',
            'value': True
        },
        "filtrate":{
            'type': 'checkbox',
            'value': True
        },
        "disable_skip_area":{
            'type': 'checkbox',
            'value': True
        },
        "detect_scale": "3",
        "merge_threshold": "2",
        "force_expand":{
            'type': 'checkbox',
            'value': False,
            'description': '是否强制扩展图片像素，会导致识别速度下降'
        },
        "low_accuracy_mode":{
            'type': 'checkbox',
            'value': False,
        },
        'description': '星河云(团子翻译器) OCR API'
    }

    @property
    def User(self):
        return self.params['User']
    
    @property
    def Password(self):
        return self.params['Password']
    
    @property
    def force_refresh_token(self):
        return self.params['force_refresh_token']['value']
    
    @property
    def expand_ratio(self):
        return float(self.params['expand_ratio'])
    
    @property
    def refine(self):
        return  self.params['refine']['value']
     
    @property
    def filtrate(self):
        self.params['filtrate']['value']

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
    def force_expand(self):
        return self.params['force_expand']['value']
    
    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.token = ''
        self.url = "https://dl.ap-sh.starivercs.cn/v2/manga_trans/advanced/manga_ocr"
        self.token_obtained = False  # 添加一个标志位来判断token是否已经获取过
        
        # 在初始化时尝试获取token
        if not self.token_obtained:
            self.update_token_if_needed()
            self.token_obtained = True  # 将标志位设置为True，表示已获取token

    def update_token_if_needed(self):
        if "填入你的用户名" not in self.User and "填入你的密码。请注意，密码会明文保存，请勿在公共电脑上使用" not in self.Password:
            if not self.token_obtained or self.force_refresh_token:  # 检查标志位，只有在第一次运行时获取token
                if len(self.Password) > 7 and len(self.User) >= 1:
                    self.token = self.get_token()
                    if self.token != '':
                        self.token_obtained = True  # 获取成功后，将标志位设置为True
        else:
            self.logger.warning('stariver ocr 用户名或密码为空，无法更新token。')

    def get_token(self):
        response = requests.post('https://capiv1.ap-sh.starivercs.cn/OCR/Admin/Login', json={
            "User": self.User,
            "Password": self.Password
        }).json()
        if response.get('Status', -1) != "Success":
            self.logger.error(f'stariver ocr 登录失败，错误信息：{response.get("ErrorMsg", "")}')
        token = response.get('Token', '')
        if token != '':
            self.logger.info(f'登录成功，token前10位：{token[:10]}')

        return token

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]):
        im_h, im_w = img.shape[:2]
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if y2 < im_h and x2 < im_w and \
                    x1 > 0 and y1 > 0 and x1 < x2 and y1 < y2:
                blk.text = self.ocr(img[y1:y2, x1:x2])
            else:
                self.logger.warning('invalid textbbox to target img')
                blk.text = ['']

    def ocr_img(self, img: np.ndarray) -> str:
        self.logger.debug(f'ocr_img: {img.shape}')
        return self.ocr(img)

    def ocr(self, img: np.ndarray) -> str:
        
        payload = {
            "token": self.token,
            "mask": False,
            "refine": self.refine,
            "filtrate": self.filtrate,
            "disable_skip_area": self.disable_skip_area,
            "detect_scale": self.detect_scale,
            "merge_threshold": self.merge_threshold,
            "low_accuracy_mode": self.params['low_accuracy_mode']['value'],
            "force_expand": self.force_expand
        }

        img_base64 = base64.b64encode(
            cv2.imencode('.jpg', img)[1]).decode('utf-8')
        payload["image"] = img_base64

        response = requests.post(self.url, data=json.dumps(payload))

        if response.status_code != 200:
            print(f'stariver ocr 请求失败，状态码：{response.status_code}')
            if response.json().get('Code', -1) != 0:
                print(f'stariver ocr 错误信息：{response.json().get("Message", "")}')
                with open('stariver_ocr_error.txt', 'w', encoding='utf-8') as f:
                    f.write(response.text)
            raise ValueError('stariver ocr 请求失败。')

        response_data = response.json()['Data']

        if self.debug:
            id = response.json().get('RequestID', '')
            file_name = f"stariver_ocr_response_{id}.json"
            print(f"stariver ocr 请求成功，响应数据已保存至{file_name}")
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=4)

        texts_list = ["".join(block.get('texts', '')).strip()
                      for block in response_data.get('text_block', [])]
        texts_str = "".join(texts_list).replace('<skip>', '')
        return texts_str

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == 'User' or param_key == 'Password':
            if not self.token_obtained or self.force_refresh_token:  # 检查标志位，只有在第一次运行时获取token
                self.update_token_if_needed()
            if param_key == 'force_refresh_token':
                self.token_obtained = False  # 强制刷新token时，将标志位设置为False