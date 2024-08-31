import numpy as np
import json
import cv2
import requests
import base64
from typing import List

from .base import register_OCR, OCRBase, TextBlock
from utils import create_error_dialog, create_info_dialog


@register_OCR('stariver_ocr')
class OCRStariver(OCRBase):
    params = {
        'User': "填入你的用户名",
        'Password': "填入你的密码。请注意，密码会明文保存，请勿在公共电脑上使用",
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
        'update_token_btn': {
            'type': 'pushbtn',
            'value': '',
            'description': '删除旧 Token 并重新申请',
            'display_name': '更新 Token'
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
        self.url = 'https://dl.ap-qz.starivercs.cn/v2/manga_trans/advanced/manga_ocr'
        self.debug = False
        self.token = ''
        self.token_obtained = False
        # 初始化时设置用户名和密码为空
        self.register_username = None
        self.register_password = None


    def get_token(self):
        response = requests.post('https://capiv1.ap-sh.starivercs.cn/OCR/Admin/Login', json={
            "User": self.User,
            "Password": self.Password
        }).json()
        if response.get('Status', -1) != "Success":
            error_msg = f'stariver ocr 登录失败，错误信息：{response.get("ErrorMsg", "")}'
            raise   Exception(error_msg)
        token = response.get('Token', '')
        if token != '':
            self.logger.info(f'登录成功，token前10位：{token[:10]}')

        return token

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        self.update_token_if_needed() # 在向服务器发送请求前尝试更新 Token
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
        self.update_token_if_needed() # 在向服务器发送请求前尝试更新 Token
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

    def update_token_if_needed(self):
        token_updated = False
        if (self.User != self.register_username or 
            self.Password != self.register_password):
            if self.token_obtained == False:
                if "填入你的用户名" not in self.User and "填入你的密码。请注意，密码会明文保存，请勿在公共电脑上使用" not in self.Password:
                    if len(self.Password) > 7 and len(self.User) >= 1:
                        new_token = self.get_token()
                        if new_token:  # 确保新获取到有效token再更新信息
                            self.token = new_token
                            self.register_username = self.User
                            self.register_password = self.Password
                            self.token_obtained = True
                            self.logger.info("Token updated due to credential change.")
                            token_updated = True
        return token_updated

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if param_key == 'update_token_btn':
            self.token_obtained = False  # 强制刷新token时，将标志位设置为False
            self.token = ''  # 强制刷新token时，将token置空
            self.register_username = None  # 强制刷新token时，将用户名置空
            self.register_password = None  # 强制刷新token时，将密码置空
            try:
                if self.update_token_if_needed():
                    create_info_dialog('Token 更新成功')
            except Exception as e:
                create_error_dialog(e, 'Token 更新失败', 'TokenUpdateFailed')