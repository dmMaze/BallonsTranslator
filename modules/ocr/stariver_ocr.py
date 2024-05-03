import cv2
import requests
import json
import base64
import numpy as np


class StariverOCR:

    def __init__(self, token, detect_scale=3, merge_threshold=0.5, refine=True, filtrate=True, disable_skip_area=True, force_expand=False):
        self.token = token
        self.url = 'https://dl.ap-sh.starivercs.cn/v2/manga_trans/advanced/manga_ocr'
        self.debug = False
        self.params = {
            "token": self.token,
            "mask": False,
            "refine": refine,
            "filtrate": filtrate,
            "disable_skip_area": disable_skip_area,
            "detect_scale": detect_scale,
            "merge_threshold": merge_threshold,
            "low_accuracy_mode": True,
            "force_expand": force_expand
        }

    def ocr(self, img: np.ndarray) -> str:
        if not self.params['token'] or self.params['token'] == 'Replace with your token':
            raise ValueError('token 没有设置。')

        img_base64 = base64.b64encode(
            cv2.imencode('.jpg', img)[1]).decode('utf-8')
        self.params["image"] = img_base64

        response = requests.post(self.url, data=json.dumps(self.params))

        if response.status_code != 200:
            print(f'请求失败，状态码：{response.status_code}')
            if response.json().get('Code', -1) != 0:
                print(f'错误信息：{response.json().get("Message", "")}')
                with open('stariver_ocr_error.txt', 'w', encoding='utf-8') as f:
                    f.write(response.text)
            raise ValueError('请求失败。')

        response_data = response.json()['Data']

        if self.debug:
            id = response.json().get('RequestID', '')
            file_name = f"stariver_ocr_response_{id}.json"
            print(f"请求成功，响应数据已保存至{file_name}")
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=4)

        texts_list = ["".join(block.get('texts', '')).strip()
                      for block in response_data.get('text_block', [])]
        texts_str = "".join(texts_list).replace('<skip>', '')
        return texts_str
