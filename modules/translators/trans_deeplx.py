"""
Modified From PyDeepLX

Author: Vincent Young
Date: 2023-04-27 00:44:01
LastEditors: Vincent Young
LastEditTime: 2023-05-21 03:58:18
FilePath: /PyDeepLX/PyDeepLX/PyDeepLX.py
Telegram: https://t.me/missuo

Copyright © 2023 by Vincent, All Rights Reserved. 

MIT License

Copyright (c) 2023 OwO Network Limited

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from .base import *
# import signal

# class TimeoutException(Exception):
#     pass

# def timeout_handler(signum, frame):
#     raise TimeoutException("Timed out!")



import random
import time
import json
import httpx
from langdetect import detect

from utils.logger import logger as LOGGER


deeplAPI = "https://www2.deepl.com/jsonrpc"
headers = {
    "Content-Type": "application/json",
    "Accept": "*/*",
    "x-app-os-name": "iOS",
    "x-app-os-version": "16.3.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "x-app-device": "iPhone13,2",
    "User-Agent": "DeepL-iOS/2.9.1 iOS 16.3.0 (iPhone13,2)",
    "x-app-build": "510265",
    "x-app-version": "2.9.1",
    "Connection": "keep-alive",
}


class TooManyRequestsException(Exception):
    "Raised when there is a 429 error"

    def __str__(self):
        return "Error: Too many requests, your IP has been blocked by DeepL temporarily, please don't request it frequently in a short time."


def detectLang(translateText) -> str:
    language = detect(translateText)
    return language.upper()


def getICount(translateText) -> int:
    return translateText.count("i")


def getRandomNumber() -> int:
    random.seed(time.time())
    num = random.randint(8300000, 8399998)
    return num * 1000


def getTimestamp(iCount: int) -> int:
    ts = int(time.time() * 1000)

    if iCount == 0:
        return ts

    iCount += 1
    return ts - ts % iCount + iCount


def translate(
    text,
    sourceLang=None,
    targetLang=None,
    numberAlternative=0,
    printResult=False,
    proxies=None,
):
    iCount = getICount(text)
    id = getRandomNumber()

    if sourceLang is None:
        sourceLang = detectLang(text)
    if targetLang is None:
        targetLang = "EN"

    numberAlternative = max(min(3, numberAlternative), 0)

    postData = {
        "jsonrpc": "2.0",
        "method": "LMT_handle_texts",
        "id": id,
        "params": {
            "texts": [{"text": text, "requestAlternatives": numberAlternative}],
            "splitting": "newlines",
            "lang": {
                "source_lang_user_selected": sourceLang,
                "target_lang": targetLang,
            },
            "timestamp": getTimestamp(iCount),
            "commonJobParams": {
                "wasSpoken": False,
                "transcribe_as": "",
            },
        },
    }
    postDataStr = json.dumps(postData, ensure_ascii=False)

    if (id + 5) % 29 == 0 or (id + 3) % 13 == 0:
        postDataStr = postDataStr.replace('"method":"', '"method" : "', -1)
    else:
        postDataStr = postDataStr.replace('"method":"', '"method": "', -1)

    # Add proxy (e.g. proxies='socks5://127.0.0.1:7890')
    with httpx.Client(proxies=proxies) as client:
        resp = client.post(url=deeplAPI, data=postDataStr, headers=headers)
        respStatusCode = resp.status_code

        if respStatusCode == 429:
            raise TooManyRequestsException
            return 

        if respStatusCode != 200:
            print("Error", respStatusCode)
            return

        respText = resp.text
        respJson = json.loads(respText)

        if numberAlternative <= 1:
            targetText = respJson["result"]["texts"][0]["text"]
            if printResult:
                print(targetText)
            return targetText

        targetTextArray = []
        for item in respJson["result"]["texts"][0]["alternatives"]:
            targetTextArray.append(item["text"])
            if printResult:
                print(item["text"])

        return targetTextArray


@register_translator('DeepL Free')
class DeepLX(BaseTranslator):
    cht_require_convert = True
    params: Dict = {
        'delay': 0.0,
    }
    concate_text = True
    
    def _setup_translator(self):
        self.lang_map['简体中文'] = 'zh'
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'en'
        self.lang_map['Français'] = 'fr'
        self.lang_map['Deutsch'] = 'de'
        self.lang_map['Italiano'] = 'it'
        self.lang_map['Português'] = 'pt'
        self.lang_map['Brazilian Portuguese'] = 'pt-br'
        self.lang_map['русский язык'] = 'ru'
        self.lang_map['Español'] = 'es'
        self.lang_map['български език'] = 'bg'
        self.lang_map['Český Jazyk'] = 'cs'
        self.lang_map['Dansk'] = 'da'
        self.lang_map['Ελληνικά'] = 'el'
        self.lang_map['Eesti'] = 'et'
        self.lang_map['Suomi'] = 'fi'
        self.lang_map['Magyar'] = 'hu'
        self.lang_map['Lietuvių'] = 'lt'
        self.lang_map['latviešu'] = 'lv'
        self.lang_map['Nederlands'] = 'nl'
        self.lang_map['Polski'] = 'pl'
        self.lang_map['Română'] = 'ro'
        self.lang_map['Slovenčina'] = 'sk'
        self.lang_map['Slovenščina'] = 'sl'
        self.lang_map['Svenska'] = 'sv'
        self.lang_map['Indonesia'] = 'id'
        self.lang_map['украї́нська мо́ва'] = 'uk'
        self.lang_map['한국어'] = 'ko'
        self.textblk_break = '\n'

    def _translate(self, src_list: List[str]) -> List[str]:
        result = []
        source = self.lang_map[self.lang_source]
        target = self.lang_map[self.lang_target]
    
        for t in src_list:
            # signal.signal(signal.SIGALRM, timeout_handler)
            # signal.alarm(5)  # 5 seconds
            # try:
            #     tl = translate(t,source,target)
            #     signal.alarm(0)
            # except TimeoutException as e:
            #     print(e)
            #     t1 = e
            tl = translate(t,source,target)
            result.append(tl)
            
        return result 
    
