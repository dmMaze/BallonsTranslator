# https://learn.microsoft.com/en-us/windows/powertoys/text-extractor#how-to-query-for-ocr-language-packs
from winsdk.windows.media.ocr import OcrEngine
from winsdk.windows.globalization import Language
from winsdk.windows.storage.streams import DataWriter
from winsdk.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat

import numpy as np
import cv2, asyncio

def get_supported_language_packs():
    return list(OcrEngine.available_recognizer_languages)

def ocr(byte, width, height, lang='en'):
    writer = DataWriter()
    writer.write_bytes(byte)
    sb = SoftwareBitmap.create_copy_from_buffer(writer.detach_buffer(), BitmapPixelFormat.RGBA8, width, height)
    return OcrEngine.try_create_from_language(Language(lang)).recognize_async(sb)

async def coroutine(awaitable):
    return await awaitable 

class WindowsOCR:
    lang = get_supported_language_packs()[0].language_tag
    
    def __call__(self, img: np.ndarray) -> str:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        w, h = img.shape[1], img.shape[0]
        return asyncio.run(coroutine(ocr(img.tobytes(), w, h, self.lang))).text