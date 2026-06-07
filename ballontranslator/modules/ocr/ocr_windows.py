# https://learn.microsoft.com/en-us/windows/powertoys/text-extractor#how-to-query-for-ocr-language-packs
import platform

if platform.system() == 'Windows' and platform.version() >= '10.0.10240.0':

    try:
        from winsdk.windows.media.ocr import OcrEngine
        from winsdk.windows.globalization import Language
        from winsdk.windows.storage.streams import DataWriter
        from winsdk.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat
        import numpy as np
        import cv2, asyncio
        from typing import List

        from .base import register_OCR, OCRBase, LOGGER, TextBlock

        def get_supported_language_packs():
            return list(OcrEngine.available_recognizer_languages)

        def ocr(byte, width, height, lang='en'):
            writer = DataWriter()
            writer.write_bytes(byte)
            sb = SoftwareBitmap.create_copy_from_buffer(writer.detach_buffer(), BitmapPixelFormat.RGBA8, width, height)
            return OcrEngine.try_create_from_language(Language(lang)).recognize_async(sb)

        async def coroutine(awaitable):
            return await awaitable 

        winocr_available_recognizer_languages = get_supported_language_packs()

        if len(winocr_available_recognizer_languages) > 0:
            class WindowsOCR:
                lang = winocr_available_recognizer_languages[0].language_tag
                
                def __call__(self, img: np.ndarray) -> str:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
                    w, h = img.shape[1], img.shape[0]
                    return asyncio.run(coroutine(ocr(img.tobytes(), w, h, self.lang))).text

            languages_display_name = [lang.display_name for lang in winocr_available_recognizer_languages]
            languages_tag = [lang.language_tag for lang in winocr_available_recognizer_languages]
            @register_OCR('windows_ocr')
            class OCRWindows(OCRBase):
                params = {
                    'language': {
                        'type':'selector',
                        'options': languages_display_name,
                        'value': languages_display_name[0],
                    }
                }
                language = languages_display_name[0]

                def __init__(self, **params) -> None:
                    super().__init__(**params)
                    self.engine = WindowsOCR()
                    self.engine.lang = self.get_engine_lang()

                def get_engine_lang(self) -> str:
                    language = self.params['language']['value'] 
                    tag_name = languages_tag[languages_display_name.index(language)]
                    return tag_name

                def ocr_img(self, img: np.ndarray) -> str:
                    self.engine(img)

                def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
                    im_h, im_w = img.shape[:2]
                    for blk in blk_list:
                        x1, y1, x2, y2 = blk.xyxy
                        if y2 < im_h and x2 < im_w and \
                            x1 > 0 and y1 > 0 and x1 < x2 and y1 < y2: 
                            blk.text = self.engine(img[y1:y2, x1:x2])
                        else:
                            self.logger.warning('invalid textbbox to target img')
                            blk.text = ['']
                
                def updateParam(self, param_key: str, param_content):
                    super().updateParam(param_key, param_content)
                    self.engine.lang = self.get_engine_lang()

        else:
            LOGGER.warning(f'No supported language packs found for windows, Windows OCR will be unavailable.')
    except Exception as e:
        LOGGER.error(f'Failed to initialize windows OCR:')
        LOGGER.error(e)

