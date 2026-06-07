
import platform
    
if platform.system() == 'Darwin' and platform.mac_ver()[0] >= '10.15':
    
    from typing import Tuple, List
    import numpy as np
    from PIL import Image
    from io import BytesIO
    import Vision
    import objc

    from .base import register_OCR, OCRBase, TextBlock, LOGGER

    def get_revision_level():
        with objc.autorelease_pool():
            ver = platform.mac_ver()[0]
            if ver >= '13':
                revision = Vision.VNRecognizeTextRequestRevision3
            # python might return 10.16 instead of 11.0 for Big Sur and above
            elif ver >= '10.16': # ver[0] >= '11'
                revision = Vision.VNRecognizeTextRequestRevision2
            elif ver >= '10.15':
                revision = Vision.VNRecognizeTextRequestRevision1
            return revision

    def get_supported_languages(recognition_level='accurate', revision=get_revision_level()) -> Tuple[Tuple[str], Tuple[str]]:
        """Get supported languages for text detection from Vision framework.

        Returns: Tuple of ((language code), (error))
        """        

        if recognition_level == 'fast':
            recognition_level = 1
        else:
            recognition_level = 0
        return Vision.VNRecognizeTextRequest.supportedRecognitionLanguagesForTextRecognitionLevel_revision_error_(
            recognition_level, revision, None
            )

    def text_from_image(image: np.ndarray, language_preference=None, recognition_level='accurate'):
        recognition_level = recognition_level.lower()
        if language_preference == 'Auto':
            language_preference = None

        img_buf = BytesIO()
        Image.fromarray(image).save(img_buf, format='PNG')

        with objc.autorelease_pool():
            req = Vision.VNRecognizeTextRequest.alloc().init()

            if recognition_level == 'fast':
                req.setRecognitionLevel_(1)
            else:
                req.setRecognitionLevel_(0)

            if language_preference is not None:
                req.setRecognitionLanguages_(language_preference)

            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
                img_buf.getvalue(), None
            )

            success = handler.performRequests_error_([req], None)
            res = []
            if success:
                for result in req.results():
                    # bbox = result.boundingBox()
                    # w, h = bbox.size.width, bbox.size.height
                    # x, y = bbox.origin.x, bbox.origin.y

                    res.append((result.text(), result.confidence())) #, [x, y, w, h]))

            req.dealloc()
            handler.dealloc()

            return res

    class AppleOCR:
        def __init__(self, lang=[], recog_level='accurate', min_confidence='0.1'):
            self.lang = lang
            self.recog_level = recog_level 
            self.min_confidence = min_confidence

        def __call__(self, img: np.ndarray) -> str:
            result = []
            results = text_from_image(img, self.lang, self.recog_level)
            for res in results:
                if res[1] >= float(self.min_confidence):
                    result.append(res[0])
            return '\n'.join(result)

    macos_ocr_supported_languages = get_supported_languages()

    if len(macos_ocr_supported_languages) > 0:
        @register_OCR('macos_ocr')
        class OCRApple(OCRBase):
            params = {
                'language': {
                    'type':'selector',
                    'options': list(get_supported_languages()[0]),
                    'value': 'en-US',
                },
                # While this does appear 
                # it doesn't update the languages available
                # different recog level, different available langs
                # 'recognition_level': {
                #     'type': 'selector',
                #     'options': [
                #         'accurate',
                #         'fast',
                #     ],
                #     'value': 'accurate',
                # },
                'confidence_level': '0.1',
            }
            language = 'en-US'
            recognition = 'accurate'
            confidence = '0.1'

            def __init__(self, **params) -> None:
                super().__init__(**params)
                self.model = AppleOCR(lang=[self.language])

            def ocr_img(self, img: np.ndarray) -> str:
                return self.model(img)

            def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
                im_h, im_w = img.shape[:2]
                for blk in blk_list:
                    x1, y1, x2, y2 = blk.xyxy
                    if y2 < im_h and x2 < im_w and \
                        x1 > 0 and y1 > 0 and x1 < x2 and y1 < y2: 
                        blk.text = self.model(img[y1:y2, x1:x2])
                    else:
                        self.logger.warning('invalid textbbox to target img')
                        blk.text = ['']

            def updateParam(self, param_key: str, param_content):
                super().updateParam(param_key, param_content)
                self.language = self.params['language']['value']
                self.model.lang = [self.language]

                # self.recognition = self.params['recognition_level']['value']
                # self.model.recog_level = self.recognition
                # self.params['language']['options'] = list(get_supported_languages(self.recognition)[0])

                self.confidence = self.params['confidence_level']
                self.model.min_confidence = self.confidence
    else:
        LOGGER.warning(f'No supported language packs found for MacOS, MacOS OCR will be unavailable.')