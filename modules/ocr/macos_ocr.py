# https://github.com/straussmaximilian/ocrmac/blob/main/ocrmac/ocrmac.py
# https://gist.github.com/RhetTbull/1c34fc07c95733642cffcd1ac587fc4c
# https://github.com/RhetTbull/textinator/blob/main/src/macvision.py

import Vision
import objc
import platform
from typing import Tuple
import numpy as np

# Vision.VNRequestTextRecognitionLevelAccurate  0
# Vision.VNRequestTextRecognitionLevelFast      1
# Vision.VNRecognizeTextRequestRevision1        1
# Vision.VNRecognizeTextRequestRevision2        2
# Vision.VNRecognizeTextRequestRevision3        3

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

def get_supported_languages(recognition_level=0, revision=get_revision_level()) -> Tuple[Tuple[str], Tuple[str]]:
    """Get supported languages for text detection from Vision framework.

    Returns: Tuple of ((language code), (error))
    """        
    return Vision.VNRecognizeTextRequest.supportedRecognitionLanguagesForTextRecognitionLevel_revision_error_(
        recognition_level, revision, None
        )

def text_from_image(image: np.ndarray, recognition_level="accurate", language_preference=None):
    recognition_level = recognition_level.lower()
    if language_preference == 'Auto':
        language_preference = None
    image = image.tobytes()

    with objc.autorelease_pool():
        req = Vision.VNRecognizeTextRequest.alloc().init()

        if recognition_level == "fast":
            req.setRecognitionLevel_(1)
        else:
            req.setRecognitionLevel_(0)

        if language_preference is not None:
            req.setRecognitionLanguages_(language_preference)

        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
            image, None
        )

        success = handler.performRequests_error_([req], None)
        res = []
        if success:
            for result in req.results():
                bbox = result.boundingBox()
                w, h = bbox.size.width, bbox.size.height
                x, y = bbox.origin.x, bbox.origin.y

                res.append((result.text(), result.confidence(), [x, y, w, h]))

        req.dealloc()
        handler.dealloc()

        return res


class AppleOCR:
    def __init__(self):
        pass

    def __call__(self, img) -> str:
        pass