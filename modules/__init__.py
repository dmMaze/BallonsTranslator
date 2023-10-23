from .ocr import OCR, OCRBase, OCRMIT32px, OCRMIT48pxCTC, MangaOCR
from .textdetector import TEXTDETECTORS, TextDetectorBase, ComicTextDetector
from .translators import TRANSLATORS, BaseTranslator
from .inpaint import INPAINTERS, InpainterBase, PatchmatchInpainter, AOTInpainter, LamaInpainterMPE
from .base import DEFAULT_DEVICE, GPUINTENSIVE_SET

GET_VALID_TEXTDETECTORS = lambda : list(TEXTDETECTORS.module_dict.keys())
GET_VALID_TRANSLATORS = lambda : list(TRANSLATORS.module_dict.keys())
GET_VALID_INPAINTERS = lambda : list(INPAINTERS.module_dict.keys())
GET_VALID_OCR = lambda : list(OCR.module_dict.keys())

# TODO: use manga-image-translator as backend...