from .ocr import OCR, OCRBase, OCRMIT32px, OCRMIT48pxCTC
from .textdetector import TEXTDETECTORS, TextDetectorBase, ComicTextDetector
from .translators import TRANSLATORS, TranslatorBase
from .inpaint import INPAINTERS, InpainterBase, PatchmatchInpainter, AOTInpainter
from .moduleparamparser import DEFAULT_DEVICE

VALID_TEXTDETECTORS = list(TEXTDETECTORS.module_dict.keys())
VALID_TRANSLATORS = list(TRANSLATORS.module_dict.keys())
VALID_INPAINTERS = list(INPAINTERS.module_dict.keys())
VALID_OCR = OCR.module_dict.keys()