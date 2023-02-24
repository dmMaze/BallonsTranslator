from .ocr import OCR, OCRBase, OCRMIT32px, OCRMIT48pxCTC, MangaOCR
from .textdetector import TEXTDETECTORS, TextDetectorBase, ComicTextDetector
from .translators import TRANSLATORS, TranslatorBase, SugoiTranslator
from .inpaint import INPAINTERS, InpainterBase, PatchmatchInpainter, AOTInpainter, LamaInpainterMPE
from .moduleparamparser import DEFAULT_DEVICE

VALID_TEXTDETECTORS = list(TEXTDETECTORS.module_dict.keys())
VALID_TRANSLATORS = list(TRANSLATORS.module_dict.keys())
VALID_INPAINTERS = list(INPAINTERS.module_dict.keys())
VALID_OCR = list(OCR.module_dict.keys())