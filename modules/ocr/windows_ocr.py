# https://gist.github.com/dantmnf/23f060278585d6243ffd9b0c538beab2
# https://github.com/GitHub30/winocr/blob/main/winocr.py
# https://learn.microsoft.com/en-us/windows/powertoys/text-extractor#how-to-query-for-ocr-language-packs

from winsdk.windows.media.ocr import OcrEngine
from winsdk.windows.globalization import Language
from winsdk.windows.storage.streams import DataWriter
from winsdk.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat

import platform
from distutils.version import LooseVersion

def does_windows_version_support_this():
    pass


def get_supported_language_packs():
    pass

def get_installed_ocr_language_packs():
    pass

def install_ocr_language_pack():
    pass

def uninstall_ocr_language_pack():
    pass

def image2text():
    pass

class WindowsOCR:
    def __init__(self):
        pass
    
    def __call__(self, img) -> str:
        pass