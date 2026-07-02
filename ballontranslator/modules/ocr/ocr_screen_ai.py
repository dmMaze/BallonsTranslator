import os
import sys
import platform
import ctypes
import threading
from pathlib import Path
from typing import Union, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .base import register_OCR, OCRBase, TextBlock, LOGGER
from ballontranslator.utils.shared import PROGRAM_PATH
from ballontranslator.utils import shared

# Import generated protobufs from module utils
from ballontranslator.modules.ocr.utils.screen_ai_protos.chrome_screen_ai_pb2 import VisualAnnotation as PBVisualAnnotation
from ballontranslator.modules.ocr.utils.screen_ai_protos.view_hierarchy_pb2 import ViewHierarchy as PBViewHierarchy

def get_binary_name() -> str:
    """Return the name of the dynamic library on the current OS."""
    if sys.platform == "win32":
        return "chrome_screen_ai.dll"
    return "libchromescreenai.so"

# === Skia ctypes Mapping structures ===
class SkColorInfo(ctypes.Structure):
    _fields_ = [
        ('fColorSpace', ctypes.c_void_p),
        ('fColorType', ctypes.c_int32),
        ('fAlphaType', ctypes.c_int32)
    ]

class SkISize(ctypes.Structure):
    _fields_ = [
        ('fWidth', ctypes.c_int32),
        ('fHeight', ctypes.c_int32)
    ]

class SkImageInfo(ctypes.Structure):
    _fields_ = [
        ('fColorInfo', SkColorInfo),
        ('fDimensions', SkISize)
    ]

class SkPixmap(ctypes.Structure):
    _fields_ = [
        ('fPixels', ctypes.c_void_p),
        ('fRowBytes', ctypes.c_size_t),
        ('fInfo', SkImageInfo)
    ]

class SkBitmap(ctypes.Structure):
    _fields_ = [
        ('fPixelRef', ctypes.c_void_p),
        ('fPixmap', SkPixmap),
        ('fFlags', ctypes.c_uint32)
    ]


# === Python Dataclasses for API outputs ===
@dataclass
class Rect:
    x: int
    y: int
    width: int
    height: int
    angle: float

@dataclass
class SymbolBox:
    text: str
    bounding_box: Rect
    confidence: float

@dataclass
class WordBox:
    text: str
    bounding_box: Rect
    language: str
    confidence: float
    symbols: List[SymbolBox]

@dataclass
class LineBox:
    text: str
    bounding_box: Rect
    language: str
    confidence: float
    words: List[WordBox]
    paragraph_id: int
    block_id: int

@dataclass
class OCRResult:
    text: str
    lines: List[LineBox]
    raw_proto: Optional[PBVisualAnnotation] = None


class SilenceStderr:
    """Context manager to silence standard error at the OS descriptor level.
    
    This prevents compiled C/C++ libraries from printing verbose log/debug info to the console.
    """
    def __enter__(self):
        if not hasattr(sys.stderr, 'fileno'):
            return self
        try:
            self.stderr_fd = sys.stderr.fileno()
            self.saved_stderr_fd = os.dup(self.stderr_fd)
            self.devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self.devnull, self.stderr_fd)
            self.redirected = True
        except Exception:
            self.redirected = False
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'redirected') and self.redirected:
            try:
                os.dup2(self.saved_stderr_fd, self.stderr_fd)
                os.close(self.saved_stderr_fd)
                os.close(self.devnull)
            except Exception:
                pass


# === Core Wrapper Engine class ===
class ScreenAIEngine:
    """Python bindings wrapper for Google ScreenAI dynamic library."""
    
    def __init__(self, model_dir: Union[str, Path]):
        self.model_dir = Path(model_dir)
        self.resource_dir = self.model_dir / "resources"
        
        lib_name = get_binary_name()
        self.lib_path = self.resource_dir / lib_name
        
        if not self.lib_path.exists():
            raise FileNotFoundError(f"ScreenAI library not found at {self.lib_path}")
            
        self._lock = threading.Lock()
        self._ocr_initialized = False
        self._mce_initialized = False
        self._max_image_dimension = 2048
        
        mode = os.RTLD_LAZY if hasattr(os, 'RTLD_LAZY') else ctypes.DEFAULT_MODE
        self._lib = ctypes.CDLL(str(self.lib_path), mode=mode)
            
        self._setup_bindings()
        self._setup_callbacks()
        
    def _setup_bindings(self):
        # General functions
        self._lib.GetLibraryVersion.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
        self._lib.GetLibraryVersion.restype = None
        
        self._lib.EnableDebugMode.argtypes = []
        self._lib.EnableDebugMode.restype = None
        
        self._lib.SetFileContentFunctions.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._lib.SetFileContentFunctions.restype = None
        
        self._lib.FreeLibraryAllocatedCharArray.argtypes = [ctypes.c_void_p]
        self._lib.FreeLibraryAllocatedCharArray.restype = None
        
        self._lib.FreeLibraryAllocatedInt32Array.argtypes = [ctypes.c_void_p]
        self._lib.FreeLibraryAllocatedInt32Array.restype = None
        
        # OCR functions
        self._lib.InitOCRUsingCallback.argtypes = []
        self._lib.InitOCRUsingCallback.restype = ctypes.c_bool
        
        self._lib.SetOCRLightMode.argtypes = [ctypes.c_bool]
        self._lib.SetOCRLightMode.restype = None
        
        self._lib.GetMaxImageDimension.argtypes = []
        self._lib.GetMaxImageDimension.restype = ctypes.c_uint32
        
        self._lib.PerformOCR.argtypes = [ctypes.POINTER(SkBitmap), ctypes.POINTER(ctypes.c_uint32)]
        self._lib.PerformOCR.restype = ctypes.c_void_p
        
        # Main Content Extraction functions
        self._lib.InitMainContentExtractionUsingCallback.argtypes = []
        self._lib.InitMainContentExtractionUsingCallback.restype = ctypes.c_bool
        
        self._lib.ExtractMainContent.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
        self._lib.ExtractMainContent.restype = ctypes.c_void_p
        
        # Uninitialization functions (if available in DLL)
        if hasattr(self._lib, "UninitializeOCR"):
            self._lib.UninitializeOCR.argtypes = []
            self._lib.UninitializeOCR.restype = None
        if hasattr(self._lib, "UninitializeMainContentExtraction"):
            self._lib.UninitializeMainContentExtraction.argtypes = []
            self._lib.UninitializeMainContentExtraction.restype = None

    def _setup_callbacks(self):
        @ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_char_p)
        def cb_size(p):
            filename_str = p.decode('utf-8')
            path = self.resource_dir / filename_str
            return os.path.getsize(path) if path.exists() else 0

        @ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p)
        def cb_read(p, size, ptr):
            filename_str = p.decode('utf-8')
            path = self.resource_dir / filename_str
            if path.exists():
                with open(path, 'rb') as f:
                    ctypes.memmove(ptr, f.read(size), size)

        self._cb_size = cb_size
        self._cb_read = cb_read
        self._lib.SetFileContentFunctions(self._cb_size, self._cb_read)

    def init_ocr(self, light_mode: bool = False) -> bool:
        with self._lock:
            if self._ocr_initialized:
                self._lib.SetOCRLightMode(light_mode)
                return True
                
            with SilenceStderr():
                success = self._lib.InitOCRUsingCallback()
            if success:
                self._lib.SetOCRLightMode(light_mode)
                self._max_image_dimension = self._lib.GetMaxImageDimension()
                self._ocr_initialized = True
            return success

    def perform_ocr(self, image: Image.Image, light_mode: bool = False) -> Optional[OCRResult]:
        if not self._ocr_initialized:
            if not self.init_ocr(light_mode):
                raise RuntimeError("OCR subsystem could not be initialized.")
                
        self._lib.SetOCRLightMode(light_mode)
        
        w, h = image.size
        if max(w, h) > self._max_image_dimension:
            scale = self._max_image_dimension / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = image.size
            
        rgba_img = image.convert("RGBA")
        raw_bytes = rgba_img.tobytes()
        
        bmp = SkBitmap()
        bmp.fPixmap.fPixels = ctypes.cast(ctypes.c_char_p(raw_bytes), ctypes.c_void_p)
        bmp.fPixmap.fRowBytes = w * 4
        bmp.fPixmap.fInfo.fColorInfo.fColorType = 4  # kRGBA_8888
        bmp.fPixmap.fInfo.fColorInfo.fAlphaType = 1   # kPremul
        bmp.fPixmap.fInfo.fDimensions.fWidth = w
        bmp.fPixmap.fInfo.fDimensions.fHeight = h
        
        out_len = ctypes.c_uint32(0)
        with self._lock:
            result_ptr = self._lib.PerformOCR(ctypes.byref(bmp), ctypes.byref(out_len))
            
        if not result_ptr:
            return None
            
        try:
            proto_bytes = ctypes.string_at(result_ptr, out_len.value)
            
            pb_ann = PBVisualAnnotation()
            pb_ann.ParseFromString(proto_bytes)
            
            lines_list = []
            all_text_lines = []
            
            for line_pb in pb_ann.lines:
                words_list = []
                for word_pb in line_pb.words:
                    symbols_list = []
                    for sym_pb in word_pb.symbols:
                        rect_pb = sym_pb.bounding_box
                        symbols_list.append(SymbolBox(
                            text=sym_pb.utf8_string,
                            bounding_box=Rect(rect_pb.x, rect_pb.y, rect_pb.width, rect_pb.height, rect_pb.angle),
                            confidence=sym_pb.confidence
                        ))
                    
                    rect_w = word_pb.bounding_box
                    words_list.append(WordBox(
                        text=word_pb.utf8_string,
                        bounding_box=Rect(rect_w.x, rect_w.y, rect_w.width, rect_w.height, rect_w.angle),
                        language=word_pb.language,
                        confidence=word_pb.confidence,
                        symbols=symbols_list
                    ))
                
                rect_l = line_pb.bounding_box
                lines_list.append(LineBox(
                    text=line_pb.utf8_string,
                    bounding_box=Rect(rect_l.x, rect_l.y, rect_l.width, rect_l.height, rect_l.angle),
                    language=line_pb.language,
                    confidence=line_pb.confidence,
                    words=words_list,
                    paragraph_id=line_pb.paragraph_id,
                    block_id=line_pb.block_id
                ))
                if line_pb.utf8_string.strip():
                    all_text_lines.append(line_pb.utf8_string.strip())
                    
            full_text = "\n".join(all_text_lines)
            return OCRResult(text=full_text, lines=lines_list, raw_proto=pb_ann)
        finally:
            self._lib.FreeLibraryAllocatedCharArray(result_ptr)

    def close(self):
        """Cleanly uninitialize and unload models to free memory."""
        with self._lock:
            if self._ocr_initialized and hasattr(self._lib, "UninitializeOCR"):
                self._lib.UninitializeOCR()
                self._ocr_initialized = False
            if self._mce_initialized and hasattr(self._lib, "UninitializeMainContentExtraction"):
                self._lib.UninitializeMainContentExtraction()
                self._mce_initialized = False

    def __del__(self):
        try:
            self.close()
        except:
            pass


@register_OCR('google_screen_ai_ocr')
class OCRScreenAI(OCRBase):
    """Google ScreenAI OCR engine module.

    >>> engine = OCRScreenAI()
    >>> engine.params['light_mode']['value']
    False
    """
    _line_only = False
    _load_model_keys = {'model'}

    dependencies = ['protobuf>=4.0.0', 'pillow>=9.0.0']

    # Platform-specific download files registered statically for AST scanner
    if sys.platform == 'win32':
        download_file_list = [{
            'url': 'https://chrome-infra-packages.appspot.com/dl/chromium/third_party/screen-ai/windows-amd64/+/latest',
            'files': ['resources'],
            'save_files': ['data/models/screen_ai/resources'],
            'archived_files': 'screen-ai-windows.zip',
        }]
    elif sys.platform == 'darwin':
        # shared.ON_APPLE_SILICON handles both native arm64 and Rosetta-translated Python.
        # SafeEval in lazy_registry.py has a matching handler so this condition is
        # also evaluated correctly during AST-based lazy module spec building.
        if shared.ON_APPLE_SILICON:
            download_file_list = [{
                'url': 'https://chrome-infra-packages.appspot.com/dl/chromium/third_party/screen-ai/mac-arm64/+/latest',
                'files': ['resources'],
                'save_files': ['data/models/screen_ai/resources'],
                'archived_files': 'screen-ai-mac-arm64.zip',
            }]
        else:
            download_file_list = [{
                'url': 'https://chrome-infra-packages.appspot.com/dl/chromium/third_party/screen-ai/mac-amd64/+/latest',
                'files': ['resources'],
                'save_files': ['data/models/screen_ai/resources'],
                'archived_files': 'screen-ai-mac-amd64.zip',
            }]
    else:
        download_file_list = [{
            'url': 'https://chrome-infra-packages.appspot.com/dl/chromium/third_party/screen-ai/linux/+/latest',
            'files': ['resources'],
            'save_files': ['data/models/screen_ai/resources'],
            'archived_files': 'screen-ai-linux.zip',
        }]

    params = {
        'light_mode': {
            'type': 'checkbox',
            'value': False,
            'display_name': 'Light Mode',
            'description': 'Use the faster, lighter ScreenAI OCR model'
        },
        'newline_handling': {
            'type': 'selector',
            'options': [
                'preserve',
                'remove'
            ],
            'value': 'preserve',
            'display_name': 'Newline Handling',
            'description': 'Choose how to handle newline characters in OCR results'
        },
        'description': 'Google ScreenAI OCR engine. Automatically downloads files on module load.'
    }

    @property
    def newline_handling(self):
        return self.get_param_value('newline_handling')

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model = None

    def _load_model(self):
        default_dir = os.path.join(PROGRAM_PATH, 'data', 'models', 'screen_ai')
        
        # Verify dynamic library existence
        lib_path = os.path.join(default_dir, 'resources', get_binary_name())
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"ScreenAI components not found at {default_dir}. "
                "Please download the models using module installer."
            )

        LOGGER.debug(f"Initializing ScreenAI from {default_dir}...")
        self.model = ScreenAIEngine(model_dir=default_dir)

    def ocr_img(self, img: np.ndarray, **kwargs) -> str:
        if self.model is None:
            return ""
            
        pil_img = Image.fromarray(img)
        light_mode = self.get_param_value('light_mode')
        res = self.model.perform_ocr(pil_img, light_mode=light_mode)
        if res is None:
            return ""
        
        full_text = res.text
        if self.newline_handling == 'remove':
            full_text = full_text.replace('\n', ' ')
            
        return full_text
