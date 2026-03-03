import re
import numpy as np
import time
import cv2
import os
from typing import List
import ctypes
from ctypes import (
    Structure, byref, POINTER, c_int64, c_int32, c_float, c_ubyte, c_char, c_char_p
)
from PIL import Image as PilImage
from contextlib import contextmanager
import logging

from .base import register_OCR, OCRBase, TextBlock


ONE_OCR_PATH = os.path.join("data", "models", "one-ocr")
MODEL_NAME = "oneocr.onemodel"
DLL_NAME = "oneocr.dll"
MODEL_KEY = b'kj)TGtrK>f]b[Piow.gU+nC@s""""""4'
MIN_DIM_SIZE = 56  # Minimum height *and* width for padding

c_int64_p = POINTER(c_int64)
c_float_p = POINTER(c_float)
c_ubyte_p = POINTER(c_ubyte)


class ImageStructure(Structure):
    _fields_ = [("type", c_int32), ("width", c_int32), ("height", c_int32),
                ("_reserved", c_int32), ("step_size", c_int64), ("data_ptr", c_ubyte_p)]


class BoundingBox(Structure):
    _fields_ = [("x1", c_float), ("y1", c_float),
                ("x2", c_float), ("y2", c_float)]


BoundingBox_p = POINTER(BoundingBox)

DLL_FUNCTIONS = [  # Function definitions for the DLL
    ("CreateOcrInitOptions", [c_int64_p], c_int64), ("OcrInitOptionsSetUseModelDelayLoad", [
        c_int64, c_char], c_int64),
    ("CreateOcrPipeline", [c_char_p, c_char_p, c_int64, c_int64_p],
     c_int64), ("CreateOcrProcessOptions", [c_int64_p], c_int64),
    ("OcrProcessOptionsSetMaxRecognitionLineCount", [c_int64, c_int64], c_int64), (
        "RunOcrPipeline", [c_int64, POINTER(ImageStructure), c_int64, c_int64_p], c_int64),
    ("GetImageAngle", [c_int64, c_float_p], c_int64), ("GetOcrLineCount", [
        c_int64, c_int64_p], c_int64),
    ("GetOcrLine", [c_int64, c_int64, c_int64_p], c_int64), ("GetOcrLineContent", [
        c_int64, POINTER(c_char_p)], c_int64),
    ("GetOcrLineBoundingBox", [c_int64, POINTER(
        BoundingBox_p)], c_int64), ("GetOcrLineWordCount", [c_int64, c_int64_p], c_int64),
    ("GetOcrWord", [c_int64, c_int64, c_int64_p], c_int64), ("GetOcrWordContent", [
        c_int64, POINTER(c_char_p)], c_int64),
    ("GetOcrWordBoundingBox", [c_int64, POINTER(
        BoundingBox_p)], c_int64), ("GetOcrWordConfidence", [c_int64, c_float_p], c_int64),
    ("ReleaseOcrResult", [c_int64],
     None), ("ReleaseOcrInitOptions", [c_int64], None),
    ("ReleaseOcrPipeline", [c_int64],
     None), ("ReleaseOcrProcessOptions", [c_int64], None),
]


@contextmanager
def suppress_output():  # Context manager to suppress stdout/stderr, useful for noisy DLL init
    devnull = os.open(os.devnull, os.O_WRONLY)
    original_stdout, original_stderr = os.dup(1), os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(original_stdout, 1)
        os.dup2(original_stderr, 2)
        os.close(original_stdout)
        os.close(original_stderr)
        os.close(devnull)


class OcrEngine:
    def __init__(self, config_dir, logger=None):
        self.ocr_dll = self.init_options = self.pipeline = self.process_options = None
        self.config_dir = config_dir
        self.model_path = os.path.join(self.config_dir, MODEL_NAME)
        self.dll_path = os.path.join(self.config_dir, DLL_NAME)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.empty_result = {"text": "", "text_angle": None, "lines": []}
        try:
            self._load_and_bind_dll()
            self.init_options = self._create_init_options()
            self.pipeline = self._create_pipeline()
            self.process_options = self._create_process_options()
            if self.logger:
                self.logger.debug("OcrEngine initialized")
        except Exception as e:
            raise e  # Error logged internally

    def _load_and_bind_dll(self):
        try:
            if os.name == 'nt':
                k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                if hasattr(k32, "SetDllDirectoryW"):
                    k32.SetDllDirectoryW(str(self.config_dir))
            self.ocr_dll = ctypes.WinDLL(
                str(self.dll_path), use_last_error=True)
            for name, argtypes, restype in DLL_FUNCTIONS:
                try:
                    func = getattr(self.ocr_dll, name)
                    func.argtypes = argtypes
                    func.restype = restype
                except AttributeError as e:
                    raise RuntimeError(f"Missing DLL func: {name}") from e
        except (OSError, RuntimeError, AttributeError) as e:
            code = ctypes.get_last_error() if os.name == "nt" else 0
            msg = f"Failed load/bind DLL ({self.dll_path}) from {self.config_dir}. Code: {code}. Error: {e}"
            if self.logger:
                self.logger.error(msg)
                raise RuntimeError(msg) from e

    def _check_dll_result(self, result_code, error_message):
        if result_code != 0:
            raise RuntimeError(f"{error_message} (Native Code: {result_code})")

    def _create_init_options(self):
        h = c_int64()
        self._check_dll_result(self.ocr_dll.CreateOcrInitOptions(
            byref(h)), "Init options create failed")
        self._check_dll_result(self.ocr_dll.OcrInitOptionsSetUseModelDelayLoad(
            h, 0), "Model load config failed")
        return h

    def _create_pipeline(self):
        mb = ctypes.create_string_buffer(self.model_path.encode("utf-8"))
        kb = ctypes.create_string_buffer(MODEL_KEY)
        h = c_int64()
        with suppress_output():
            self._check_dll_result(self.ocr_dll.CreateOcrPipeline(
                mb, kb, self.init_options, byref(h)), f"Pipeline create failed ({self.model_path})")
        return h

    def _create_process_options(self):
        h = c_int64()
        self._check_dll_result(self.ocr_dll.CreateOcrProcessOptions(
            byref(h)), "Process options create failed")
        self._check_dll_result(self.ocr_dll.OcrProcessOptionsSetMaxRecognitionLineCount(
            h, 1000), "Line count config failed")
        return h

    def recognize_pil(self, image: PilImage.Image):
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        return self._process_image(cols=image.width, rows=image.height, step=image.width*4, data=image.tobytes())

    def _process_image(self, cols, rows, step, data):
        dp = ctypes.cast(data, c_ubyte_p) if not isinstance(
            data, bytes) else (c_ubyte*len(data)).from_buffer_copy(data)
        img_struct = ImageStructure(3, cols, rows, 0, step, dp)
        return self._perform_ocr(img_struct)

    def _perform_ocr(self, image_struct: ImageStructure):
        res_h = c_int64()
        code = self.ocr_dll.RunOcrPipeline(self.pipeline, byref(
            image_struct), self.process_options, byref(res_h))
        if code != 0:
            # Let the caller (OCROneAPI) handle specific error code interpretation
            # This will raise RuntimeError
            self._check_dll_result(code, "RunOcrPipeline failed")
            # Alternative: log here and return empty, but raising is cleaner for caller
            # if self.logger: self.logger.warning(f"RunOcrPipeline failed: {code}")
            # return self.empty_result
        parsed = self._parse_ocr_results(res_h.value)
        self.ocr_dll.ReleaseOcrResult(res_h)
        return parsed

    def _parse_ocr_results(self, res_h: int):
        lc = c_int64()
        if self.ocr_dll.GetOcrLineCount(res_h, byref(lc)) != 0:
            if self.logger:
                self.logger.warning("Failed get line count")
                return self.empty_result
        lines = self._get_lines(res_h, lc.value)
        angle = self._get_text_angle(res_h)
        return {"text": None, "text_angle": angle, "lines": lines}

    def _get_text_angle(self, res_h: int): a = c_float(
    ); return a.value if self.ocr_dll.GetImageAngle(res_h, byref(a)) == 0 else None

    def _get_lines(self, res_h: int, n: int): return [
        self._process_line(res_h, i) for i in range(n)]

    def _process_line(self, res_h: int, idx: int):
        lh = c_int64()
        if self.ocr_dll.GetOcrLine(res_h, idx, byref(lh)) != 0:
            if self.logger:
                self.logger.warning(f"Failed get line handle {idx}")
                return {"text": None, "bounding_rect": None, "words": []}
        lhv = lh.value
        text = self._get_text(lhv, self.ocr_dll.GetOcrLineContent)
        bbox = self._get_bounding_box(lhv, self.ocr_dll.GetOcrLineBoundingBox)
        words = self._get_words(lhv)
        return {"text": text, "bounding_rect": bbox, "words": words}

    def _get_words(self, line_h: int):
        wc = c_int64()
        if self.ocr_dll.GetOcrLineWordCount(line_h, byref(wc)) != 0:
            return []
        return [self._process_word(line_h, i) for i in range(wc.value)]

    def _process_word(self, line_h: int, idx: int):
        wh = c_int64()
        if self.ocr_dll.GetOcrWord(line_h, idx, byref(wh)) != 0:
            if self.logger:
                self.logger.warning(f"Failed get word handle {idx}")
                return {"text": None, "bounding_rect": None, "confidence": None}
        whv = wh.value
        text = self._get_text(whv, self.ocr_dll.GetOcrWordContent)
        bbox = self._get_bounding_box(whv, self.ocr_dll.GetOcrWordBoundingBox)
        conf = self._get_word_confidence(whv)
        return {"text": text, "bounding_rect": bbox, "confidence": conf}

    def _get_text(self, handle: int, func):
        content = c_char_p()
        if func(handle, byref(content)) == 0 and content.value:
            try:
                return content.value.decode("utf-8", errors="ignore")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error decoding text: {e}")
        return None

    def _get_word_confidence(self, word_h: int): c = c_float(
    ); return c.value if self.ocr_dll.GetOcrWordConfidence(word_h, byref(c)) == 0 else None

    def _get_bounding_box(self, handle: int, func):
        bp = BoundingBox_p()
        if func(handle, byref(bp)) == 0 and bp:
            b = bp.contents
            return {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2}
        return None


@register_OCR("one_ocr")
class OCROneAPI(OCRBase):
    params = {
        "expand_small_blocks": {"type": "checkbox", "value": True, "description": f"Expand image width/height if < {MIN_DIM_SIZE}px before recognition by padding (helps with small images)"},
        "newline_handling": {"type": "selector", "options": ["preserve", "remove"], "value": "preserve", "description": "Newline char handling (preserve/remove)"},
        "reverse_line_order": {"type": "checkbox", "value": False, "description": "Reverse line order (for vertical CJK)"},
        "no_uppercase": {"type": "checkbox", "value": False, "description": "Convert text to lowercase (except sentence start)"},
        "description": "Local OCR using OneOCR library (Windows Only)",
    }

    @property
    def expand_small_blocks(self): v = self.get_param_value(
        "expand_small_blocks"); return bool(v) if v is not None else True

    @property
    def newline_handling(self): return self.get_param_value(
        "newline_handling") or "preserve"

    @property
    def no_uppercase(self): return bool(self.get_param_value("no_uppercase"))

    @property
    def reverse_line_order(self): return bool(
        self.get_param_value("reverse_line_order"))

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.engine = None
        self.available = False
        self.config_dir = ONE_OCR_PATH
        if os.name != "nt":
            if self.logger:
                self.logger.warning("OneOCR is Windows-only.")
                return
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            dll_p = os.path.join(self.config_dir, DLL_NAME)
            model_p = os.path.join(self.config_dir, MODEL_NAME)
            if not os.path.exists(dll_p) or not os.path.exists(model_p):
                msg = f"OneOCR init fail: DLL/Model not found in '{self.config_dir}'. See guide."
                (self.logger or logging).warning(msg)
                return
            self.engine = OcrEngine(self.config_dir, self.logger)
            self.available = True
            if self.logger:
                self.logger.info("OneOCR engine ready.")
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Failed to create OcrEngine: {e}", exc_info=self.debug_mode)
            self.engine = None
            self.available = False

    def _ocr_blk_list(self, img: np.ndarray, blk_list: list[TextBlock], *args, **kwargs):
        if not self.available:
            return
        im_h, im_w = img.shape[:2]
        for i, blk in enumerate(blk_list):
            x1, y1, x2, y2 = blk.xyxy
            if 0 <= y1 < y2 <= im_h and 0 <= x1 < x2 <= im_w:
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    blk.text = ""
                    continue
                is_vertical = blk.vertical
                if self.logger:
                    self.logger.debug(f"Block {i+1}: xyxy={blk.xyxy}, src_is_vertical={blk.src_is_vertical}, vertical={blk.vertical}, is_vertical={is_vertical}")
                try:
                    blk.text = self.ocr(crop, apply_postprocessing=True, reverse_lines=is_vertical)
                except Exception as e:  # Log error from main ocr call
                    if self.logger:
                        self.logger.error(
                            f"OCR err block {i+1} {blk.xyxy}: {e}", exc_info=self.debug_mode)
                        blk.text = ""
            else:
                blk.text = ""  # Invalid coords

    def ocr_img(self, img: np.ndarray) -> str: return self.ocr(img,
                                                               apply_postprocessing=True, reverse_lines=self.reverse_line_order) if self.available else ""

    def ocr(self, img: np.ndarray, apply_postprocessing: bool = True, reverse_lines: bool = None) -> str:
        if not self.available or self.engine is None or img is None or img.size == 0:
            return ""
        start_time = time.time()
        original_h, original_w = img.shape[:2]
        padded = False
        if self.debug_mode and self.logger:
            self.logger.debug(f"OCR start shape: {original_h}x{original_w}")

        try:
            img_to_process = img  # Start with original image
            if self.expand_small_blocks and (original_h < MIN_DIM_SIZE or original_w < MIN_DIM_SIZE):
                pad_h_total = max(0, MIN_DIM_SIZE - original_h)
                pad_w_total = max(0, MIN_DIM_SIZE - original_w)
                pad_top = pad_h_total//2
                pad_bottom = pad_h_total - pad_top
                pad_left = pad_w_total//2
                pad_right = pad_w_total - pad_left
                # Determine padding color (white) based on channels
                if len(img.shape) == 2:
                    color = 255  # Grayscale
                elif img.shape[2] == 3:
                    color = (255, 255, 255)  # BGR
                else:
                    color = (255, 255, 255, 255)  # BGRA (or assume if > 3)
                img_to_process = cv2.copyMakeBorder(
                    img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
                padded = True
                if self.debug_mode and self.logger:
                    self.logger.debug(
                        f"Padded img from {original_h}x{original_w} to {img_to_process.shape[:2]} (min={MIN_DIM_SIZE})")

            # Convert potentially padded image to RGB for PIL
            if len(img_to_process.shape) == 2:
                img_rgb = cv2.cvtColor(img_to_process, cv2.COLOR_GRAY2RGB)
            elif img_to_process.shape[2] == 3:
                img_rgb = img_to_process
            elif img_to_process.shape[2] == 4:
                img_rgb = cv2.cvtColor(img_to_process, cv2.COLOR_RGBA2RGB)
            else:
                raise ValueError(
                    f"Unsupported channels: {img_to_process.shape[2]}")

            pil_image = PilImage.fromarray(img_rgb).convert("RGBA")
            # This might raise RuntimeError on failure code
            result_dict = self.engine.recognize_pil(pil_image)
            raw_lines = result_dict.get("lines", [])
            if self.logger:
                self.logger.debug(f"  reverse_lines={reverse_lines}, reverse_line_order={self.reverse_line_order}, total_lines={len(raw_lines)}")
                for li, ln in enumerate(raw_lines):
                    bb = ln.get("bounding_rect")
                    self.logger.debug(f"  line[{li}]: text={repr(ln.get('text',''))[:40]}, bbox={bb}")
            lines = [ln["text"] for ln in raw_lines if ln.get("text")]
            base_reverse = reverse_lines if reverse_lines is not None else False
            should_reverse = base_reverse ^ self.reverse_line_order
            if self.logger:
                self.logger.debug(f"  should_reverse={should_reverse}")
            if should_reverse:
                lines.reverse()
            full_text = "\n".join(lines)
            if apply_postprocessing:
                full_text = self._apply_text_postprocessing(full_text)

            if self.debug_mode and self.logger:
                self.logger.debug(
                    f"OCR done {(time.time()-start_time):.3f}s, padded: {padded}, lines: {len(lines)}")
            return full_text

        except RuntimeError as e:  # Catch errors from _check_dll_result
            err_code_match = re.search(r"\(Native Code: (\d+)\)", str(e))
            err_code = int(err_code_match.group(1)) if err_code_match else None
            log_msg = f"Critical OCR error: {e}"
            # Check if it's code 3 and padding *wasn't* applied to a small image
            if err_code == 3 and not padded and (original_h < MIN_DIM_SIZE or original_w < MIN_DIM_SIZE):
                log_msg += f" (Native Code 3 often means image too small. Try enabling 'expand_small_blocks' in params if disabled, or check if padding to {MIN_DIM_SIZE}px is sufficient)"
            if self.logger:
                self.logger.error(log_msg, exc_info=self.debug_mode)
            return ""
        except Exception as e:  # Catch other errors (PIL conversion, etc.)
            if self.logger:
                self.logger.error(
                    f"Unexpected OCR error: {e}", exc_info=self.debug_mode)
            return ""

    def _apply_text_postprocessing(self, text: str) -> str:
        if not text:
            return ""
        if self.newline_handling == "remove":
            text = text.replace("\n", " ").replace("\r", "")
        elif self.newline_handling == "preserve":
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self._apply_punctuation_and_spacing(text)
        if self.no_uppercase:
            text = self._apply_no_uppercase(text)
        return text

    def _apply_no_uppercase(self, text: str) -> str:
        if not text:
            return ""
        return " ".join([s[0].upper()+s[1:].lower() for s in re.split(r"(?<=[.!?…])\s+", text) if s])

    def _apply_punctuation_and_spacing(self, text: str) -> str:
        if not text:
            return ""
        # Remove space before punct
        text = re.sub(r"\s+([,.!?…;:])", r"\1", text)
        text = re.sub(r"([,.!?…;:])(?=[^\s,.!?…;:])", r"\1 ",
                      text)  # Add space after punct if missing
        text = re.sub(r"\s{2,}", " ", text)  # Collapse multiple spaces
        return text.strip()

    def updateParam(self, key: str, content):
        super().updateParam(key, content)
        if self.debug_mode and self.logger:
            self.logger.debug(f"Param '{key}' updated in OCROneAPI.")
