import os
import os.path as osp
import sys
import glob
import ctypes
from typing import List
import argparse

import numpy as np
import yaml
import cv2

from .base import DEVICE_SELECTOR, OCRBase, TextBlock, register_OCR
from ballontranslator.utils.textblock import collect_textblock_regions

# ── Paths ────────────────────────────────────────────────────────────────────

MODEL_DIR = "data/models/PP-OCRv6_medium_rec_onnx"


_CUDA_LIBRARY_PATTERNS = (
    "libcudart.so*",
    "libnvrtc.so*",
    "libnvJitLink.so*",
    "libcusparse.so*",
    "libcublas.so*",
    "libcublasLt.so*",
    "libcufft.so*",
    "libcurand.so*",
    "libcudnn.so*",
    "libcudnn_*.so*",
)


def _dedupe_existing_dirs(paths):
    """Return existing directories in input order without duplicates.

    >>> _dedupe_existing_dirs(["", ".", "."])
    ['.']
    """
    dirs = []
    seen = set()
    for path in paths:
        if not path or not osp.isdir(path):
            continue
        norm_path = osp.normpath(path)
        real_path = osp.realpath(norm_path)
        if real_path in seen:
            continue
        seen.add(real_path)
        dirs.append(norm_path)
    return dirs


def _cuda_library_dirs(torch_module=None):
    """Find CUDA runtime library directories bundled with the active env.

    >>> isinstance(_cuda_library_dirs(), list)
    True
    """
    candidates = []

    if torch_module is not None:
        torch_dir = osp.dirname(torch_module.__file__)
        candidates.append(osp.join(torch_dir, "lib"))
        site_packages_dir = osp.dirname(torch_dir)
        candidates.extend(glob.glob(osp.join(site_packages_dir, "nvidia", "*", "lib")))

    for env_key in ("CUDA_HOME", "CUDA_PATH", "CONDA_PREFIX", "VIRTUAL_ENV"):
        prefix = os.environ.get(env_key)
        if prefix:
            candidates.extend((osp.join(prefix, "lib64"), osp.join(prefix, "lib")))

    candidates.extend((osp.join(sys.prefix, "lib64"), osp.join(sys.prefix, "lib")))
    for search_path in sys.path:
        candidates.extend(glob.glob(osp.join(search_path, "nvidia", "*", "lib")))

    return _dedupe_existing_dirs(candidates)


def _dir_has_cuda_runtime(lib_dir: str):
    """Return whether ``lib_dir`` looks like it contains CUDA runtime libs.

    >>> _dir_has_cuda_runtime("")
    False
    """
    if not lib_dir or not osp.isdir(lib_dir):
        return False
    return any(glob.glob(osp.join(lib_dir, pattern)) for pattern in _CUDA_LIBRARY_PATTERNS)


def _ensure_cuda_home_from_env(cuda_dirs):
    """Point ``CUDA_HOME`` at the active Python env when it owns CUDA libs.

    >>> _ensure_cuda_home_from_env([])
    False
    """
    if os.environ.get("CUDA_HOME"):
        return False

    cuda_real_dirs = {osp.realpath(lib_dir) for lib_dir in cuda_dirs}
    for prefix in (os.environ.get("CONDA_PREFIX"), os.environ.get("VIRTUAL_ENV"), sys.prefix):
        if not prefix:
            continue
        lib_dirs = (osp.join(prefix, "lib64"), osp.join(prefix, "lib"))
        if any(
            osp.realpath(lib_dir) in cuda_real_dirs and _dir_has_cuda_runtime(lib_dir)
            for lib_dir in lib_dirs
        ):
            os.environ["CUDA_HOME"] = prefix
            return True

    return False


def _register_windows_dll_directory(lib_dir: str):
    """Register ``lib_dir`` for Windows DLL lookup when the API is available.

    >>> _register_windows_dll_directory("")
    False
    """
    if os.name != "nt" or not lib_dir or not osp.isdir(lib_dir):
        return False

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        return add_dll_directory(lib_dir)

    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    if lib_dir not in path_dirs:
        os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")
    return True


def _preload_linux_cuda_libraries(cuda_dirs):
    """Preload CUDA/cuDNN shared libraries so ONNX Runtime can resolve them.

    >>> _preload_linux_cuda_libraries([])
    []
    """
    if not sys.platform.startswith("linux"):
        return []

    loaded_handles = []
    loaded_paths = set()
    for pattern in _CUDA_LIBRARY_PATTERNS:
        for lib_dir in cuda_dirs:
            for lib_path in sorted(glob.glob(osp.join(lib_dir, pattern))):
                real_path = osp.realpath(lib_path)
                if real_path in loaded_paths:
                    continue
                try:
                    loaded_handles.append(ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL))
                    loaded_paths.add(real_path)
                except OSError:
                    continue

    return loaded_handles


def _write_ppocr_dict_from_yaml(yaml_path: str, dict_path: str):
    """Write OnnxOCR's plain text character dictionary from PaddleOCR YAML.

    >>> _write_ppocr_dict_from_yaml("", "")
    False
    """
    if not yaml_path or not dict_path:
        return False

    with open(yaml_path, encoding='utf8') as f:
        character_dict = yaml.safe_load(f)['PostProcess']['character_dict']

    # Keep the model dictionary byte-for-byte meaningful: ASCII decoding with
    # errors ignored turns characters such as "¢" into YAML null entries.
    characters = ["" if character is None else str(character) for character in character_dict]
    with open(dict_path, 'w', encoding='utf8') as f:
        f.write('\n'.join(characters))
    return True


def _restore_text_recognizer_call(recognizer_cls):
    """Undo stale class-level recognizer patches from earlier module loads.

    >>> _restore_text_recognizer_call(type("Recognizer", (), {}))
    False
    """
    call = getattr(recognizer_cls, "__call__", None)
    restored = False

    while getattr(call, "__name__", None) == "_patched":
        original_call = None
        for cell in getattr(call, "__closure__", None) or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if callable(value) and getattr(value, "__name__", None) in {"__call__", "_patched"}:
                original_call = value
                break
        if original_call is None or original_call is call:
            break
        call = original_call
        restored = True

    if restored:
        recognizer_cls.__call__ = call
    return restored


def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


@register_OCR("ppv6_onnx")
class PaddleOCRv6ONNX(OCRBase):
    params = {
        "device": DEVICE_SELECTOR(not_supported=['privateuseone']),
        "rec_batch_num": {
            "value": 6,
            "description": "Recognition batch size (higher = faster, more VRAM)",
            "display_name": "Recognition Batch Size",
        },
        "description": "PP-OCRv6 ONNX recognition-only — crops text blocks then recognizes via ONNX Runtime",
    }

    dependencies = ["onnxruntime", "torch", "pyyaml"]

    download_file_list = [
        {
            "url": "https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/main/inference.onnx",
            "files": osp.join(MODEL_DIR, "inference.onnx"),
            "sha256_pre_calculated": "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba"
        },
        {
            "url": "https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/main/inference.yml",
            "files": osp.join(MODEL_DIR, "inference.yml")
        }
    ]

    _load_model_keys = {"recognizer"}

    def __init__(self, **params):
        super().__init__(**params)
        self.recognizer = None
        self._dll_loaded = False
        self._dll_dir_handle = None
        self._cuda_lib_handles = []

    def updateParam(self, param_key: str, param_content):
        device = self.get_param_value("device")
        super().updateParam(param_key, param_content)

        if device != param_content and self.recognizer is not None:
            del self.recognizer
            self.recognizer = None

    # ── Model lifecycle ─────────────────────────────────────────────────
    def _ensure_cuda_dll_path(self):
        """Expose PyTorch/env CUDA runtime libraries to ONNX Runtime.

        On Windows, register PyTorch's ``torch/lib`` directory for DLL lookup.
        On Linux, preload CUDA/cuDNN ``.so`` files from torch, the active
        conda/venv prefix, or pip ``nvidia`` packages so the CUDA
        ExecutionProvider can resolve its dynamic links.
        """
        if self._dll_loaded:
            return

        try:
            import torch
        except ImportError:
            return  # torch not installed — CUDA EP won't work; fine

        if os.name == "nt":
            lib_dir = osp.join(osp.dirname(torch.__file__), "lib")
            self._dll_dir_handle = _register_windows_dll_directory(lib_dir)
        elif sys.platform.startswith("linux"):
            cuda_dirs = _cuda_library_dirs(torch)
            _ensure_cuda_home_from_env(cuda_dirs)
            self._cuda_lib_handles = _preload_linux_cuda_libraries(cuda_dirs)

        self._dll_loaded = True

    def _load_model(self):
        """Initialise the OnnxOCR recognizer.

        Called by ``BaseModule.load_model()`` after dependency & model-file
        checks have passed.  Only loads the recognition model (``rec.onnx``);
        text detection is handled by a separate detector module.
        """
        # 1. Make PyTorch/env CUDA runtime libraries visible before ONNX
        #    Runtime tries to load its CUDA provider.
        self._ensure_cuda_dll_path()
        from .utils.onnxocr import TextRecognizer
        import onnxruntime

        _providers = onnxruntime.get_available_providers()
        _device = self.get_param_value("device")
        _use_cuda = _device == "cuda"
        _use_coreml = _device == "mps"
        if _use_cuda and "CUDAExecutionProvider" not in _providers:
            self.logger.warning(
                "CUDA device selected but onnxruntime CUDA provider not found.\n"
                "  Install onnxruntime-gpu for GPU acceleration:\n"
                "    pip install onnxruntime-gpu\n"
                "  Falling back to CPU for this session."
            )
            _device = 'cpu'
        if _use_coreml and "CoreMLExecutionProvider" not in _providers:
            self.logger.warning(
                "MPS device selected but onnxruntime CoreML provider not found.\n"
                "  Install a macOS ONNX Runtime build with CoreML support.\n"
                "  Falling back to CPU for this session."
            )
            _device = 'cpu'


        # 3. Create the recognizer directly — no detector involved.

        rec_path = osp.join(MODEL_DIR, 'inference.onnx')
        dict_path = osp.abspath(osp.join(MODEL_DIR, "ppocrv6_dict_proper.txt"))
        if not osp.exists(dict_path) or osp.getsize(dict_path) == 0:
            _write_ppocr_dict_from_yaml(osp.join(MODEL_DIR, "inference.yml"), dict_path)

        args = argparse.Namespace(
            rec_algorithm="SVTR_LCNet",
            rec_model_dir=rec_path,
            rec_image_shape="3, 48, 320",
            rec_batch_num=self.get_param_value("rec_batch_num"),
            max_text_length=25,
            rec_char_dict_path=dict_path,
            use_space_char=True,
            device=_device,
            drop_score=0.5,
        )
        self.recognizer = TextRecognizer(args)
        _restore_text_recognizer_call(self.recognizer.__class__)

        # 4. On GPU, CUDA kernel caches are per-shape — any change to the
        #    batch dimension *or* the spatial width invalidates the cache
        #    and triggers full kernel recompilation (~1.8 s per batch).
        #    The recognizer's ``resize_norm_img`` computes a dynamic width
        #    per batch (e.g. 320 for narrow crops, 374 for wide ones),
        #    destroying the cache across calls.
        #
        #    Two-part fix:
        #      a) Force ``resize_norm_img`` to *always* pad to
        #         ``rec_image_shape`` width (320) — wide crops get clamped,
        #         which is fine because the model was trained at this width.
        #      b) Pad the last batch to ``rec_batch_num`` entries so the batch
        #         dimension never changes either.
        if _use_cuda:
            _orig_resize = self.recognizer.resize_norm_img

            def _fixed_resize(img, max_wh_ratio):
                result = _orig_resize(img, max_wh_ratio)
                _, _, fixed_w = (3, 48, 320)  # rec_image_shape
                if result.shape[2] != fixed_w:
                    import numpy as _np

                    fixed = _np.zeros(
                        (result.shape[0], result.shape[1], fixed_w), dtype=_np.float32
                    )
                    w = min(result.shape[2], fixed_w)
                    fixed[:, :, :w] = result[:, :, :w]
                    result = fixed
                return result

            self.recognizer.resize_norm_img = _fixed_resize

    # ── GPU uniform-batch recognition ───────────────────────────────────

    def _recognize_crops(self, img_list: List[np.ndarray]):
        """Recognize crops, padding only this call for stable GPU batch shapes.

        >>> PaddleOCRv6ONNX._pad_to_batch([], 6)
        []
        """
        if self.get_param_value("device") != "cuda":
            return self.recognizer(img_list)

        original_len = len(img_list)
        padded_img_list = self._pad_to_batch(img_list, self.get_param_value("rec_batch_num"))
        results = self.recognizer(padded_img_list)
        return results[:original_len]

    @staticmethod
    def _pad_to_batch(img_list: List[np.ndarray], batch_num: int):
        """Pad a crop list with blank images until its length fits ``batch_num``.

        >>> PaddleOCRv6ONNX._pad_to_batch([], 6)
        []
        """
        if not img_list or batch_num <= 1:
            return img_list

        remainder = len(img_list) % batch_num
        if remainder == 0:
            return img_list

        pad_count = batch_num - remainder
        first_crop = img_list[0]
        dummy_crop = np.zeros_like(first_crop)
        # Keep padding local to this OCR call; patching TextRecognizer.__call__
        # globally stacks across module reloads and can change recognition.
        return img_list + [dummy_crop.copy() for _ in range(pad_count)]

    # ── OCRBase interface — crop-based recognition ──────────────────────

    def _ocr_blk_list(
        self, img: np.ndarray, blk_list: List[TextBlock],
        split_textblk=False, seg_func=None, *args, **kwargs
    ):
        """Recognise text in pre‑detected blocks by cropping each polygon.

        Iterates each ``TextBlock.lines``, crops the region from the source
        image using perspective-correct cropping, and runs the ONNX
        recognizer on all crops in a single batched call.
        """
        if not blk_list:
            return

        if self.recognizer is None:
            self.load_model()

        # Collect all line crops alongside their block index
        all_crops: list[np.ndarray] = []
        crop_to_blk: list[int] = []  # parallel: blk index for each crop

        for blk_idx, blk in enumerate(blk_list):
            lines = blk.lines

            if split_textblk and len(blk) == 1:
                split_crops, _ = collect_textblock_regions(
                    img, [blk], text_height=48, maxwidth=8100,
                    split_textblk=True, seg_func=seg_func
                )
                if split_crops:
                    all_crops.extend(split_crops)
                    crop_to_blk.extend([blk_idx] * len(split_crops))
                    continue

            for line in lines:
                poly = np.array(line, dtype=np.float32)
                crop = get_rotate_crop_image(img, poly)
                all_crops.append(crop)
                crop_to_blk.append(blk_idx)

        if not all_crops:
            return

        # Single batched call to the recognizer
        rec_results = self._recognize_crops(all_crops)

        # Group recognised text back to each block
        block_texts: list[list[str]] = [[] for _ in range(len(blk_list))]
        for i, blk_idx in enumerate(crop_to_blk):
            if i < len(rec_results):
                text, score = rec_results[i]
                if text and score >= 0.3:
                    block_texts[blk_idx].append(text)

        for blk_idx, texts in enumerate(block_texts):
            if texts:
                blk_list[blk_idx].text = texts

    def ocr_img(self, img: np.ndarray, **kwargs) -> str:
        self.logger.warning(
            "ocr_img() is not supported in recognition-only mode — "
            "use a text detector + OCR pipeline instead."
        )
        return ""
