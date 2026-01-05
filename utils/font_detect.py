import os
import hashlib
import logging
import sys
import torch

from typing import Tuple
from typing import Dict

logger = logging.getLogger(__name__)

try:
    import torch.nn as nn
    from torchvision import models
    from PIL import Image
    from torchvision import transforms
except Exception as e:
    torch = None
    nn = None
    models = None
    Image = None
    transforms = None

from utils import shared
from utils import download_util

MODEL_REL_PATH = os.path.join('data', 'models', 'YuzuMarker.FontDetection', 'name=4x-epoch=18-step=368676.ckpt')
MODEL_URL = 'https://huggingface.co/gyrojeff/YuzuMarker.FontDetection/resolve/main/name=4x-epoch=18-step=368676.ckpt'
MODEL_SHA256 = '4544568829be10a98653a2c965f82fb229d5e02146578ccb3402518d9c022b1a'
CACHE_REL_PATH = os.path.join('data','font_demo_cache.bin')

# Add the YuzuMarker.FontDetection directory to the Python path so we can import font_dataset
YUZUMARKER_DIR = os.path.join(shared.PROGRAM_PATH, 'data', 'models', 'YuzuMarker.FontDetection')
if YUZUMARKER_DIR not in sys.path:
    sys.path.insert(0, YUZUMARKER_DIR)


def _sha256_of_file(path: str) -> str:
    logger.debug(f"Computing SHA256 for file: {path}")
    try:
        if not os.path.exists(path):
            logger.error(f"File does not exist: {path}")
            raise FileNotFoundError(f"File does not exist: {path}")
        result = download_util.calculate_sha256(path)
        logger.debug(f"SHA256 computation completed: {result[:16]}...")
        return result
    except FileNotFoundError as e:
        logger.error(f"File not found during SHA256 computation: {e}")
        raise
    except PermissionError as e:
        logger.error(f"Permission error during SHA256 computation: {e}")
        raise
    except OSError as e:
        logger.error(f"OS error during SHA256 computation: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during SHA256 computation: {e}")
        logger.exception("Exception details:")
        raise


def _download_file(url: str, dst: str) -> None:
    logger.info(f"Starting download from {url} to {dst}")
    try:
        download_util.download_url_to_file(url, dst, progress=True)
        logger.info(f"Model successfully downloaded to {dst}")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise RuntimeError(f'Download failed: {e}')


def _clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("state_dict."):
            new_k = new_k[len("state_dict.") :]
        if new_k.startswith("model."):
            new_k = new_k[len("model.") :]
        # compile may add _orig_mod.
        new_k = new_k.replace("_orig_mod.", "")
        # DataParallel
        if new_k.startswith("module."):
            new_k = new_k[len("module.") :]
        new_sd[new_k] = v
    return new_sd

def prepare_fonts(cache_path: str = None):
    """Load font list from cache file"""
    try:
        if cache_path and os.path.exists(cache_path):
            if YUZUMARKER_DIR not in sys.path:
                sys.path.insert(0, YUZUMARKER_DIR)
            
            with open(cache_path, 'rb') as f:
                import pickle
                font_objects = pickle.load(f)
                
                # Convert font objects to their path strings
                font_list = []
                for font_obj in font_objects:
                    if hasattr(font_obj, 'path'):
                        font_list.append(font_obj.path)
                    else:
                        # Fallback: if the object doesn't have path attribute, keep the original object
                        font_list.append(font_obj)
                
                return font_list
        else:
            pass
    except FileNotFoundError as e:
        logger.warning(f"Font cache file not found at {cache_path}: {e}")
    except PermissionError as e:
        logger.warning(f"Permission error accessing font cache at {cache_path}: {e}")
    except pickle.PickleError as e:
        logger.warning(f"Error unpickling font cache at {cache_path}: {e}")
        logger.exception("Exception details:")
    except Exception as e:
        logger.warning(f"Could not load font cache at {cache_path}: {e}")
        logger.exception("Exception details:")
    # Return a default font list if cache is not available
    return []

class FontDetector:
    _instance = None

    def __init__(self, device: str = 'cpu', input_size: int = 512):
        if torch is None or models is None:
            raise RuntimeError('Torch or torchvision is not available')
        self.device = torch.device(device)
        self.input_size = input_size
        self.model = None
        self.font_list = None
        self.model_path = os.path.join(shared.PROGRAM_PATH, MODEL_REL_PATH)
        self.cache_path = os.path.join(shared.PROGRAM_PATH, CACHE_REL_PATH)

    @classmethod
    def get_instance(cls) -> 'FontDetector':
        if cls._instance is None:
            cls._instance = FontDetector()
        return cls._instance

    def ensure_model(self):
        # ensure model file exists, otherwise download
        if not os.path.exists(self.model_path):
            logger.info(f'Model file does not exist at {self.model_path}, preparing to download')
            try:
                # create dir
                model_dir = os.path.dirname(self.model_path)
                os.makedirs(model_dir, exist_ok=True)
                logger.info(f'Downloading font detection model to {self.model_path}...')
                _download_file(MODEL_URL, self.model_path)
                logger.info(f'Model download completed')
            except Exception as e:
                logger.error(f'Failed to download model: {e}')
                raise
        else:
            pass
        # verify sha
        try:
            sha = _sha256_of_file(self.model_path)
            if sha != MODEL_SHA256:
                logger.warning('Model sha256 mismatch: %s != %s', sha, MODEL_SHA256)
        except Exception as e:
            logger.error(f'Failed to compute model sha256: {e}')
            logger.exception('Exception details:')
            raise

    def load(self):
        if self.model is not None:
            return
        try:
            self.ensure_model()
            
            # Load the checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extract state dict
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                raw_sd = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                raw_sd = checkpoint
            else:
                raw_sd = checkpoint
                
            # Clean state dict keys
            clean_sd = _clean_state_dict_keys(raw_sd)
            
            # Create ResNet50 model
            self.model = models.resnet50(pretrained=False)
            # Modify the final layer based on the expected number of font classes
            # We'll determine the number of classes from the loaded weights
            num_classes = None
            for key, value in clean_sd.items():
                if 'fc.weight' in key:
                    num_classes = value.shape[0]
                    break
            
            if num_classes is not None:
                self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            else:
                logger.warning("Could not determine number of classes from state dict, using default")
                
            # If so, remove the prefix to match the model architecture
            if any(key.startswith('model.') for key in clean_sd.keys()):
                new_clean_sd = {}
                for key, value in clean_sd.items():
                    if key.startswith('model.'):
                        new_key = key[6:]  # Remove 'model.' prefix
                        new_clean_sd[new_key] = value
                    else:
                        new_clean_sd[key] = value
                clean_sd = new_clean_sd
        
            # Load the state dict
            try:
                self.model.load_state_dict(clean_sd, strict=True)
            except Exception as e:
                logger.warning(f"Strict load failed: {e}")
                logger.info("Attempting load with strict=False")
                self.model.load_state_dict(clean_sd, strict=False)
            
            self.model.to(self.device)
            self.model.eval()

            # Load fonts cache
            try:
                self.font_list = prepare_fonts(self.cache_path)
            except Exception as e:
                logger.warning(f"Could not load font cache: {e}")
                self.font_list = prepare_fonts()

            # simple transform
            self.transform = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)), 
                transforms.ToTensor()
            ])
        except Exception as e:
            logger.error(f"Error loading font detector: {e}")
            logger.exception("Exception details:")
            raise

    def detect(self, img_bgr) -> Tuple[str, float]:
        """Detect font from a BGR numpy image (cv2 style). Returns (font_name, confidence).
        If confidence < 0.6 returns ("UNKNOWN", 0.0) per requirement.
        """
        if self.model is None:
            self.load()
        if img_bgr is None:
            logger.warning("Input image is None, returning UNKNOWN")
            return "UNKNOWN", 0.0
        try:
            import cv2
            # convert to RGB PIL Image
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb).convert('RGB')
            t = self.transform(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(t)
                
                # The model may have more outputs than actual fonts, so we only consider the first FONT_COUNT
                font_count = min(6150, out.shape[1])  # Use the smaller of 6150 or actual output size
                probs = out[0][:font_count].softmax(dim=0)
                top_idx = int(probs.argmax().cpu().item())
                conf = float(probs[top_idx].cpu().item())
            # map index to font_list
            font_name = None
            if self.font_list is not None and top_idx < len(self.font_list):
                font_path = self.font_list[top_idx]
                font_name = os.path.splitext(os.path.basename(font_path))[0]
            else:
                font_name = f'font_{top_idx}'

            if conf < 0.6:
                return 'UNKNOWN', 0.0
            return font_name, conf
        except Exception as e:
            logger.error(f"Error during font detection: {e}")
            logger.exception("Exception details:")
            return "UNKNOWN", 0.0


# convenience function
def detect_font_from_block(img, blk) -> Tuple[str, float]:
    """Given full page image and a TextBlock, crop a reasonable region and run detection."""
    try:
        detector = FontDetector.get_instance()
        # choose bounding rect
        try:
            x, y, w, h = blk.bounding_rect()
        except Exception as e:
            logger.warning(f"Could not get bounding rect from block: {e}, using full image as fallback")
            # fallback to full image
            x, y, w, h = 0, 0, img.shape[1], img.shape[0]
        x2 = min(img.shape[1], x + w)
        y2 = min(img.shape[0], y + h)
        x1 = max(0, x)
        y1 = max(0, y)
        if x2 <= x1 or y2 <= y1:
            region = img
        else:
            region = img[y1:y2, x1:x2]
        # if region too small, resize with padding
        import cv2
        h, w = region.shape[:2]
        if h == 0 or w == 0:
            logger.warning(f"Region has zero size (h={h}, w={w}), using full image")
            region = img
        result = detector.detect(region)
        return result
    except Exception as e:
        logger.error(f"Error in detect_font_from_block: {e}")
        logger.exception("Exception details:")
        return "UNKNOWN", 0.0
