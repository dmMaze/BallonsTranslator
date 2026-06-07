# Lens_OCR_exp.py
# https://github.com/AuroraWright/owocr
import re
import numpy as np
import time
import random
from typing import List, Dict, Any, Tuple, Optional, Union
from math import sqrt
import io
import os
import json

import requests
from PIL import Image, ImageFile

import betterproto

try:
    from .utils.lens_betterproto import *
except ImportError:
    try:
        from .utils.lens_betterproto import *
    except ImportError:
        raise ImportError(
            "Could not import lens_betterproto. "
            "Make sure lens_betterproto.py exists."
        ) from None

try:
    from .base import register_OCR, OCRBase, TextBlock
except ImportError:

    class OCRBase:
        def __init__(self, **params):
            self.params = params
            self.debug_mode = int(os.environ.get("OCR_DEBUG", 0))
            # Basic logger implementation if run standalone
            import logging

            self.logger = logging.getLogger(__name__)
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "[%(levelname)-5s] %(name)s:%(funcName)s:%(lineno)d - %(message)s"
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)
            self.logger.info(
                "Running Lens_OCR_exp without .base module. Using placeholder classes."
            )

        def get_param_value(self, key):
            return self.params.get(key)

        def updateParam(self, key, value):
            self.params[key] = value

    def register_OCR(name):
        return lambda cls: cls

    class TextBlock:
        def __init__(self):
            self.xyxy = (0, 0, 0, 0)
            self.text = ""


ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import fpng_py

    OPTIMIZED_PNG_ENCODE = True
except ImportError:
    OPTIMIZED_PNG_ENCODE = False


def _pil_image_to_bytes(
    img: Image.Image,
    img_format="png",
    png_compression=6,
    jpeg_quality=80,
    optimize=False,
) -> bytes:
    """Converts PIL Image object to bytes of the specified format."""
    if img_format == "png" and OPTIMIZED_PNG_ENCODE and not optimize:
        try:
            rgba_img = img.convert("RGBA")
            raw_data = rgba_img.tobytes()
            image_bytes = fpng_py.fpng_encode_image_to_memory(
                raw_data, img.width, img.height
            )
            return image_bytes
        except Exception:
            pass  # Fallback to PIL

    image_bytes_io = io.BytesIO()
    save_kwargs = {}
    img_to_save = img
    if img_format == "jpeg":
        if img.mode == "RGBA" or (img.mode == "P" and "transparency" in img.info):
            background = Image.new("RGB", img.size, (255, 255, 255))
            try:
                background.paste(img, mask=img.split()[3])
            except IndexError:
                background.paste(img)
            img_to_save = background
        elif img.mode != "RGB":
            img_to_save = img.convert("RGB")

        save_kwargs["quality"] = jpeg_quality
        save_kwargs["subsampling"] = 0
        save_kwargs["optimize"] = optimize
    elif img_format == "png":
        save_kwargs["compress_level"] = png_compression
        save_kwargs["optimize"] = optimize

    img_to_save.save(image_bytes_io, format=img_format.upper(), **save_kwargs)
    return image_bytes_io.getvalue()


def _preprocess_image_for_lens(img: Image.Image) -> Tuple[Optional[bytes], int, int]:
    """Prepares image for Google Lens Protobuf API."""
    try:
        original_width, original_height = img.size
        max_pixels = 3_000_000
        if original_width * original_height > max_pixels:
            aspect_ratio = original_width / original_height
            new_w = int(sqrt(max_pixels * aspect_ratio))
            new_h = int(new_w / aspect_ratio)
            img_to_process = (
                img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                if new_w > 0 and new_h > 0
                else img
            )
        else:
            img_to_process = img

        img_bytes = _pil_image_to_bytes(img_to_process, img_format="png")
        return (img_bytes, img_to_process.width, img_to_process.height)
    except Exception as e:
        # Use print for safety if logger isn't available reliably
        print(f"ERROR: Image preprocessing failed: {e}")
        return (None, 0, 0)


@register_OCR("google_lens_exp")
class OCRLensAPI_exp(OCRBase):
    """
    OCR using the experimental Google Lens Protobuf API (using requests).
    Requires 'betterproto', 'requests', and 'lens_betterproto.py'.
    """

    params = {
        "delay": 1.5,
        "newline_handling": {
            "type": "selector",
            "options": ["preserve", "remove"],
            "value": "preserve",
            "description": "Handle newline characters in the final OCR string.",
        },
        "no_uppercase": {
            "type": "checkbox",
            "value": False,
            "description": "Convert text to lowercase except first letter of sentences.",
        },
        "target_language": {
            "value": "ja",
            "description": "Target language code (e.g., 'ja', 'en', 'ru').",
        },
        "proxy": {
            "value": "",
            "description": 'Proxy (requests format: e.g., http://user:pass@host:port or {"http": ..., "https": ...})',
        },
        "description": "OCR using Google Lens Protobuf API (requests backend)",
    }

    @property
    def request_delay(self) -> float:
        delay_val = self.get_param_value("delay")
        try:
            return max(0.0, float(delay_val))
        except (ValueError, TypeError, AttributeError):
            return 1.0

    @property
    def newline_handling(self) -> str:
        handling = self.get_param_value("newline_handling")
        return handling if handling in ["preserve", "remove"] else "preserve"

    @property
    def no_uppercase(self) -> bool:
        no_upper = self.get_param_value("no_uppercase")
        return bool(no_upper) if no_upper is not None else False

    @property
    def target_language(self) -> str:
        lang = self.get_param_value("target_language")
        return lang if isinstance(lang, str) and len(lang) >= 2 else "ja"

    @property
    def proxy(self) -> Optional[Dict[str, str]]:
        val = self.get_param_value("proxy")
        proxies_dict = None
        if isinstance(val, str) and val.strip().startswith("{"):
            try:
                parsed_dict = json.loads(val)
                if isinstance(parsed_dict, dict):
                    proxies_dict = {}
                    # Ensure keys are 'http' and 'https' for requests
                    http_key = next(
                        (
                            k
                            for k in parsed_dict
                            if k.lower() == "http" or k.lower() == "http://"
                        ),
                        None,
                    )
                    https_key = next(
                        (
                            k
                            for k in parsed_dict
                            if k.lower() == "https" or k.lower() == "https://"
                        ),
                        None,
                    )
                    if http_key:
                        proxies_dict["http"] = parsed_dict[http_key]
                    if https_key:
                        proxies_dict["https"] = parsed_dict[https_key]
                    return proxies_dict if proxies_dict else None
            except Exception:
                if val.strip():
                    proxies_dict = {"http": val.strip(), "https": val.strip()}
                else:
                    return None
        elif isinstance(val, str) and val.strip():
            proxies_dict = {"http": val.strip(), "https": val.strip()}
        elif isinstance(val, dict) and val:
            # Assume dict is already in {'http': ..., 'https': ...} format
            proxies_dict = val

        return proxies_dict if proxies_dict else None

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.last_request_time: float = 0
        self._api_url = "https://lensfrontend-pa.googleapis.com/v1/crupload"
        self._api_key = "AIzaSyDr2UxVnv_U85AbhhY8XSHSIavUW0DC-sY"
        self._api_headers = {
            "Host": "lensfrontend-pa.googleapis.com",
            "Connection": "keep-alive",
            "Content-Type": "application/x-protobuf",
            "X-Goog-Api-Key": self._api_key,
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Dest": "empty",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def _prepare_protobuf_request(
        self, image_bytes: bytes, width: int, height: int
    ) -> Optional[bytes]:
        """Creates and serializes the Protobuf request."""
        try:
            request = LensOverlayServerRequest()
            req_id = request.objects_request.request_context.request_id
            client_ctx = request.objects_request.request_context.client_context
            locale_ctx = client_ctx.locale_context
            img_data = request.objects_request.image_data

            req_id.uuid = random.randint(0, 2**64 - 1)
            req_id.sequence_id = 1
            req_id.image_sequence_id = 1
            req_id.analytics_id = random.randbytes(16)

            client_ctx.platform = Platform.WEB
            client_ctx.surface = Surface.CHROMIUM
            locale_ctx.language = self.target_language
            locale_ctx.region = "JP"
            locale_ctx.time_zone = "Asia/Tokyo"

            filter_obj = AppliedFilter(filter_type=LensOverlayFilterType.AUTO_FILTER)
            client_ctx.client_filters.filter.append(filter_obj)

            img_data.payload.image_bytes = image_bytes
            img_data.image_metadata.width = width
            img_data.image_metadata.height = height

            return bytes(request)
        except Exception as e:
            self.logger.error(
                f"Failed to prepare Protobuf request: {e}", exc_info=self.debug_mode
            )
            return None

    def _parse_protobuf_response(
        self, response_proto: LensOverlayServerResponse
    ) -> str:
        """Extracts text from Protobuf response, ignoring non-critical error Type=0."""
        extracted_text = ""
        has_error_type_0 = False

        try:
            if response_proto.error:
                err_type = response_proto.error.error_type
                if (
                    err_type != LensOverlayServerErrorErrorType.UNKNOWN_TYPE
                ):  # Check against enum value 0
                    error_type_name = LensOverlayServerErrorErrorType(err_type).name
                    self.logger.error(
                        f"Lens server returned critical error: Type={err_type} ({error_type_name})"
                    )
                    return ""
                else:
                    self.logger.debug(
                        "Lens server returned non-critical error: Type=0 (UNKNOWN_TYPE)."
                        " Attempting to extract text anyway."
                    )
                    has_error_type_0 = True

            text_layout = getattr(
                getattr(response_proto, "objects_response", None), "text", None
            )
            paragraphs = getattr(
                getattr(text_layout, "text_layout", None), "paragraphs", None
            )

            if paragraphs:
                paragraph_texts = []
                for paragraph in paragraphs:
                    para_text_builder = io.StringIO()
                    for line in getattr(paragraph, "lines", []):
                        for word in getattr(line, "words", []):
                            para_text_builder.write(getattr(word, "plain_text", ""))
                            separator = getattr(word, "text_separator", None)
                            if separator is not None:
                                para_text_builder.write(separator)
                    paragraph_texts.append(para_text_builder.getvalue())
                    para_text_builder.close()
                extracted_text = "".join(paragraph_texts).strip()
                extracted_text = re.sub(r"\s+", " ", extracted_text).strip()

                if self.debug_mode and has_error_type_0:
                    self.logger.debug(
                        "Successfully extracted text despite non-critical error Type=0."
                    )

            elif not has_error_type_0:
                if self.debug_mode:
                    self.logger.debug(
                        "Text layout not found in Protobuf structure (and no error reported)."
                    )

        except Exception as e:
            self.logger.error(
                f"Failed to parse Protobuf response structure: {e}",
                exc_info=self.debug_mode,
            )
            return ""

        return extracted_text

    def _execute_ocr_request(
        self, image_bytes: bytes, width: int, height: int
    ) -> Optional[LensOverlayServerResponse]:
        """Sends prepared request via requests and returns deserialized response."""
        payload = self._prepare_protobuf_request(image_bytes, width, height)
        if not payload:
            return None

        self._respect_delay()
        response_proto = None
        session = requests.Session()

        current_proxy = self.proxy
        if current_proxy:
            session.proxies = current_proxy
            if self.debug_mode:
                self.logger.debug(
                    f"Using requests proxy configuration: {current_proxy}"
                )

        # Determine SSL verification
        skip_ssl_verify = os.environ.get("OCR_SKIP_SSL_VERIFY", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        ssl_verify = not skip_ssl_verify

        try:
            if self.debug_mode:
                self.logger.debug(
                    f"Sending Protobuf request ({len(payload)} bytes) via requests "
                    f"to lens api (SSL Verify: {ssl_verify})"
                )

            response = session.post(
                self._api_url,
                data=payload,
                headers=self._api_headers,
                timeout=(10.0, 30.0),
                verify=ssl_verify,
            )
            self.last_request_time = time.time()

            if self.debug_mode:
                self.logger.debug(
                    f"Received requests response status: {response.status_code}"
                )

            response.raise_for_status()

            response_proto = LensOverlayServerResponse().parse(response.content)
            if self.debug_mode:
                self.logger.debug("Protobuf response parsed successfully (requests).")

        except requests.exceptions.SSLError as e:
            self.logger.error(
                f"SSL Error connecting to Lens API (requests): {e}. "
                f"If using a proxy or corporate network, check its configuration. "
                f"You might need to trust a custom CA or set OCR_SKIP_SSL_VERIFY=true (unsafe).",
                exc_info=self.debug_mode,
            )
        except requests.exceptions.HTTPError as e:
            response_text = getattr(e.response, "text", "N/A")[:500]
            self.logger.error(
                f"HTTP error from Lens API (requests): {e.response.status_code}. "
                f"Response: {response_text}",
                exc_info=self.debug_mode,
            )
        except requests.exceptions.RequestException as e:
            self.logger.error(
                f"Request error connecting to Lens API (requests): {e}",
                exc_info=self.debug_mode,
            )
        except (betterproto.Error, ValueError, TypeError) as e:
            self.logger.error(
                f"Failed to parse Protobuf response (requests): {e}",
                exc_info=self.debug_mode,
            )
            if "response" in locals() and hasattr(response, "content"):
                self.logger.debug(
                    f"Raw response content (first 500 bytes): {response.content[:500]}"
                )
        except Exception as e:
            self.logger.error(
                f"Unexpected error during OCR request (requests): {e}",
                exc_info=self.debug_mode,
            )
        finally:
            session.close()

        return response_proto

    def ocr(self, img: np.ndarray) -> str:
        """Main OCR method for a single image (numpy array)."""
        if self.debug_mode > 1:
            self.logger.debug(
                f"Starting OCR (Lens Protobuf / requests) on image shape: {img.shape}"
            )
        if img is None or img.size == 0:
            if self.debug_mode:
                self.logger.warning("Empty image provided")
            return ""

        full_text = ""
        try:
            pil_img = Image.fromarray(img)
            processed_bytes, width, height = _preprocess_image_for_lens(pil_img)

            if not processed_bytes:
                self.logger.error("Image preprocessing failed.")
                return ""
            if self.debug_mode > 1:
                self.logger.debug(
                    f"Preprocessed image: {width}x{height}, {len(processed_bytes)} bytes"
                )

            response_proto = self._execute_ocr_request(processed_bytes, width, height)

            if response_proto:
                full_text = self._parse_protobuf_response(response_proto)
                if self.debug_mode and full_text:
                    self.logger.debug(f"Parsed text preview: '{full_text[:100]}...'")

                if self.newline_handling == "remove":
                    full_text = re.sub(r"\s+", " ", full_text).strip()
                full_text = self._apply_punctuation_and_spacing(full_text)
                if self.no_uppercase:
                    full_text = self._apply_no_uppercase(full_text)
            else:
                self.logger.warning(
                    "OCR request did not return a valid response object."
                )

        except Exception as e:
            self.logger.error(
                f"Unexpected error in OCR process: {e}", exc_info=self.debug_mode
            )
            return ""

        return str(full_text) if full_text is not None else ""

    def ocr_img(self, img: np.ndarray) -> str:
        if self.debug_mode > 1:
            self.logger.debug(f"ocr_img shape: {img.shape}")
        return self.ocr(img)

    def _ocr_blk_list(
        self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs
    ):
        """Processes a list of text blocks on the image."""
        im_h, im_w = img.shape[:2]
        if self.debug_mode:
            self.logger.debug(
                f"Image size: {im_h}x{im_w}. Processing {len(blk_list)} blocks."
            )
        for i, blk in enumerate(blk_list):
            x1, y1, x2, y2 = blk.xyxy
            if self.debug_mode > 1:
                self.logger.debug(
                    f"Processing block {i+1}/{len(blk_list)}: ({x1, y1, x2, y2})"
                )

            y1c, y2c = max(0, y1), min(im_h, y2)
            x1c, x2c = max(0, x1), min(im_w, x2)

            if y1c < y2c and x1c < x2c:
                try:
                    cropped_img = img[y1c:y2c, x1c:x2c]
                    if cropped_img.size > 0:
                        blk.text = self.ocr(cropped_img)
                    else:
                        if self.debug_mode:
                            self.logger.warning(f"Empty cropped image for block {i+1}.")
                        blk.text = ""
                except Exception as crop_err:
                    self.logger.error(
                        f"Error cropping/processing block {i+1}: {crop_err}",
                        exc_info=self.debug_mode,
                    )
                    blk.text = ""
            else:
                if self.debug_mode:
                    self.logger.warning(
                        f"Invalid/zero-area bbox {blk.xyxy} (clamped: {x1c,y1c,x2c,y2c})"
                    )
                blk.text = ""

    def _apply_no_uppercase(self, text: str) -> str:
        """Applies lowercase except for first letter of sentences."""

        def process_sentence(sentence):
            sentence = sentence.strip()
            return sentence[0].upper() + sentence[1:].lower() if sentence else ""

        if self.target_language.lower().startswith("ja"):
            return text  # No case change for Japanese

        sentences = re.split(r"(?<=[.!?…])\s+", text)
        return " ".join(process_sentence(s) for s in sentences if s)

    def _apply_punctuation_and_spacing(self, text: str) -> str:
        """Corrects spacing around punctuation."""
        text = re.sub(r"\s+([,.!?…:;])", r"\1", text)
        text = re.sub(r"([,.!?…:;])(?=[^\s,.!?…:;])", r"\1 ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _respect_delay(self):
        """Ensures minimum delay between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        required_delay = self.request_delay
        if time_since_last < required_delay:
            sleep_time = required_delay - time_since_last
            if self.debug_mode:
                self.logger.info(f"Delay: Sleeping for {sleep_time:.3f}s")
            time.sleep(sleep_time)

    def updateParam(self, param_key: str, param_content: Any):
        """Updates a parameter."""
        # No client re-initialization needed for requests on proxy change
        if param_key == "delay":
            try:
                param_content = max(0.0, float(param_content))
            except (ValueError, TypeError):
                param_content = 1.0
        super().updateParam(param_key, param_content)
