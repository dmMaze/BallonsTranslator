from typing import Dict, List

import cv2
import numpy as np

from .base import InpainterBase, register_inpainter
from ..llm_image import LLMImageRequester
from ..textdetector import TextBlock
from ballontranslator.modules.exceptions import (
    LLMRequestStopped,
    LLMUserActionRequiredError,
)
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import (
    LLMProfile,
    runtime_profile,
)


LLM_IMAGE_REQUEST_PARAMS: Dict = {
    "max requests per minute": {
        "value": 5,
        "display_name": "Max Requests Per Minute",
        "description": "Global request limit for LLM image requests.",
    },
    "delay": {
        "value": 0.5,
        "display_name": "Delay",
        "description": "Delay between LLM image requests in seconds.",
    },
    "retry attempts": {
        "value": 3,
        "display_name": "Retry Attempts",
        "description": "Retries for API failures.",
    },
    "retry timeout": {
        "value": 7.0,
        "display_name": "Retry Timeout",
        "description": "Delay between retries in seconds.",
    },
    "request timeout": {
        "value": 180.0,
        "display_name": "Request Timeout",
        "description": (
            "HTTP timeout for image requests in seconds. Set to 0 to disable."
        ),
    },
    "max resolution": {
        "type": "selector",
        "options": [0, 256, 768, 1280],
        "value": 1280,
        "display_name": "Max Resolution",
        "description": (
            "Scale images down before sending them to the LLM. "
            "Set to 0 to keep the original size."
        ),
    },
    "proxy": {
        "value": "",
        "display_name": "Proxy",
        "description": "Proxy address used for the image request.",
    },
}


@register_inpainter("LLMInpaint")
class LLMInpaint(LLMImageRequester, InpainterBase):
    """Profile-backed image cleanup using shared LLM image transport.

    Example:
        >>> LLMInpaint._image_model_required('demo', ['demo'])
        'demo'
    """

    dependencies = ['httpx[socks,brotli]']

    params: Dict = {
        **LLM_IMAGE_REQUEST_PARAMS,
        "inpaint by block": {
            "type": "checkbox",
            "value": True,
            "display_name": "Inpaint By Block",
            "description": "Send each text block crop separately instead of sending the whole image.",
        },
        "description": "Inpaint using the selected image-capable LLM profile.",
    }

    inpaint_by_block = True

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._sync_inpaint_by_block()

    @property
    def profile(self) -> LLMProfile:
        profile = runtime_profile(
            pcfg.module.llm_profiles, pcfg.module.inpaint_llm_id
        )
        if not profile.support_image:
            raise RuntimeError(
                f'LLM profile "{profile.name}" does not have image cleanup enabled.'
            )
        self._image_model(profile)
        self._image_base_url(profile)
        return profile

    def _sync_inpaint_by_block(self) -> None:
        value = self.get_param_value('inpaint by block')
        if isinstance(value, str):
            value = value.lower().strip() == 'true'
        self.inpaint_by_block = bool(value)

    def updateParam(self, param_key: str, param_content) -> None:
        super().updateParam(param_key, param_content)
        if param_key == 'inpaint by block':
            self._sync_inpaint_by_block()

    def _inpaint(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        textblock_list: List[TextBlock] = None,
    ) -> np.ndarray:
        del textblock_list
        profile = self.profile
        retry_attempt = 0
        mask_original = (mask > 127)[..., None].astype(np.uint8)
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise LLMRequestStopped()
            try:
                result = self._request_inpaint(profile, img)
                if result.shape[:2] != img.shape[:2]:
                    result = cv2.resize(
                        result,
                        (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                result = result.astype(np.uint8, copy=False)
                return result * mask_original + img * (1 - mask_original)
            except (LLMUserActionRequiredError, LLMRequestStopped):
                raise
            except Exception as error:
                retry_attempt += 1
                if retry_attempt >= self.get_param_value('retry attempts'):
                    raise RuntimeError(
                        f'LLM image cleanup failed: {error}'
                    ) from error
                self.logger.warning(
                    'LLM image cleanup failed due to %s. Attempt: %s',
                    error,
                    retry_attempt,
                )
                self._wait(self.get_param_value('retry timeout'))
