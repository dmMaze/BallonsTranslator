"""Image-effect generation backend, context, and worker boundaries."""

import math
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
from qtpy.QtCore import QCoreApplication, QObject, Qt, Signal, Slot

from ballontranslator.modules.exceptions import LLMRequestStopped
from ballontranslator.modules.llm_image import (
    LLMImageRequester,
    LLMImageRequestPolicy,
)
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import (
    LLM_INPAINT_KEY,
    profile_by_id,
    profile_from_config,
)
from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.utils.text_effects import ImageGenerationRecipe
from ..rendering.raster import EffectRasterAllocationError


class ImageGenerationBackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ImageGenerationRequest:
    """Provider-neutral inputs for one Image-card request.

    A local Diffusers adapter can consume this same prompt/model/context
    boundary without changing the card, crop, worker, or commit ownership.

    >>> ImageGenerationRequest(ImageGenerationRecipe(), None).recipe.context
    'source'
    """

    recipe: ImageGenerationRecipe
    context_image: Optional[np.ndarray]


class ImageGenerationBackend:
    """Minimal backend contract shared by remote and future local models.

    >>> ImageGenerationBackend.backend_id
    ''
    """

    backend_id = ''

    def generate(
        self,
        request: ImageGenerationRequest,
        stop_event: threading.Event,
    ) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class LLMImageGenerationBackend(ImageGenerationBackend):
    """Adapt the shared LLM image transport to the generation boundary.

    >>> LLMImageGenerationBackend.backend_id
    'llm'
    """

    backend_id = 'llm'

    def __init__(self, profile, policy: LLMImageRequestPolicy) -> None:
        self.profile = profile
        self.requester = LLMImageRequester(image_request_policy=policy)

    def generate(
        self,
        request: ImageGenerationRequest,
        stop_event: threading.Event,
    ) -> np.ndarray:
        self.requester.set_stop_event(stop_event)
        return self.requester.request_image_with_retries(
            self.profile,
            request.context_image,
            request.recipe.prompt,
            request.recipe.model,
        )

    def close(self) -> None:
        self.requester.close()


def create_image_generation_backend(
    recipe: ImageGenerationRecipe,
) -> ImageGenerationBackend:
    if recipe.backend != 'llm':
        raise ImageGenerationBackendUnavailable(
            QCoreApplication.translate(
                'ImageGeneration',
                'Image generation backend "{backend}" is unavailable.',
            ).format(backend=recipe.backend)
        )
    profile = profile_by_id(pcfg.module.llm_profiles, recipe.profile_id)
    if profile is None or not profile.support_image:
        raise ImageGenerationBackendUnavailable(
            QCoreApplication.translate(
                'ImageGeneration',
                'Image generation profile "{profile}" is unavailable.',
            ).format(profile=recipe.profile_id)
        )
    if not recipe.model.strip():
        raise ImageGenerationBackendUnavailable(
            QCoreApplication.translate(
                'ImageGeneration',
                'Select an image generation model first.',
            )
        )
    params = pcfg.module.inpainter_params.get(LLM_INPAINT_KEY, {})
    policy = LLMImageRequestPolicy.from_module_params(
        params if isinstance(params, dict) else None
    )
    return LLMImageGenerationBackend(profile_from_config(profile), policy)


@dataclass(frozen=True)
class LogicalCrop:
    """Integer pixels covering one validated logical text rectangle.

    >>> LogicalCrop(1, 2, 5, 8, 1.0, 2.0).width
    4
    """

    left: int
    top: int
    right: int
    bottom: int
    logical_x: float
    logical_y: float

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def validate_logical_crop(item, image: np.ndarray) -> LogicalCrop:
    """Validate the exact pre-transform item rectangle against one page.

    >>> callable(validate_logical_crop)
    True
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError(QCoreApplication.translate(
            'ImageGeneration', 'The selected image context is unavailable.'
        ))
    image_height, image_width = image.shape[:2]
    rect = item.absBoundingRect(qrect=True)
    values = (rect.x(), rect.y(), rect.width(), rect.height())
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(QCoreApplication.translate(
            'ImageGeneration', 'The text item crop is not finite.'
        ))
    if rect.width() <= 0 or rect.height() <= 0:
        raise ValueError(QCoreApplication.translate(
            'ImageGeneration', 'The text item crop is empty.'
        ))
    left_f = float(rect.x())
    top_f = float(rect.y())
    right_f = left_f + float(rect.width())
    bottom_f = top_f + float(rect.height())
    if (
        left_f < 0
        or top_f < 0
        or right_f > image_width
        or bottom_f > image_height
    ):
        raise ValueError(QCoreApplication.translate(
            'ImageGeneration',
            'The text item crop lies outside the current image.',
        ))
    left = math.floor(left_f)
    top = math.floor(top_f)
    right = math.ceil(right_f)
    bottom = math.ceil(bottom_f)
    if left >= right or top >= bottom:
        raise ValueError(QCoreApplication.translate(
            'ImageGeneration',
            'The text item crop does not contain any pixels.',
        ))
    return LogicalCrop(left, top, right, bottom, left_f, top_f)


def _lettered_context(item, base: np.ndarray, crop: LogicalCrop) -> np.ndarray:
    try:
        rgba = item.effect_renderer.capture_plain_logical_rgba(
            crop.width,
            crop.height,
            crop.logical_x - crop.left,
            crop.logical_y - crop.top,
        )
    except (EffectRasterAllocationError, MemoryError) as error:
        raise ValueError(QCoreApplication.translate(
            'ImageGeneration',
            'Unable to render the Lettered image context.',
        )) from error
    if base.shape[2] == 3:
        alpha = np.full(base.shape[:2] + (1,), 255, dtype=np.uint8)
        base_rgba = np.concatenate((base, alpha), axis=2)
    elif base.shape[2] == 4:
        base_rgba = base
    else:
        raise ValueError(QCoreApplication.translate(
            'ImageGeneration',
            'The inpainted image context is not RGB(A).',
        ))
    composited = Image.alpha_composite(
        Image.fromarray(np.ascontiguousarray(base_rgba), mode='RGBA'),
        Image.fromarray(np.ascontiguousarray(rgba), mode='RGBA'),
    )
    result = np.array(composited)
    return result[:, :, :3] if base.shape[2] == 3 else result


def prepare_image_generation_context(
    item,
    project,
    context: str,
) -> Optional[np.ndarray]:
    """Return one owned context crop, or ``None`` for prompt-only requests."""
    if context == 'none':
        return None
    if context == 'source':
        image = project.img_array
    elif context in {'inpainted', 'lettered'}:
        image = project.inpainted_array
    else:
        raise ValueError(QCoreApplication.translate(
            'ImageGeneration', 'Unsupported image generation context.'
        ))
    crop = validate_logical_crop(item, image)
    result = np.array(
        image[crop.top:crop.bottom, crop.left:crop.right],
        copy=True,
        order='C',
    )
    if context == 'lettered':
        result = _lettered_context(item, result, crop)
    result.flags.writeable = False
    return result


def _encode_generated_png(image: np.ndarray) -> bytes:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] not in (3, 4)
        or image.shape[0] <= 0
        or image.shape[1] <= 0
    ):
        raise ValueError(QCoreApplication.translate(
            'ImageGeneration',
            'Image generation returned an invalid RGB(A) image.',
        ))
    import io

    output = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(output, format='PNG')
    return output.getvalue()


_LIVE_IMAGE_GENERATION_JOBS = set()
_IMAGE_GENERATION_SHUTDOWN_APP = None


def _shutdown_live_image_generation_jobs() -> None:
    """Abandon UI delivery without waiting for blocked provider POSTs."""
    for job in tuple(_LIVE_IMAGE_GENERATION_JOBS):
        job.request_stop(abandon_inflight=True)


def _ensure_image_generation_shutdown_hook() -> None:
    global _IMAGE_GENERATION_SHUTDOWN_APP
    app = QCoreApplication.instance()
    if app is None or app is _IMAGE_GENERATION_SHUTDOWN_APP:
        return
    app.aboutToQuit.connect(_shutdown_live_image_generation_jobs)
    _IMAGE_GENERATION_SHUTDOWN_APP = app


class _ImageGenerationJob(QObject):
    """Run one provider request on one retained daemon-backed QObject.

    The QObject stays in the Qt thread while its daemon performs provider IO;
    signal delivery and normal cleanup are therefore queued back to Qt.

    >>> issubclass(_ImageGenerationJob, QObject)
    True
    """

    succeeded = Signal(int, bytes)
    failed = Signal(int, object)
    completed = Signal(int)

    def __init__(
        self,
        token: int,
        backend: ImageGenerationBackend,
        request: ImageGenerationRequest,
    ) -> None:
        super().__init__(None)
        self.token = int(token)
        self.backend = backend
        self.request = request
        self.stop_event = threading.Event()
        self._abandon_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name='text-image-generation-request',
            daemon=True,
        )
        self.completed.connect(
            self._release_after_finish,
            Qt.ConnectionType.QueuedConnection,
        )

    def start(self) -> None:
        self._thread.start()

    def request_stop(self, *, abandon_inflight: bool = False) -> None:
        self.stop_event.set()
        if abandon_inflight:
            self._abandon_event.set()

    def _run(self) -> None:
        payload = None
        request_error = None
        try:
            result = self.backend.generate(
                self.request, self.stop_event
            )
            if not self.stop_event.is_set():
                payload = _encode_generated_png(result)
        except Exception as error:
            request_error = error
        finally:
            try:
                self.backend.close()
            except Exception as cleanup_error:
                LOGGER.warning(
                    'Unable to close image generation backend: %s',
                    cleanup_error,
                )
        if self._abandon_event.is_set():
            # Keep this parentless QObject retained until process teardown;
            # QApplication is already exiting and must not await the POST.
            return
        if request_error is not None and not isinstance(
            request_error, LLMRequestStopped
        ):
            if not self.stop_event.is_set():
                self.failed.emit(self.token, request_error)
        elif request_error is None and not self.stop_event.is_set():
            self.succeeded.emit(self.token, payload)
        self.completed.emit(self.token)

    @Slot(int)
    def _release_after_finish(self, token: int) -> None:
        if token != self.token:
            return
        _LIVE_IMAGE_GENERATION_JOBS.discard(self)
        self.deleteLater()


class ImageGenerationController(QObject):
    """Keep one worker active and marshal its result to the Qt thread.

    >>> controller = ImageGenerationController()
    >>> controller.active
    False
    """

    generated = Signal(int, bytes)
    failed = Signal(int, object)
    state_changed = Signal(int, str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._next_token = 1
        self._worker: Optional[_ImageGenerationJob] = None
        self._active_index = -1
        self._discard_result = False

    @property
    def active(self) -> bool:
        return self._worker is not None

    def start(
        self,
        index: int,
        backend: ImageGenerationBackend,
        request: ImageGenerationRequest,
    ) -> bool:
        if self._worker is not None:
            return False
        token = self._next_token
        self._next_token += 1
        worker = _ImageGenerationJob(token, backend, request)
        _ensure_image_generation_shutdown_hook()
        _LIVE_IMAGE_GENERATION_JOBS.add(worker)
        queued = Qt.ConnectionType.QueuedConnection
        worker.succeeded.connect(self._on_succeeded, queued)
        worker.failed.connect(self._on_failed, queued)
        worker.completed.connect(self._on_finished, queued)
        self._worker = worker
        self._active_index = int(index)
        self._discard_result = False
        self.state_changed.emit(self._active_index, 'running')
        worker.start()
        return True

    def stop(self) -> bool:
        worker = self._worker
        if worker is None:
            return False
        self._discard_result = True
        worker.request_stop()
        self.state_changed.emit(self._active_index, 'stopping')
        return True

    @Slot(int, bytes)
    def _on_succeeded(self, token: int, payload: bytes) -> None:
        worker = self._worker
        if (
            worker is not None
            and worker.token == token
            and not self._discard_result
        ):
            self.generated.emit(self._active_index, payload)

    @Slot(int, object)
    def _on_failed(self, token: int, error: Exception) -> None:
        worker = self._worker
        if (
            worker is not None
            and worker.token == token
            and not self._discard_result
        ):
            self.failed.emit(self._active_index, error)

    @Slot(int)
    def _on_finished(self, token: int) -> None:
        worker = self._worker
        if worker is None or worker.token != token:
            return
        index = self._active_index
        self._worker = None
        self._active_index = -1
        self._discard_result = False
        self.state_changed.emit(index, 'idle')
