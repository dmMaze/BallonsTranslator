"""Shared image references and Responses image-generation wire format."""

import base64
from typing import Optional

from .exceptions import LLMUserActionRequiredError


MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_BASE64_BYTES = ((MAX_IMAGE_BYTES + 2) // 3) * 4
MAX_IMAGE_RESPONSE_BYTES = MAX_IMAGE_BASE64_BYTES + 1024 * 1024


def image_references(image: Optional[bytes], mask: Optional[bytes]) -> list[dict[str, str]]:
    """Encode bounded PNG references in source-then-mask order.

    >>> image_references(None, None)
    []
    """
    if mask is not None and image is None:
        raise ValueError('An image mask requires an input image.')
    references = []
    for raw in (image, mask):
        if raw is None:
            continue
        if not isinstance(raw, bytes) or not raw.startswith(b'\x89PNG\r\n\x1a\n') or len(raw) > MAX_IMAGE_BYTES:
            raise ValueError('Image references must be PNG bytes within the image size limit.')
        references.append({'image_url': 'data:image/png;base64,' + base64.b64encode(raw).decode('ascii')})
    return references


def responses_image_payload(
    reasoning_model: str, image_model: str, prompt: str,
    references: list[dict[str, str]], cache_key: str, *, stream: bool,
) -> dict:
    """Force one stateless image operation through the selected reasoning model.

    >>> responses_image_payload('gpt-reason', 'gpt-image', 'Draw.', [], 'job', stream=False)['tools'][0]['action']
    'generate'
    """
    return {
        'model': reasoning_model, 'store': False, 'stream': stream,
        'instructions': "Use the image generation tool once to fulfill the user's request. Return the resulting image.",
        'input': [{'type': 'message', 'role': 'user', 'content': [
            {'type': 'input_text', 'text': prompt},
            *({'type': 'input_image', **reference, 'detail': 'high'} for reference in references),
        ]}],
        'tools': [{'type': 'image_generation', 'model': image_model,
                   'action': 'edit' if references else 'generate', 'quality': 'auto', 'size': 'auto'}],
        'tool_choice': {'type': 'image_generation'},
        'prompt_cache_key': cache_key,
    }


def decode_inline_image(encoded: object, *, error_type: type[Exception] = LLMUserActionRequiredError) -> bytes:
    """Decode one bounded inline result; never follow provider-returned URLs.

    >>> decode_inline_image('YQ==')
    b'a'
    """
    if not isinstance(encoded, str) or not encoded or len(encoded) > MAX_IMAGE_BASE64_BYTES:
        raise error_type('The image provider returned no valid inline image data.')
    try:
        result = base64.b64decode(encoded, validate=True)
    except ValueError:
        raise error_type('The image provider returned invalid image data.') from None
    if not result or len(result) > MAX_IMAGE_BYTES:
        raise error_type('The image provider returned empty or oversized image data.')
    return result


def decode_responses_image(response: object) -> bytes:
    """Accept exactly one completed image, without committing refused output.

    >>> decode_responses_image({'status': 'completed', 'output': [
    ...     {'type': 'image_generation_call', 'status': 'completed', 'result': 'YQ=='}]})
    b'a'
    """
    if not isinstance(response, dict) or response.get('status') != 'completed' or response.get('error'):
        raise LLMUserActionRequiredError('The assisted image request did not complete. Review the request and model pair.')
    output = response.get('output')
    if not isinstance(output, list) or any(not isinstance(item, dict) for item in output):
        raise LLMUserActionRequiredError('Assisted editing returned no valid image. Review the request or select a direct image model.')
    for item in output:
        content = item.get('content')
        if item.get('type') == 'message' and isinstance(content, list) and any(
            isinstance(part, dict) and part.get('type') == 'refusal' for part in content
        ):
            raise LLMUserActionRequiredError('The image provider declined the request. Review the current input before retrying.')
    images = [item for item in output if item.get('type') == 'image_generation_call']
    if len(images) != 1 or images[0].get('status') != 'completed':
        raise LLMUserActionRequiredError('Assisted editing did not return one completed image. Review the request or select a direct image model.')
    return decode_inline_image(images[0].get('result'))
