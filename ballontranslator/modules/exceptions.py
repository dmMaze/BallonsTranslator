class LLMUserActionRequiredError(Exception):
    """Mark an LLM failure that automatic retries cannot resolve.

    Retry owners catch this base once, while UI boundaries may still provide
    specialized handling for individual subclasses.

    >>> issubclass(LLMApiKeyRequiredError, LLMUserActionRequiredError)
    True
    """


class LLMApiKeyRequiredError(LLMUserActionRequiredError):
    """Raised when an LLM profile requires a key before a request can run.

    Example:
        >>> err = LLMApiKeyRequiredError('profile-1', 'Profile 1')
        >>> err.profile_name
        'Profile 1'
    """

    def __init__(self, profile_id: str, profile_name: str = ''):
        self.profile_id = profile_id
        self.profile_name = profile_name or profile_id
        super().__init__(f'API key is required for LLM profile "{self.profile_name}".')


class LLMModelRequiredError(LLMUserActionRequiredError):
    """Raised when an LLM profile is enabled but has no request model.

    Example:
        >>> err = LLMModelRequiredError('profile-1', 'Profile 1', target='vision_model')
        >>> err.is_vision
        True
    """

    def __init__(
        self,
        profile_id: str,
        profile_name: str = '',
        vision: bool = False,
        target: str = 'model',
    ):
        self.profile_id = profile_id
        self.profile_name = profile_name or profile_id
        if vision and target == 'model':
            target = 'vision_model'
        if target not in {'model', 'vision_model', 'image_model'}:
            target = 'model'
        self.target = target
        self.is_vision = target == 'vision_model'
        self.is_image = target == 'image_model'
        model_label = {
            'model': 'model',
            'vision_model': 'vision model',
            'image_model': 'image model',
        }[target]
        super().__init__(f'{model_label.capitalize()} is required for LLM profile "{self.profile_name}".')


class LLMBaseURLRequiredError(LLMUserActionRequiredError):
    """Raised when an LLM profile needs a request URL before a task can run.

    Example:
        >>> err = LLMBaseURLRequiredError('profile-1', 'Profile 1', target='image_base_url')
        >>> err.target
        'image_base_url'
    """

    def __init__(self, profile_id: str, profile_name: str = '', target: str = 'base_url'):
        self.profile_id = profile_id
        self.profile_name = profile_name or profile_id
        if target not in {'base_url', 'image_base_url'}:
            target = 'base_url'
        self.target = target
        url_label = {
            'base_url': 'base URL',
            'image_base_url': 'image base URL',
        }[target]
        super().__init__(f'{url_label.capitalize()} is required for LLM profile "{self.profile_name}".')


class LLMOutputLimitError(LLMUserActionRequiredError):
    """Raised when a chat completion explicitly stops at its output limit.

    >>> error = LLMOutputLimitError('profile-1', 'Profile 1', 8192, 'low')
    >>> 'Max Tokens: 8192' in str(error)
    True
    """

    def __init__(
        self,
        profile_id: str,
        profile_name: str,
        max_tokens: int,
        thinking_level: str,
    ) -> None:
        self.profile_id = profile_id
        self.profile_name = profile_name or profile_id
        self.max_tokens = int(max_tokens)
        self.thinking_level = thinking_level
        super().__init__(
            f'LLM output limit reached for profile "{self.profile_name}" '
            f'(Max Tokens: {self.max_tokens}; '
            f'Thinking Level: {self.thinking_level}). '
            f'Please increase Max Tokens or lower or disable Thinking Level, '
            f'then retry.'
        )


class LLMMemoryCompactionError(LLMUserActionRequiredError):
    """Raised after a memory-compaction request exhausts ordinary retries.

    >>> error = LLMMemoryCompactionError(
    ...     'profile-1', 'Profile 1', 3, 'bad JSON')
    >>> 'after 3 attempts' in str(error)
    True
    """

    def __init__(
        self,
        profile_id: str,
        profile_name: str,
        attempts: int,
        reason: str,
    ) -> None:
        self.profile_id = profile_id
        self.profile_name = profile_name or profile_id
        self.attempts = int(attempts)
        attempt_label = 'attempt' if self.attempts == 1 else 'attempts'
        super().__init__(
            f'LLM memory compaction failed for profile "{self.profile_name}" '
            f'after {self.attempts} {attempt_label}: {reason}'
        )


class LLMRequestStopped(Exception):
    """Raised when an in-flight LLM request loop is stopped cooperatively.

    Example:
        >>> issubclass(LLMRequestStopped, Exception)
        True
    """

    pass


class ModuleRunError(RuntimeError):
    """Raised when a runtime module cannot complete its current task.

    The exception marks a stage failure after a module was selected and prepared,
    so callers can distinguish invalid empty results from an execution failure.

    Example:
        >>> err = ModuleRunError('ocr', 'demo', 'backend unavailable')
        >>> str(err)
        'ocr module demo failed to run: backend unavailable'
    """

    def __init__(self, module_key: str, module_name: str = '', message: str = ''):
        self.module_key = module_key
        self.module_name = module_name

        detail = f'{module_key} module'
        if module_name:
            detail += f' {module_name}'
        detail += ' failed to run'
        if message:
            detail += f': {message}'

        super().__init__(detail)
