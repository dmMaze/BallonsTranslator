class LLMApiKeyRequiredError(Exception):
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


class LLMModelRequiredError(Exception):
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


class LLMBaseURLRequiredError(Exception):
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
