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
        >>> err = LLMModelRequiredError('profile-1', 'Profile 1', vision=True)
        >>> err.is_vision
        True
    """

    def __init__(self, profile_id: str, profile_name: str = '', vision: bool = False):
        self.profile_id = profile_id
        self.profile_name = profile_name or profile_id
        self.is_vision = bool(vision)
        model_label = 'vision model' if self.is_vision else 'model'
        super().__init__(f'{model_label.capitalize()} is required for LLM profile "{self.profile_name}".')


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
