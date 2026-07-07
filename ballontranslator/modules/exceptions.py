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
