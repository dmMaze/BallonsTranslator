from typing import Mapping, Tuple

from qtpy.QtCore import QObject


ModuleParamKey = Tuple[str, str, str, str]
_module_param_translator = None


def register_module_param_translator(translator):
    """Register the UI-owned translator used by render helpers.

    Example:
        >>> register_module_param_translator(None)
        >>> _module_param_translator is None
        True
    """

    global _module_param_translator
    _module_param_translator = translator


def tr_module_description(params: dict, module_type: str = '', module_key: str = ''):
    """Return the translated module description when the catalog source matches.

    Example:
        >>> tr_module_description({'description': 'Demo module.'})
        'Demo module.'
    """

    source = ''
    if isinstance(params, dict) and isinstance(params.get('description'), str):
        source = params['description']
    translator = _module_param_translator
    if module_type and module_key and translator is not None:
        return translator.translate_text(module_type, module_key, '', 'description', source)
    return source


def tr_param_display_name(
    params: dict,
    param_key: str,
    param_dict: dict = None,
    module_type: str = '',
    module_key: str = '',
):
    """Return the translated parameter label, falling back to display_name or key.

    Example:
        >>> tr_param_display_name({}, 'delay', {'display_name': 'Delay'})
        'Delay'
        >>> tr_param_display_name({}, 'api_url', '')
        'api_url'
    """

    param = param_dict
    if param is None and isinstance(params, dict):
        param = params.get(param_key)
    if isinstance(param, dict) and isinstance(param.get('display_name'), str) and param['display_name']:
        source = param['display_name']
    else:
        source = param_key
    translator = _module_param_translator
    if module_type and module_key and translator is not None:
        return translator.translate_text(module_type, module_key, param_key, 'display_name', source)
    return source


def tr_param_description(
    params: dict,
    param_key: str,
    param_dict: dict = None,
    module_type: str = '',
    module_key: str = '',
):
    """Return the translated parameter description when one is defined.

    Example:
        >>> tr_param_description({}, 'delay', {'description': 'Wait.'})
        'Wait.'
        >>> tr_param_description({}, 'delay', '')
        ''
    """

    param = param_dict
    if param is None and isinstance(params, dict):
        param = params.get(param_key)
    source = ''
    if isinstance(param, dict) and isinstance(param.get('description'), str):
        source = param['description']
    translator = _module_param_translator
    if module_type and module_key and translator is not None:
        return translator.translate_text(module_type, module_key, param_key, 'description', source)
    return source


class ModuleParamTranslator(QObject):
    """Resolve generated module parameter translations without mutating params.

    Example:
        >>> tr = ModuleParamTranslator()
        >>> isinstance(tr.sources, dict)
        True
    """

    def __init__(self, parent: QObject = None):
        super().__init__(parent)
        try:
            from ._generated.module_param_i18n_catalog import MODULE_PARAM_CATALOG
        except Exception:
            MODULE_PARAM_CATALOG = {}

        self.sources = {}
        for key, entry in MODULE_PARAM_CATALOG.items():
            if not isinstance(entry, Mapping):
                continue
            source = entry.get('source')
            if isinstance(source, str):
                self.sources[key] = source

        def translate_from_catalog(module_type, module_key, param_key, field):
            entry = MODULE_PARAM_CATALOG.get((module_type, module_key, param_key, field), {})
            if not isinstance(entry, Mapping):
                return ''
            translator = entry.get('translate')
            if not callable(translator):
                return ''
            translated = translator()
            return translated if isinstance(translated, str) else ''

        self._translate = translate_from_catalog

    def translate_text(
        self,
        module_type: str,
        module_key: str,
        param_key: str,
        field: str,
        current_source: str,
    ) -> str:
        """Translate one catalog entry if its source still matches metadata.

        Example:
            >>> key = ('translator', 'demo', 'delay', 'display_name')
            >>> tr = ModuleParamTranslator()
            >>> tr.sources = {key: 'Delay'}
            >>> tr._translate = lambda *args: 'Translated Delay'
            >>> tr.translate_text('translator', 'demo', 'delay', 'display_name', 'Changed')
            'Changed'
        """

        if not current_source:
            return ''
        expected_source = self.sources.get((module_type, module_key, param_key, field))
        if expected_source != current_source:
            return current_source
        return self._translate_key(module_type, module_key, param_key, field, current_source)

    def _translate_key(self, module_type: str, module_key: str, param_key: str, field: str, fallback: str) -> str:
        translated = self._translate(module_type, module_key, param_key, field)
        if isinstance(translated, str) and translated:
            return translated
        return fallback
