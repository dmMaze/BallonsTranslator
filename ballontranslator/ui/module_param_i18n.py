from typing import Mapping, Tuple

ModuleParamKey = Tuple[str, str, str, str]

try:
    from ._generated.module_param_dialog_i18n_catalog import MODULE_PARAM_CATALOG
except Exception:
    MODULE_PARAM_CATALOG = {}


def _translate_catalog_entry(key: ModuleParamKey, current_source: str) -> str:
    entry = MODULE_PARAM_CATALOG.get(key, {})
    if not isinstance(entry, Mapping) or entry.get('source') != current_source:
        return current_source
    translate = entry.get('translate')
    translated = translate() if callable(translate) else ''
    return translated if isinstance(translated, str) and translated else current_source


def tr_module_description(params: dict, module_type: str = '', module_key: str = ''):
    """Return the translated module description when the catalog source matches.

    Example:
        >>> tr_module_description({'description': 'Demo module.'})
        'Demo module.'
    """

    source = ''
    if isinstance(params, dict) and isinstance(params.get('description'), str):
        source = params['description']
    if module_type and module_key:
        return _translate_catalog_entry(
            (module_type, module_key, '', 'description'),
            source,
        )
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
    if module_type and module_key:
        return _translate_catalog_entry(
            (module_type, module_key, param_key, 'display_name'),
            source,
        )
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
    if module_type and module_key:
        return _translate_catalog_entry(
            (module_type, module_key, param_key, 'description'),
            source,
        )
    return source
