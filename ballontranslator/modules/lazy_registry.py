import ast
import importlib.metadata
import os
import platform
import re
import sys
import tempfile
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ballontranslator.utils.registry import ModuleSpec
from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.utils.torch_install_helper import detect_nvidia_gpus

from .base import CUSTOM_MODULE_ROOT, MODULE_ROOT, MODULE_SCRIPTS, torch_available


UNKNOWN = object()

# This file builds registry metadata from AST so startup never imports module code.

DECORATORS = {
    'translator': {'register_translator'},
    'textdetector': {'register_textdetectors'},
    'inpainter': {'register_inpainter'},
    'ocr': {'register_OCR'},
}

EXTRA_MODULE_FILES = {
    'translator': [str(MODULE_ROOT / 'translators' / 'base.py')],
}
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

BASE_TRANSLATOR_LANGS = [
    'Auto',
    '简体中文',
    '繁體中文',
    '日本語',
    'English',
    '한국어',
    'Tiếng Việt',
    'čeština',
    'Nederlands',
    'Français',
    'Deutsch',
    'magyar nyelv',
    'Italiano',
    'Polski',
    'Português',
    'Brazilian Portuguese',
    'limba română',
    'русский язык',
    'Español',
    'Türk dili',
    'украї́нська мо́ва',
    'Thai',
    'Arabic',
    'Hindi',
    'Malayalam',
    'Tamil',
]

INITIALIZED_REGISTRIES = set()


def _package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_package_backend():
    if _nvidia_cuda_available():
        return 'cuda'
    version = _package_version('torch')
    if version is None:
        return None
    if sys.platform == 'darwin':
        return 'mps'
    if '+' not in version:
        return None
    local_version = version.split('+', 1)[1].lower()
    if local_version.startswith(('cu', 'rocm')):
        return 'cuda'
    if local_version.startswith('xpu'):
        return 'xpu'
    return None


def probe_torch_package() -> Tuple[Optional[str], Optional[str]]:
    """Return installed Torch version and inferred device without importing Torch.

    >>> version, device = probe_torch_package()
    >>> (version is None) == (device is None)
    True
    """

    if not torch_available():
        return None, None
    version = _package_version('torch')
    if version is None:
        return None, None
    return version, _torch_package_backend() or 'cpu'


@lru_cache(maxsize=1)
def _nvidia_cuda_available() -> bool:
    """Return whether the driver reports NVIDIA GPU availability.

    >>> isinstance(_nvidia_cuda_available(), bool)
    True
    """

    if sys.platform not in {'win32', 'linux'}:
        return False
    return bool(detect_nvidia_gpus())


def _candidate_device_options():
    options = ['cpu']
    backend = _torch_package_backend()
    if backend is not None:
        options.append(backend)
    return options


def _preferred_device_value(options):
    preferred = ['mps'] if sys.platform == 'darwin' else ['cuda', 'xpu']
    for device in preferred:
        if device in options:
            return device
    if 'cpu' in options:
        return 'cpu'
    return options[0] if options else 'cpu'


def _device_selector(not_supported=None):
    if not_supported is None:
        not_supported = []
    options = _candidate_device_options()
    options = [opt for opt in options if all(device not in opt for device in not_supported)]
    return {
        'type': 'selector',
        'options': options,
        'value': _preferred_device_value(options),
        '__device_not_supported': not_supported,
    }


def _find_model_paths(model_dir, prefixes):
    """Return local model paths for selector metadata without importing modules.

    Example:
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _ = Path(tmp, 'ysgyolo_demo.pt').write_text('')
        ...     _ = Path(tmp, 'other.pt').write_text('')
        ...     _find_model_paths(tmp, ('ysgyolo',))  # doctest: +ELLIPSIS
        ['...ysgyolo_demo.pt', 'data/models/ysgyolo_yolo26_2.0.pt', 'data/models/ysgyolo_yolo26OBB_2.0.pt']
    """

    default_path_list = [
        'data/models/ysgyolo_yolo26_2.0.pt',
        'data/models/ysgyolo_yolo26OBB_2.0.pt'
    ]

    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    try:
        names = sorted(os.listdir(model_dir))
    except OSError:
        return []
    found_list = [
        os.path.join(model_dir, name).replace('\\', '/')
        for name in names
        if name.startswith(tuple(prefixes))
    ]

    for p in default_path_list:
        # Built-in download targets must obey the same prefix filter as local
        # files; SafeEval can encounter this helper with non-YSG prefixes.
        if not os.path.basename(p).startswith(tuple(prefixes)):
            continue
        if p not in found_list:
            found_list.append(p)

    return found_list


class SafeEval:
    """Evaluate the small literal subset needed for lazy module metadata.

    Unknown or unsafe expressions return ``UNKNOWN`` so the lazy scanner can
    keep building specs without importing the module being scanned.

    Example:
        >>> expr = ast.parse('base + 3', mode='eval').body
        >>> SafeEval({'base': 2}).eval(expr)
        5
        >>> unknown = ast.parse('load_model()', mode='eval').body
        >>> SafeEval({}).eval(unknown) is UNKNOWN
        True
        >>> expr = ast.parse('list(lang_map.keys())', mode='eval').body
        >>> SafeEval({'lang_map': {'Japanese': 'ja'}}).eval(expr)
        ['Japanese']
        >>> expr = ast.parse('lang_map.items()', mode='eval').body
        >>> SafeEval({'lang_map': {'Japanese': 'ja'}}).eval(expr)
        [('Japanese', 'ja')]
    """

    def __init__(self, env: Dict[str, Any]):
        self.env = env

    def eval(self, node):
        try:
            return self.visit(node)
        except Exception:
            return UNKNOWN

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, None)
        if visitor is None:
            return UNKNOWN
        return visitor(node)

    def visit_Constant(self, node):
        return node.value

    def visit_Name(self, node):
        if node.id in self.env:
            return self.env[node.id]
        if node.id == 'None':
            return None
        return UNKNOWN

    def visit_List(self, node):
        values = [self.visit(v) for v in node.elts]
        return UNKNOWN if any(v is UNKNOWN for v in values) else values

    def visit_Tuple(self, node):
        values = [self.visit(v) for v in node.elts]
        return UNKNOWN if any(v is UNKNOWN for v in values) else tuple(values)

    def visit_Set(self, node):
        values = [self.visit(v) for v in node.elts]
        return UNKNOWN if any(v is UNKNOWN for v in values) else set(values)

    def visit_Dict(self, node):
        out = {}
        for key_node, value_node in zip(node.keys, node.values):
            value = self.visit(value_node)
            if value is UNKNOWN:
                return UNKNOWN
            if key_node is None:
                if not isinstance(value, dict):
                    return UNKNOWN
                out.update(value)
                continue
            key = self.visit(key_node)
            if key is UNKNOWN:
                return UNKNOWN
            out[key] = value
        return out

    def visit_UnaryOp(self, node):
        value = self.visit(node.operand)
        if value is UNKNOWN:
            return UNKNOWN
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.Not):
            return not value
        return UNKNOWN

    def visit_BoolOp(self, node):
        values = [self.visit(v) for v in node.values]
        if isinstance(node.op, ast.And):
            for value in values:
                if value is False:
                    return False
                if value is UNKNOWN:
                    return UNKNOWN
            return True
        if isinstance(node.op, ast.Or):
            for value in values:
                if value is True:
                    return True
                if value is UNKNOWN:
                    return UNKNOWN
            return False
        return UNKNOWN

    def visit_Compare(self, node):
        left = self.visit(node.left)
        if left is UNKNOWN:
            return UNKNOWN
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if right is UNKNOWN:
                return UNKNOWN
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.In):
                ok = left in right
            elif isinstance(op, ast.NotIn):
                ok = left not in right
            elif isinstance(op, ast.Is):
                ok = left is right
            elif isinstance(op, ast.IsNot):
                ok = left is not right
            else:
                return UNKNOWN
            if not ok:
                return False
            left = right
        return True

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if left is UNKNOWN or right is UNKNOWN:
            return UNKNOWN
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Mod):
            return left % right
        return UNKNOWN

    def visit_IfExp(self, node):
        test = self.visit(node.test)
        if test is UNKNOWN:
            return self.visit(node.orelse)
        return self.visit(node.body if test else node.orelse)

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        if value is UNKNOWN:
            return UNKNOWN
        index = self.visit(node.slice)
        if index is UNKNOWN:
            return UNKNOWN
        try:
            return value[index]
        except Exception:
            return UNKNOWN

    def visit_Slice(self, node):
        lower = None if node.lower is None else self.visit(node.lower)
        upper = None if node.upper is None else self.visit(node.upper)
        step = None if node.step is None else self.visit(node.step)
        if lower is UNKNOWN or upper is UNKNOWN or step is UNKNOWN:
            return UNKNOWN
        return slice(lower, upper, step)

    def visit_Attribute(self, node):
        value = self.visit(node.value)
        if value is UNKNOWN:
            if isinstance(node.value, ast.Name):
                root = node.value.id
                if root == 'sys' and node.attr == 'platform':
                    return sys.platform
                if root == 'shared':
                    if node.attr == 'ON_WINDOWS':
                        return sys.platform == 'win32'
                    if node.attr == 'ON_MACOS':
                        return sys.platform == 'darwin'
                    if node.attr == 'ON_LINUX':
                        return sys.platform.startswith('linux')
                    if node.attr == 'ON_APPLE_SILICON':
                        from ballontranslator.utils.shared import ON_APPLE_SILICON
                        return ON_APPLE_SILICON
            return UNKNOWN
        return getattr(value, node.attr, UNKNOWN)

    def visit_Call(self, node):
        func_name = _call_name(node.func)
        args = [self.visit(arg) for arg in node.args]
        if any(arg is UNKNOWN for arg in args):
            return UNKNOWN

        if func_name == 'platform.system':
            return platform.system()
        if func_name == 'platform.mac_ver':
            return platform.mac_ver()
        if func_name == 'platform.version':
            return platform.version()
        if func_name == 'platform.machine':
            return platform.machine()

        if isinstance(node.func, ast.Attribute) and not args and not node.keywords:
            value = self.visit(node.func.value)
            if value is UNKNOWN:
                return UNKNOWN
            if isinstance(value, dict):
                if node.func.attr == 'keys':
                    return list(value.keys())
                if node.func.attr == 'values':
                    return list(value.values())
                if node.func.attr == 'items':
                    return list(value.items())
            if isinstance(value, str):
                if node.func.attr == 'lower':
                    return value.lower()
                if node.func.attr == 'upper':
                    return value.upper()
                if node.func.attr == 'strip':
                    return value.strip()
            return UNKNOWN

        if func_name == 'DEVICE_SELECTOR':
            not_supported = args[0] if args else []
            for kw in node.keywords:
                if kw.arg == 'not_supported':
                    not_supported = self.visit(kw.value)
                    if not_supported is UNKNOWN:
                        not_supported = []
            return _device_selector(not_supported)
        if func_name in {'deepcopy', 'copy.deepcopy'} and len(args) == 1:
            return deepcopy(args[0])
        if func_name == 'list' and len(args) == 1:
            return list(args[0])
        if func_name == 'tuple' and len(args) == 1:
            return tuple(args[0])
        if func_name == 'set' and len(args) == 1:
            return set(args[0])
        if func_name == 'str' and len(args) == 1:
            return str(args[0])
        if func_name == 'int' and len(args) == 1:
            return int(args[0])
        if func_name == 'float' and len(args) == 1:
            return float(args[0])
        if func_name in {'os.path.join', 'osp.join'}:
            return os.path.join(*args)
        if func_name == 'find_model_paths' and len(args) == 2:
            return _find_model_paths(*args)
        return UNKNOWN


def _call_name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f'{parent}.{node.attr}' if parent else node.attr
    return ''


def _module_name_from_path(path: str) -> str:
    path_obj = Path(path).resolve()
    try:
        rel_path = path_obj.relative_to(PACKAGE_ROOT)
        return 'ballontranslator.' + '.'.join(rel_path.with_suffix('').parts)
    except ValueError:
        try:
            rel_path = path_obj.relative_to(CUSTOM_MODULE_ROOT.resolve())
            return 'custom_modules.' + '.'.join(rel_path.with_suffix('').parts)
        except ValueError:
            module_name = path.replace(os.sep, '.').replace('/', '.')
            if module_name.endswith('.py'):
                module_name = module_name[:-3]
            return module_name


def _decorator_key(node, module_type: str, env: Dict[str, Any]) -> Optional[str]:
    if not isinstance(node, ast.Call):
        return None
    if _call_name(node.func) not in DECORATORS[module_type]:
        return None
    if len(node.args) == 0:
        return None
    value = SafeEval(env).eval(node.args[0])
    return value if isinstance(value, str) else None


def _assign_name(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id, node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id, node.value
    return None, None


def _walk_assignments(stmts: Iterable[ast.stmt], env: Dict[str, Any]):
    evaluator = SafeEval(env)
    for node in stmts:
        name, value_node = _assign_name(node)
        if name is not None and value_node is not None:
            value = evaluator.eval(value_node)
            if value is not UNKNOWN:
                env[name] = value


def _collect_class_attrs(class_node: ast.ClassDef, env: Dict[str, Any]) -> Dict[str, Any]:
    """Collect class-level metadata without executing the class body.

    Example:
        >>> node = ast.parse('class Demo:\\n    params = {"device": "cpu"}\\n').body[0]
        >>> _collect_class_attrs(node, {})['params']
        {'device': 'cpu'}
    """

    attrs = {}
    warnings = []
    class_env = env.copy()

    def walk(stmts):
        evaluator = SafeEval(class_env)
        for node in stmts:
            name, value_node = _assign_name(node)
            if name is not None and value_node is not None:
                value = evaluator.eval(value_node)
                if value is not UNKNOWN:
                    class_env[name] = value
                    if name in {'params', 'download_file_list', 'download_file_on_load', 'dependencies'}:
                        attrs[name] = value
                elif name in {'params', 'download_file_list', 'download_file_on_load', 'dependencies'}:
                    warnings.append(f'{class_node.name}.{name} could not be evaluated lazily')
            elif isinstance(node, ast.If):
                cond = evaluator.eval(node.test)
                if cond is True:
                    walk(node.body)
                elif cond is False:
                    walk(node.orelse)
                else:
                    walk(node.body)
                    walk(node.orelse)
    walk(class_node.body)
    if warnings:
        attrs['__metadata_warnings'] = warnings
    return attrs


def _return_list(func_node: ast.FunctionDef, env: Dict[str, Any]):
    evaluator = SafeEval(env)
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return) and node.value is not None:
            value = evaluator.eval(node.value)
            if isinstance(value, list):
                return value
    return None


def _is_self_lang_map(node):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == 'lang_map'
        and isinstance(node.value, ast.Name)
        and node.value.id == 'self'
    )


def _lang_map_subscript_key(target, evaluator):
    if not isinstance(target, ast.Subscript):
        return UNKNOWN
    if not _is_self_lang_map(target.value):
        return UNKNOWN
    return evaluator.eval(target.slice)


def _is_self_lang_map_assign(target):
    return _is_self_lang_map(target)


def _append_lang_if_supported(langs: List[str], key, value) -> bool:
    if isinstance(key, str) and value not in {'', None, UNKNOWN}:
        if key not in langs:
            langs.append(key)
        return True
    return False


def _collect_translator_langs(class_node: ast.ClassDef, env: Dict[str, Any]) -> Tuple[Optional[List[str]], Optional[List[str]], List[str]]:
    """Infer translator language lists from simple class metadata and setup code.

    Example:
        >>> source = '''
        ... class Demo:
        ...     cht_require_convert = True
        ...     def _setup_translator(self):
        ...         self.lang_map["简体中文"] = "zh"
        ...         self.lang_map.update({"English": "en"})
        ... '''
        >>> node = ast.parse(source).body[0]
        >>> _collect_translator_langs(node, {})[1]
        ['简体中文', 'English', '繁體中文']
    """

    langs = []
    src = tgt = None
    warnings = []
    cht_require_convert = False
    evaluator = SafeEval(env)

    for node in class_node.body:
        name, value_node = _assign_name(node)
        if name == 'cht_require_convert' and value_node is not None:
            value = evaluator.eval(value_node)
            if isinstance(value, bool):
                cht_require_convert = value

        if isinstance(node, ast.FunctionDef):
            if node.name in {'supported_src_list', 'supported_tgt_list'}:
                value = _return_list(node, env)
                if node.name == 'supported_src_list':
                    src = value
                else:
                    tgt = value
            if node.name == '_setup_translator':
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign) and len(child.targets) == 1:
                        target = child.targets[0]
                        if _is_self_lang_map_assign(target):
                            mapping = evaluator.eval(child.value)
                            if not isinstance(mapping, dict):
                                warnings.append(f'{class_node.name} has unsupported lazy lang_map assignment')
                                continue
                            supported_items = [
                                (key, value)
                                for key, value in mapping.items()
                                if _append_lang_if_supported(langs, key, value)
                            ]
                            if len(supported_items) != len(mapping):
                                warnings.append(f'{class_node.name} has unsupported lazy lang_map values')
                            continue
                        key = _lang_map_subscript_key(target, evaluator)
                        if key is UNKNOWN:
                            continue
                        value = evaluator.eval(child.value)
                        if not _append_lang_if_supported(langs, key, value):
                            warnings.append(f'{class_node.name} has unsupported lazy lang_map assignment')
                    elif isinstance(child, ast.Expr) and isinstance(child.value, ast.Call):
                        call = child.value
                        if not isinstance(call.func, ast.Attribute) or call.func.attr != 'update':
                            continue
                        if not _is_self_lang_map(call.func.value):
                            continue
                        if len(call.args) != 1 or call.keywords:
                            warnings.append(f'{class_node.name} has unsupported lazy lang_map.update call')
                            continue
                        mapping = evaluator.eval(call.args[0])
                        if not isinstance(mapping, dict):
                            warnings.append(f'{class_node.name} has unsupported lazy lang_map.update call')
                            continue
                        supported_items = [
                            (key, value)
                            for key, value in mapping.items()
                            if _append_lang_if_supported(langs, key, value)
                        ]
                        if len(supported_items) != len(mapping):
                            warnings.append(f'{class_node.name} has unsupported lazy lang_map.update values')

    if class_node.name in {'TransNone', 'TransSource'}:
        langs = BASE_TRANSLATOR_LANGS.copy()
    if cht_require_convert and '简体中文' in langs and '繁體中文' not in langs:
        langs.append('繁體中文')
    if src is None:
        src = langs or None
    if tgt is None:
        tgt = langs or None
    if not langs and src is None and tgt is None:
        warnings.append(f'{class_node.name} has no lazy translator language metadata')
    return src, tgt, warnings


def validate_lazy_module_specs(specs: Iterable[ModuleSpec]) -> List[str]:
    """Return lazy metadata issues that would make selection-time UI stale.

    Example:
        >>> spec = ModuleSpec(key='demo', import_path='demo', class_name='Demo', module_type='translator')
        >>> validate_lazy_module_specs([spec])
        ['demo: translator has no lazy supported language metadata']
    """

    warnings = []
    for spec in specs:
        for warning in spec.metadata_warnings:
            warnings.append(f'{spec.key}: {warning}')
        if spec.module_type == 'translator':
            if not spec.supported_src_list or not spec.supported_tgt_list:
                warnings.append(f'{spec.key}: translator has no lazy supported language metadata')
        if not spec.params:
            continue
        for param_key, param in spec.params.items():
            if not isinstance(param, dict) or param.get('type') != 'selector':
                continue
            options = param.get('options')
            if options is UNKNOWN:
                warnings.append(f'{spec.key}.{param_key}: selector options could not be evaluated lazily')
                continue
            if options:
                continue
            has_editable_fallback = param.get('editable', False) and param.get('value') not in {'', None}
            if not has_editable_fallback:
                warnings.append(f'{spec.key}.{param_key}: selector has no lazy options or editable fallback')
    return warnings


def _scan_file(path: str, module_type: str, include_inactive_platform_branches: bool = False) -> List[ModuleSpec]:
    """Build lazy module specs from decorators and class attributes in one file.

    Example:
        >>> import tempfile
        >>> source = '''
        ... @register_translator("demo")
        ... class DemoTranslator:
        ...     params = {"device": DEVICE_SELECTOR(not_supported=["cuda", "xpu", "mps"])}
        ... '''
        >>> with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        ...     _ = f.write(source)
        ...     path = f.name
        >>> specs = _scan_file(path, 'translator')
        >>> os.unlink(path)
        >>> specs[0].key, specs[0].class_name
        ('demo', 'DemoTranslator')
        >>> specs[0].params['device']['value']
        'cpu'
        >>> source = '''
        ... @register_OCR("llm_style")
        ... class LLMStyleOCR:
        ...     lang_map = {"Japanese": "ja"}
        ...     popular_models = ["OAI: gpt-4o"]
        ...     params = {
        ...         "language": {"type": "selector", "options": list(lang_map.keys()), "value": "Japanese"},
        ...         "model": {"type": "selector", "options": popular_models + ["custom"], "value": "OAI: gpt-4o"},
        ...     }
        ... '''
        >>> with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        ...     _ = f.write(source)
        ...     path = f.name
        >>> specs = _scan_file(path, 'ocr')
        >>> os.unlink(path)
        >>> specs[0].params['language']['options']
        ['Japanese']
        >>> specs[0].params['model']['options']
        ['OAI: gpt-4o', 'custom']
    """

    # Scan decorators/classes without executing top-level imports or model code.
    with open(path, 'r', encoding='utf8') as f:
        source = f.read()
    tree = ast.parse(source, filename=path)
    module_path = _module_name_from_path(path)
    specs = []
    env = {
        'sys': sys,
        'platform': platform,
        'DEFAULT_DEVICE': 'cpu',
        'BF16_SUPPORTED': False,
        'True': True,
        'False': False,
        'None': None,
    }

    def is_platform_condition(node):
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name):
                    if child.value.id == 'sys' and child.attr == 'platform':
                        return True
                    if child.value.id == 'shared' and child.attr in {
                        'ON_WINDOWS', 'ON_MACOS', 'ON_LINUX', 'ON_APPLE_SILICON',
                    }:
                        return True
                if (
                    isinstance(child.value, ast.Name)
                    and child.value.id == 'platform'
                    and child.attr in {'system', 'mac_ver', 'version', 'machine'}
                ):
                    return True
        return False

    def walk(stmts):
        _walk_assignments(stmts, env)
        evaluator = SafeEval(env)
        for node in stmts:
            if isinstance(node, ast.ClassDef):
                key = None
                for decorator in node.decorator_list:
                    key = _decorator_key(decorator, module_type, env)
                    if key is not None:
                        break
                if key is None:
                    continue
                attrs = _collect_class_attrs(node, env)
                src = tgt = None
                metadata_warnings = attrs.get('__metadata_warnings', []).copy()
                if module_type == 'translator':
                    src, tgt, lang_warnings = _collect_translator_langs(node, env)
                    metadata_warnings.extend(lang_warnings)
                specs.append(ModuleSpec(
                    key=key,
                    import_path=module_path,
                    class_name=node.name,
                    module_type=module_type,
                    params=attrs.get('params'),
                    download_file_list=attrs.get('download_file_list'),
                    download_file_on_load=attrs.get('download_file_on_load', False),
                    dependencies=deepcopy(attrs.get('dependencies', [])),
                    supported_src_list=src,
                    supported_tgt_list=tgt,
                    metadata_warnings=metadata_warnings,
                ))
            elif isinstance(node, ast.If):
                cond = evaluator.eval(node.test)
                if cond is True:
                    walk(node.body)
                elif cond is False:
                    walk(node.orelse)
                    if include_inactive_platform_branches and is_platform_condition(node.test):
                        walk(node.body)
                else:
                    walk(node.body)
                    walk(node.orelse)
            elif isinstance(node, ast.Try):
                walk(node.body)

    walk(tree.body)
    return specs


def _module_files(module_type: str) -> List[str]:
    script = MODULE_SCRIPTS[module_type]
    pattern = re.compile(script['module_pattern'])
    files = []
    module_dir = script['module_dir']
    if os.path.isdir(module_dir):
        for name in sorted(os.listdir(module_dir)):
            if pattern.match(name):
                files.append(os.path.join(module_dir, name))
    files.extend(EXTRA_MODULE_FILES.get(module_type, []))
    if os.path.isdir(CUSTOM_MODULE_ROOT):
        for name in sorted(os.listdir(CUSTOM_MODULE_ROOT)):
            if pattern.match(name):
                files.append(os.path.join(CUSTOM_MODULE_ROOT, name))
    return [path for path in files if os.path.exists(path)]


def _is_custom_module_file(path: str) -> bool:
    try:
        Path(path).resolve().relative_to(CUSTOM_MODULE_ROOT.resolve())
        return True
    except ValueError:
        return False


def iter_lazy_module_specs(include_inactive_platform_branches: bool = False):
    """Yield module metadata without changing the active runtime registries.

    The translation catalog needs metadata from platform-specific modules too;
    those modules must remain absent from the runtime registry on unsupported
    platforms.

    Example:
        >>> list(iter_lazy_module_specs())  # doctest: +SKIP
        [...]
    """

    for module_type in sorted(MODULE_SCRIPTS):
        for path in _module_files(module_type):
            try:
                yield from _scan_file(path, module_type, include_inactive_platform_branches)
            except Exception as e:
                if not _is_custom_module_file(path):
                    raise
                LOGGER.warning(f'Failed to scan custom module {path}: {e}')


def init_lazy_module_registries(target_modules=None):
    """Register lightweight module specs while leaving real imports deferred.

    Example:
        >>> init_lazy_module_registries('ocr')  # doctest: +SKIP
        >>> OCR.get_spec('mit48px').resolved_class is None  # doctest: +SKIP
        True
    """

    from . import MODULETYPE_TO_REGISTRIES

    def _targets(target_modules=None):
        if target_modules is None:
            return list(MODULE_SCRIPTS.keys())
        if isinstance(target_modules, str):
            return [target_modules]
        return list(target_modules)

    for module_type in _targets(target_modules):
        if module_type in INITIALIZED_REGISTRIES:
            continue
        registry = MODULETYPE_TO_REGISTRIES[module_type]
        for path in _module_files(module_type):
            try:
                for spec in _scan_file(path, module_type):
                    registry.register_lazy_module(spec)
                    if _is_custom_module_file(path):
                        LOGGER.info(f'Discovered custom {module_type} module "{spec.key}" from {path}')
            except Exception as e:
                if not _is_custom_module_file(path):
                    raise
                LOGGER.warning(f'Failed to register custom module {path}: {e}')
        # Registry groups are idempotent; re-scanning could overwrite live classes.
        INITIALIZED_REGISTRIES.add(module_type)
