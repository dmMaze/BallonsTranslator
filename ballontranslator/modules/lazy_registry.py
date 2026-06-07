import ast
import importlib.metadata
import os
import platform
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ballontranslator.utils.registry import ModuleSpec

from .base import MODULE_ROOT, MODULE_SCRIPTS


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

OPTIONAL_DEPENDENCY_OVERRIDES = {
    # Keep optional dependency checks explicit when imports do not tell the full story.
    ('inpainter', 'opencv-tela'): [],
    ('inpainter', 'patchmatch'): [],
    ('inpainter', 'aot'): ['torch'],
    ('inpainter', 'lama_mpe'): ['torch'],
    ('inpainter', 'lama_large_512px'): ['torch'],
    ('inpainter', 'flux2-klein'): ['torch', 'diffusers', 'safetensors', 'transformers', 'gguf'],
    ('textdetector', 'ctd'): ['torch', 'torchvision'],
    ('textdetector', 'ysgyolo'): ['torch', 'ultralytics'],
    ('ocr', 'mit32px'): ['torch'],
    ('ocr', 'mit48px_ctc'): ['torch'],
    ('ocr', 'mit48px'): ['torch'],
}

INITIALIZED_REGISTRIES = set()


def _package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_package_backend():
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


class SafeEval:
    # Small evaluator for literal module metadata; unknown expressions are ignored.
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
            return UNKNOWN
        return getattr(value, node.attr, UNKNOWN)

    def visit_Call(self, node):
        func_name = _call_name(node.func)
        args = [self.visit(arg) for arg in node.args]
        if any(arg is UNKNOWN for arg in args):
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
        if func_name == 'platform.system':
            return platform.system()
        if func_name == 'platform.mac_ver':
            return platform.mac_ver()
        if func_name == 'platform.version':
            return platform.version()
        if func_name in {'os.path.join', 'osp.join'}:
            return os.path.join(*args)
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
    attrs = {}
    class_env = env.copy()

    def walk(stmts):
        evaluator = SafeEval(class_env)
        for node in stmts:
            name, value_node = _assign_name(node)
            if name is not None and value_node is not None:
                value = evaluator.eval(value_node)
                if value is not UNKNOWN:
                    class_env[name] = value
                    if name in {'params', 'download_file_list', 'download_file_on_load'}:
                        attrs[name] = value
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
    return attrs


def _return_list(func_node: ast.FunctionDef, env: Dict[str, Any]):
    evaluator = SafeEval(env)
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return) and node.value is not None:
            value = evaluator.eval(node.value)
            if isinstance(value, list):
                return value
    return None


def _collect_translator_langs(class_node: ast.ClassDef, env: Dict[str, Any]):
    langs = []
    src = tgt = None
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
                    if not isinstance(child, ast.Assign) or len(child.targets) != 1:
                        continue
                    target = child.targets[0]
                    if not isinstance(target, ast.Subscript):
                        continue
                    if not isinstance(target.value, ast.Attribute):
                        continue
                    if target.value.attr != 'lang_map':
                        continue
                    key = evaluator.eval(target.slice)
                    value = evaluator.eval(child.value)
                    if isinstance(key, str) and value not in {'', None, UNKNOWN} and key not in langs:
                        langs.append(key)

    if class_node.name in {'TransNone', 'TransSource'}:
        langs = BASE_TRANSLATOR_LANGS.copy()
    if cht_require_convert and '简体中文' in langs and '繁體中文' not in langs:
        langs.append('繁體中文')
    if src is None:
        src = langs or None
    if tgt is None:
        tgt = langs or None
    return src, tgt


def _collect_optional_imports(tree: ast.AST):
    names = []
    optional_roots = {
        'torch', 'torchvision', 'transformers', 'diffusers', 'safetensors',
        'ultralytics', 'paddleocr', 'paddle', 'ctranslate2', 'sentencepiece',
        'msl', 'winsdk', 'Vision', 'objc', 'openai', 'deepl', 'translators',
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split('.')[0]
                if root in optional_roots and root not in names:
                    names.append(root)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split('.')[0]
            if root in optional_roots and root not in names:
                names.append(root)
    return names


def _scan_file(path: str, module_type: str) -> List[ModuleSpec]:
    # Scan decorators/classes without executing top-level imports or model code.
    with open(path, 'r', encoding='utf8') as f:
        source = f.read()
    tree = ast.parse(source, filename=path)
    module_path = _module_name_from_path(path)
    optional_imports = _collect_optional_imports(tree)
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
                if module_type == 'translator':
                    src, tgt = _collect_translator_langs(node, env)
                optional_dependencies = OPTIONAL_DEPENDENCY_OVERRIDES.get(
                    (module_type, key),
                    optional_imports,
                )
                specs.append(ModuleSpec(
                    key=key,
                    import_path=module_path,
                    class_name=node.name,
                    module_type=module_type,
                    params=attrs.get('params'),
                    download_file_list=attrs.get('download_file_list'),
                    download_file_on_load=attrs.get('download_file_on_load', False),
                    optional_dependencies=optional_dependencies,
                    supported_src_list=src,
                    supported_tgt_list=tgt,
                ))
            elif isinstance(node, ast.If):
                cond = evaluator.eval(node.test)
                if cond is True:
                    walk(node.body)
                elif cond is False:
                    walk(node.orelse)
                else:
                    walk(node.body)
                    walk(node.orelse)
            elif isinstance(node, ast.Try):
                walk(node.body)

    walk(tree.body)
    return specs


def init_lazy_module_registries(target_modules=None):
    from . import MODULETYPE_TO_REGISTRIES

    def _module_files(module_type: str) -> List[str]:
        script = MODULE_SCRIPTS[module_type]
        module_dir = script['module_dir']
        pattern = re.compile(script['module_pattern'])
        files = []
        if os.path.isdir(module_dir):
            for name in sorted(os.listdir(module_dir)):
                if pattern.match(name):
                    files.append(os.path.join(module_dir, name))
        files.extend(EXTRA_MODULE_FILES.get(module_type, []))
        return [path for path in files if os.path.exists(path)]


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
            for spec in _scan_file(path, module_type):
                registry.register_lazy_module(spec)
        # Registry groups are idempotent; re-scanning could overwrite live classes.
        INITIALIZED_REGISTRIES.add(module_type)
