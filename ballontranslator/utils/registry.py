# modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py

import importlib
import inspect
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List


class LazyModuleError(Exception):
    pass


@dataclass
class ModuleSpec:
    """Static module metadata used by the UI before importing heavy modules.

    Example:
        >>> class DemoModule:
        ...     pass
        >>> spec = ModuleSpec(
        ...     key='demo',
        ...     import_path='unused',
        ...     class_name='DemoModule',
        ...     params={'device': {'value': 'cpu'}},
        ...     resolved_class=DemoModule,
        ... )
        >>> spec.resolve() is DemoModule
        True
        >>> copied = spec.params_copy()
        >>> copied == spec.params
        True
        >>> copied is spec.params
        False
    """

    key: str
    import_path: str
    class_name: str
    module_type: str = ''
    params: Dict = None
    download_file_list: List = None
    download_file_on_load: bool = False
    dependencies: List[str] = field(default_factory=list)
    supported_src_list: List[str] = None
    supported_tgt_list: List[str] = None
    available: bool = True
    availability_error: str = ''
    resolved_class: type = None
    metadata_warnings: List[str] = field(default_factory=list)

    def resolve(self):
        # Import the concrete module only after the user selects it.
        if self.resolved_class is not None:
            return self.resolved_class
        if not self.available:
            raise LazyModuleError(self.availability_error or f'{self.key} is not available on this platform.')
        try:
            module = importlib.import_module(self.import_path)
            self.resolved_class = getattr(module, self.class_name)
            setattr(self.resolved_class, '_module_spec', self)
            return self.resolved_class
        except ModuleNotFoundError as e:
            missing = e.name or str(e)
            raise LazyModuleError(
                f'Module "{self.key}" requires Python package "{missing}". '
                f'Install the dependency before selecting this module.'
            ) from e
        except Exception as e:
            raise LazyModuleError(f'Failed to import module "{self.key}" from {self.import_path}: {e}') from e

    def params_copy(self):
        return deepcopy(self.params)

    @property
    def name(self):
        return self.key

class Registry:
    """A registry to map strings to classes.

    Registered object could be built from registry.

    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODELS.build(dict(type='ResNet'))

    Please refer to
    https://mmcv.readthedocs.io/en/latest/understand_mmcv/registry.html for
    advanced usage.

    Args:
        name (str): Registry name.
        build_func(func, optional): Build function to construct instance from
            Registry, func:`build_from_cfg` is used if neither ``parent`` or
            ``build_func`` is specified. If ``parent`` is specified and
            ``build_func`` is not given,  ``build_func`` will be inherited
            from ``parent``. Default: None.
        parent (Registry, optional): Parent registry. The class registered in
            children registry could be built from parent. Default: None.
        scope (str, optional): The scope of registry. It is the key to search
            for children registry. If not specified, scope will be the name of
            the package where class is defined, e.g. mmdet, mmcls, mmseg.
            Default: None.
    """

    def __init__(self, name, build_func=None, parent=None, scope=None):
        self._name = name
        self._module_dict = dict()
        self._children = dict()
        
        # self._scope = self.infer_scope() if scope is None else scope

        # self.build_func will be set with the following priority:
        # 1. build_func
        # 2. parent.build_func
        # 3. build_from_cfg
        # if build_func is None:
        #     if parent is not None:
        #         self.build_func = parent.build_func
        #     else:
        #         self.build_func = build_from_cfg
        # else:
        #     self.build_func = build_func
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_children(self)
            self.parent = parent
        else:
            self.parent = None

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, ' \
                     f'items={self._module_dict})'
        return format_str

    @staticmethod
    def infer_scope():
        """Infer the scope of registry.

        The name of the package where registry is defined will be returned.

        Example:
            >>> # in mmdet/models/backbone/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            The scope of ``ResNet`` will be ``mmdet``.

        Returns:
            str: The inferred scope name.
        """
        # inspect.stack() trace where this function is called, the index-2
        # indicates the frame where `infer_scope()` is called
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__
        split_filename = filename.split('.')
        return split_filename[0]

    @staticmethod
    def split_scope_key(key):
        """Split scope and key.

        The first scope will be split from key.

        Examples:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'

        Return:
            tuple[str | None, str]: The former element is the first scope of
            the key, which can be ``None``. The latter is the remaining key.
        """
        split_index = key.find('.')
        if split_index != -1:
            return key[:split_index], key[split_index + 1:]
        else:
            return None, key

    @property
    def name(self):
        return self._name

    # @property
    # def scope(self):
    #     return self._scope

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    def get(self, key):
        """Get the registry record.

        Args:
            key (str): The class name in string format.

        Returns:
            class: The corresponding class.
        """
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            # get from self
            if real_key in self._module_dict:
                return self._module_dict[real_key]
        else:
            # get from self._children
            if scope in self._children:
                return self._children[scope].get(real_key)
            else:
                # goto root
                parent = self.parent
                while parent.parent is not None:
                    parent = parent.parent
                return parent.get(key)

    # def build(self, *args, **kwargs):
    #     return self.build_func(*args, **kwargs, registry=self)

    def _add_children(self, registry):
        """Add children for a registry.

        The ``registry`` will be added as children based on its scope.
        The parent registry could build objects from children registry.

        Example:
            >>> models = Registry('models')
            >>> mmdet_models = Registry('models', parent=models)
            >>> @mmdet_models.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet = models.build(dict(type='mmdet.ResNet'))
        """

        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert registry.scope not in self.children, \
            f'scope {registry.scope} exists in {self.name} registry'
        self.children[registry.scope] = registry

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, '
                            f'but got {type(module_class)}')

        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
            
        for name in module_name:
            existing = self._module_dict.get(name)
            if isinstance(existing, ModuleSpec):
                # A lazy spec can be replaced by the real class after import.
                existing.resolved_class = module_class
                setattr(module_class, '_module_spec', existing)
                self._module_dict[name] = module_class
                continue
            if not force and name in self._module_dict:
                raise KeyError(f'{name} is already registered '
                               f'in {self.name}')
            self._module_dict[name] = module_class

    def register_lazy_module(self, spec: ModuleSpec, force=False):
        if not isinstance(spec, ModuleSpec):
            raise TypeError(f'spec must be a ModuleSpec, but got {type(spec)}')
        existing = self._module_dict.get(spec.key)
        if not force and existing is not None:
            if isinstance(existing, ModuleSpec) and existing.import_path == spec.import_path and existing.class_name == spec.class_name:
                return spec
            if inspect.isclass(existing) and existing.__name__ == spec.class_name:
                spec.resolved_class = existing
                setattr(existing, '_module_spec', spec)
                return spec
            raise KeyError(f'{spec.key} is already registered in {self.name}')
        self._module_dict[spec.key] = spec
        return spec

    def resolve_module(self, key):
        module = self.get(key)
        if isinstance(module, ModuleSpec):
            # Keep the resolved class cached so later access is normal registry use.
            resolved = module.resolve()
            self._module_dict[key] = resolved
            return resolved
        return module

    def get_spec(self, key):
        module = self.get(key)
        if isinstance(module, ModuleSpec):
            return module
        if inspect.isclass(module):
            spec = getattr(module, '_module_spec', None)
            if isinstance(spec, ModuleSpec):
                spec.resolved_class = module
                return spec
            return ModuleSpec(
                key=key,
                import_path=module.__module__,
                class_name=module.__name__,
                params=deepcopy(getattr(module, 'params', None)),
                download_file_list=deepcopy(getattr(module, 'download_file_list', None)),
                download_file_on_load=getattr(module, 'download_file_on_load', False),
                dependencies=deepcopy(getattr(module, 'dependencies', [])),
                metadata_warnings=[],
                resolved_class=module,
            )
        return None

    def module_params(self, key):
        spec = self.get_spec(key)
        if spec is not None:
            return spec.params_copy()
        module = self.get(key)
        if module is None:
            return None
        return deepcopy(getattr(module, 'params', None))
        

    def deprecated_register_module(self, cls=None, force=False):
        warnings.warn(
            'The old API of register_module(module, force=False) '
            'is deprecated and will be removed, please use the new API '
            'register_module(name=None, force=False, module=None) instead.',
            DeprecationWarning)
        if cls is None:
            return partial(self.deprecated_register_module, force=force)
        self._register_module(cls, force=force)
        return cls

    def register_module(self, name=None, force=False, module=None):
        """Register a module.

        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(ResNet)

        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')
        # NOTE: This is a walkaround to be compatible with the old api,
        # while it may introduce unexpected bugs.
        if isinstance(name, type):
            return self.deprecated_register_module(name, force=force)

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(
                'name must be either of None, an instance of str or a sequence'
                f'  of str, but got {type(name)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            
            self._register_module(
                module_class=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(
                module_class=cls, module_name=name, force=force)
            return cls

        return _register
    
    def __getitem__(self, key: str):
        return self.get(key)
