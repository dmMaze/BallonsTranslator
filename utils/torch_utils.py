from typing import List, Union, Tuple, Callable, Optional, Dict
import random
import inspect
import functools
import os
from functools import reduce
import os.path as osp
from packaging.version import Version, parse
import logging
import importlib
import sys
import operator as op
from requests import HTTPError
from pathlib import Path
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


import accelerate
import torch.nn as nn
import torch
from einops import rearrange
from torchvision.transforms.functional import pil_to_tensor
from torch import Tensor
import numpy as np
from PIL import Image
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    validate_hf_hub_args,
)
from huggingface_hub import hf_hub_download
import safetensors

import torchvision.transforms.functional as tv_functional



logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


_is_gguf_available = importlib.util.find_spec("gguf") is not None
if _is_gguf_available:
    try:
        _gguf_version = importlib_metadata.version("gguf")
        logger.debug(f"Successfully import gguf version {_gguf_version}")
    except importlib_metadata.PackageNotFoundError:
        _is_gguf_available = False


_torch_available = importlib.util.find_spec("torch") is not None
if _torch_available:
    try:
        _torch_version = importlib_metadata.version("torch")
        logger.info(f"PyTorch version {_torch_version} available.")
    except importlib_metadata.PackageNotFoundError:
        _torch_available = False


STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}
DEPRECATED_REVISION_ARGS = ["fp16", "non-ema"]
HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
WEIGHTS_NAME = "diffusion_pytorch_model.bin"
SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"


def is_gguf_available():
    return _is_gguf_available


def is_torch_available():
    return _torch_available


def load_gguf_checkpoint(gguf_checkpoint_path, return_tensors=False):
    """
    Load a GGUF file and return a dictionary of parsed parameters containing tensors, the parsed tokenizer and config
    attributes.

    Args:
        gguf_checkpoint_path (`str`):
            The path the to GGUF file to load
        return_tensors (`bool`, defaults to `True`):
            Whether to read the tensors from the file and return them. Not doing so is faster and only loads the
            metadata in memory.
    """

    if is_gguf_available() and is_torch_available():
        import gguf
        from gguf import GGUFReader

        from ..quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter
    else:
        logger.error(
            "Loading a GGUF checkpoint in PyTorch, requires both PyTorch and GGUF>=0.10.0 to be installed. Please see "
            "https://pytorch.org/ and https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    reader = GGUFReader(gguf_checkpoint_path)

    parsed_parameters = {}
    for tensor in reader.tensors:
        name = tensor.name
        quant_type = tensor.tensor_type

        # if the tensor is a torch supported dtype do not use GGUFParameter
        is_gguf_quant = quant_type not in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]
        if is_gguf_quant and quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
            _supported_quants_str = "\n".join([str(type) for type in SUPPORTED_GGUF_QUANT_TYPES])
            raise ValueError(
                (
                    f"{name} has a quantization type: {str(quant_type)} which is unsupported."
                    "\n\nCurrently the following quantization types are supported: \n\n"
                    f"{_supported_quants_str}"
                    "\n\nTo request support for this quantization type please open an issue here: https://github.com/huggingface/diffusers"
                )
            )

        weights = torch.from_numpy(tensor.data.copy())
        parsed_parameters[name] = GGUFParameter(weights, quant_type=quant_type) if is_gguf_quant else weights

    return parsed_parameters



def make_grid(x_max=.9693, y_max=.9375, n_x=28, n_y=16, device='cpu', dtype=torch.float32, flatten=False, target_size=None):
    '''
    returns:
        grid: shape (n_y, n_x, 3)
    '''
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-y_max, y_max, n_y),
        torch.linspace(-x_max, x_max, n_x),
        indexing="ij",
    )
    v = torch.ones_like(grid_x)
    grid = torch.stack([grid_x, grid_y, v], dim=-1)
    if target_size is not None:
        grid[..., 0] = grid[..., 0] * target_size[0] / 2
        grid[..., 1] = grid[..., 1] * target_size[1] / 2
    if flatten:
        grid = grid.reshape(1, -1, 3)
    return grid.to(device=device, dtype=dtype)








def tensor2img(t: torch.Tensor, output_type='numpy', denormalize=False, mean = 0., std = 255., from_mode: str = 'RGB', convert_mode: str = None, src_dim_order: str = 'chw', dtype=np.uint8, clip=(0, 255)) -> Union[Image.Image, np.array]:
    
    def _check_denormalize_params(values, num_channels):
        if isinstance(values, (int, float, np.ScalarType)):
            return values
        else:
            if isinstance(values, list):
                values = np.array(values)
            elif isinstance(values, torch.Tensor):
                values = values.to(device='cpu', dtype=torch.float32).numpy()
            else:
                raise Exception(f'invalid normalizing values: {values}')

            if len(values) > num_channels:
                values = values[:num_channels]
            assert len(values) == num_channels
            values = values.reshape((1, 1, -1))
            return values

    t = t.detach().to(device='cpu', dtype=torch.float32).squeeze().numpy()

    if t.ndim == 3:
        if src_dim_order == 'chw':
            t = rearrange(t, 'c h w -> h w c')
        c = t.shape[-1]
    else:
        assert t.ndim == 2, "t.ndim should be 2 or 3 after squeeze"
        c = 1

    if denormalize:
        t = (t * _check_denormalize_params(std, c)) + _check_denormalize_params(mean, c)

    if clip is not None:
        t = np.clip(t, clip[0], clip[1])
    image = t.astype(dtype)

    if output_type == 'pil':
        if len(image.shape) == 2:
            from_mode = 'L'
        image = Image.fromarray(image, mode=from_mode)
        if convert_mode is not None:
            image = image.convert(convert_mode)
    else:
        image = np.ascontiguousarray(image)
        assert output_type == 'numpy'
    return image


_IMG2TENSOR_IMGTYPE = (Image.Image, np.ndarray, str)
_IMG2TENSOR_DIMORDER = ('bchw', 'chw', 'hwc')
def img2tensor(img: Union[Image.Image, np.ndarray, str, torch.Tensor], normalize = False, mean = 0., std = 255., dim_order: str = 'bchw', dtype=torch.float32, device: str = 'cpu', imread_mode='RGB') -> Tensor:

    def _check_normalize_values(values, num_channels):
        if isinstance(values, tuple):
            values = list(values)
        elif isinstance(values, (int, float, np.ScalarType)):
            values = [values] * num_channels
        else:
            assert isinstance(values, (np.ndarray, list))
        if len(values) > num_channels:
            values = values[:num_channels]
        assert len(values) == num_channels
        return values

    assert isinstance(img, _IMG2TENSOR_IMGTYPE)
    assert dim_order in _IMG2TENSOR_DIMORDER

    if isinstance(img, str):
        img = load_image(img, mode=imread_mode)

    if isinstance(img, Image.Image):
        img = pil_to_tensor(img)
        if dim_order == 'bchw':
            img = img.unsqueeze(0)
        elif dim_order == 'hwc':
            img = img.permute((1, 2, 0))
    else:
        if img.ndim == 2:
            img = img[..., None]
        else:
            assert img.ndim == 3
        if dim_order == 'bchw':
            img = rearrange(img, 'h w c -> c h w')[None, ...]
        elif dim_order == 'chw':
            img = rearrange(img, 'h w c -> c h w')
        img = torch.from_numpy(np.ascontiguousarray(img))


    img = img.to(device=device, dtype=dtype)

    if normalize:

        if dim_order == 'bchw':
            c = img.shape[1]
        elif dim_order == 'chw':
            c = img.shape[0]
        else:
            c = img.shape[2]

        if mean is not None and std is not None:
            mean = _check_normalize_values(mean, c)
            std = _check_normalize_values(std, c)
            img = tv_functional.normalize(img, mean=mean, std=std)

    return img


def convert_tensor(t, dtype=torch.float32, device='cpu'):
    if isinstance(t, List) or isinstance(t, tuple) or isinstance(t, float) or isinstance(t, int):
        return torch.tensor(t, dtype=dtype, device=device)
    elif isinstance(t, np.ndarray):
        return torch.from_numpy(t).to(dtype=dtype, device=device)
    elif isinstance(t, torch.Tensor):
        return t.to(device=device, dtype=dtype)
    else:
        raise


TORCH_DTYPE_DICT = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
def get_torch_dtype(torch_dtype):
    if isinstance(torch_dtype, str):
        return TORCH_DTYPE_DICT[torch_dtype]
    return torch_dtype


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accelerate_should_save(accelerator):
    from accelerate import Accelerator, DistributedType
    # https://github.com/huggingface/diffusers/issues/2606#issuecomment-1463193509
    return accelerator.distributed_type == DistributedType.DEEPSPEED \
        or accelerator.is_main_process


def run_mp_func(rank, ws, args_split_by_rank, target_func, *args):

    file_list = args[0]
    devices = args[1]
    args = args[2:]

    arg_list = []
    if args_split_by_rank is not None:
        func_input_args = inspect.getfullargspec(target_func)
        _args_split_by_rank = set(args_split_by_rank)
        for arg_idx, argname in enumerate(func_input_args.args):
            if arg_idx < 2:    # skip file_list
                continue
            arg_val = args[arg_idx - 2]
            if isinstance(arg_val, str) and argname in _args_split_by_rank:
                if ',' in arg_val:
                    arg_val = arg_val.split(',')
                    arg_val = arg_val[rank]

            arg_list.append(arg_val)
    else:
        arg_list = args

    torch.cuda.set_device(devices[rank])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank])
    file_list = load_exec_list(file_list, rank, ws, check_exist=False)
    # with open(f'test_{rank}.txt', 'w') as f:
    #     f.write('\n'.join(file_list))
    arg_list = [file_list, devices] + arg_list
    target_func(*arg_list)


def torch_mp_wrapper(args_split_by_rank: list = None):

    def decorator(func):
        @functools.wraps(func)
        def wrapper(**kwargs):
            func_name = func.__name__
            assert func_name.endswith('_mp')
            target_func = getattr(inspect.getmodule(func), func_name[:-3])

            func_input_args = inspect.getfullargspec(target_func)
            target_func_kwargs = {}
            for arg_idx, argname in enumerate(func_input_args.args):
                target_func_kwargs[argname] = kwargs[argname]

            kwargs = target_func_kwargs
            assert 'devices' in kwargs and 'file_list' in kwargs
            kwargs_keys = list(kwargs.keys())
            assert kwargs_keys[1] == 'devices' and kwargs_keys[0] == 'file_list'
            file_list = kwargs['file_list']

            if 'seed' in kwargs:
                seed = kwargs['seed']
            else:
                seed = 0
            seed_everything(seed)

            file_list = load_exec_list(file_list, check_exist=False)

            if kwargs['devices'] == None:
                devices = [0]
            else:
                devices = kwargs['devices'].split(',')
                devices = [int(d) for d in devices]

            if len(devices) <= 1:
                kwargs['file_list'] = file_list
                target_func(**kwargs)
            else:
                random.shuffle(file_list)
                import torch.multiprocessing as mp
                ws = len(devices)
                arg_list = [v for v in kwargs.values()]
                arg_list[1] = devices
                arg_list[0] = file_list
                func_args = [ws, args_split_by_rank, target_func] + arg_list
                # func_args = tuple(func_args)
                mp.spawn(run_mp_func, nprocs=ws, args=func_args)
        return wrapper
    return decorator


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def get_module_by_name(module: Union[torch.Tensor, nn.Module],
                       access_string: str):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.

    https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8
    """
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)


# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L319
def compare_versions(library_or_version: Union[str, Version], operation: str, requirement_version: str):
    """
    Compares a library version to some requirement using a given operation.

    Args:
        library_or_version (`str` or `packaging.version.Version`):
            A library name or a version to check.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}")
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib_metadata.version(library_or_version))
    return operation(library_or_version, parse(requirement_version))


# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L338
def is_torch_version(operation: str, version: str):
    """
    Compares the current PyTorch version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of PyTorch
    """
    return compare_versions(parse(_torch_version), operation, version)


SAFETENSORS_FILE_EXTENSION = "safetensors"
GGUF_FILE_EXTENSION = "gguf"
def load_state_dict(checkpoint_file: Union[str, os.PathLike], variant: Optional[str] = None):
    """
    Reads a checkpoint file, returning properly formatted errors if they arise.
    """
    # TODO: We merge the sharded checkpoints in case we're doing quantization. We can revisit this change
    # when refactoring the _merge_sharded_checkpoints() method later.
    if isinstance(checkpoint_file, dict):
        return checkpoint_file
    try:
        file_extension = os.path.basename(checkpoint_file).split(".")[-1]
        if file_extension == SAFETENSORS_FILE_EXTENSION:
            return safetensors.torch.load_file(checkpoint_file, device="cpu")
        elif file_extension == GGUF_FILE_EXTENSION:
            return load_gguf_checkpoint(checkpoint_file)
        else:
            weights_only_kwarg = {"weights_only": True} if is_torch_version(">=", "1.13") else {}
            return torch.load(
                checkpoint_file,
                map_location="cpu",
                **weights_only_kwarg,
            )
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read().startswith("version"):
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from checkpoint file for '{checkpoint_file}' " f"at '{checkpoint_file}'. "
            )



@validate_hf_hub_args
def _get_model_file(
    pretrained_model_name_or_path: Union[str, Path],
    *,
    weights_name: str,
    subfolder: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    local_files_only: bool = False,
    token: Optional[str] = None,
    user_agent: Optional[Union[Dict, str]] = None,
    revision: Optional[str] = None,
    commit_hash: Optional[str] = None,
):
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if os.path.isfile(pretrained_model_name_or_path):
        return pretrained_model_name_or_path
    elif os.path.isdir(pretrained_model_name_or_path):
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, weights_name)):
            # Load from a PyTorch checkpoint
            model_file = os.path.join(pretrained_model_name_or_path, weights_name)
            return model_file
        elif subfolder is not None and os.path.isfile(
            os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
        ):
            model_file = os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
            return model_file
        else:
            raise EnvironmentError(
                f"Error no file named {weights_name} found in directory {pretrained_model_name_or_path}."
            )
    else:
        try:
            # 2. Load model file as usual
            model_file = hf_hub_download(
                pretrained_model_name_or_path,
                filename=weights_name,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                subfolder=subfolder,
                revision=revision or commit_hash,
            )
            return model_file

        except RepositoryNotFoundError as e:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                "token having permission to this repo with `token` or log in with `huggingface-cli "
                "login`."
            ) from e
        except RevisionNotFoundError as e:
            raise EnvironmentError(
                f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                "this model name. Check the model page at "
                f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
            ) from e
        except EntryNotFoundError as e:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {weights_name}."
            ) from e
        except HTTPError as e:
            raise EnvironmentError(
                f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n{e}"
            ) from e
        except ValueError as e:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                f" directory containing a file named {weights_name} or"
                " \nCheckout your internet connection or see how to run the library in"
                " offline mode at 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
            ) from e
        except EnvironmentError as e:
            raise EnvironmentError(
                f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing a file named {weights_name}"
            ) from e


def init_model_from_pretrained(
    pretrained_model_name_or_path: str,
    module_cls,
    subfolder=None,
    model_args = None,
    weights_name=None,
    patch_state_dict_func: Callable = None,
    download_from_hf=True,
    device='cpu',
    pass_statedict_to_model_init=False,
    dtype=torch.float32
):

    '''
    skip unnecessary param init for faster model creation, allow mismatch
    
    Args:
        module_cls (Callable): model class or model build function
    '''

    if download_from_hf:
        model_file = _get_model_file(pretrained_model_name_or_path=pretrained_model_name_or_path, subfolder=subfolder, weights_name=weights_name)
        state_dict = load_state_dict(model_file)
    else:
        if osp.exists(pretrained_model_name_or_path):
            state_dict = load_state_dict(pretrained_model_name_or_path)
        else:
            state_dict = torch.hub.load_state_dict_from_url(pretrained_model_name_or_path)

    if model_args is None:
        model_args = {}
    with accelerate.init_empty_weights(include_buffers=False):
        if pass_statedict_to_model_init:
            model, state_dict = module_cls(**model_args, state_dict=state_dict)
        else:
            model = module_cls(**model_args)

    if patch_state_dict_func is not None:
        _state_dict = patch_state_dict_func(model, state_dict)
        if _state_dict is not None:
            state_dict = _state_dict

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    incompatible_keys = model.load_state_dict(state_dict, strict=False, assign=True)
    
    missing_keys_set = set(incompatible_keys.missing_keys)
    for k in incompatible_keys.missing_keys:
        # assert k.endswith('.bias') or k.endswith('.weight')
        if k.endswith('.bias'):
            if k.replace('.bias', '.weight') in missing_keys_set:
                continue
        module = get_module_by_name(model, k.replace('.weight', '').replace('.bias', ''))
        if isinstance(module, torch.nn.Parameter):
            # print(module.data)
            module.data = torch.randn(module.data.size(), device='cpu')
        elif isinstance(module, torch.Tensor):
            pass
        else:
            module.to_empty(device="cpu")
            module.reset_parameters()

    model = model.to(device=device, dtype=dtype)
    return model


def image2np(image: Union[torch.Tensor, Image.Image, str, ], denormalize=True):

    if isinstance(image, torch.Tensor):
        image = tensor2img(image, mean=127.5, std=127.5, normalize=denormalize)
    elif isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        pass
    elif isinstance(image, str):
        image = load_image(image, output_type='numpy')
    else:
        raise Exception(f'invalid image type: {type(image)}')

    return image


def fix_params(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module