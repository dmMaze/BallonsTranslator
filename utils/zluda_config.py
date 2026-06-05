# 检测是否包含 ZLUDA 标记
def zluda_available(device_name):
    return "[ZLUDA]" in device_name


# 关闭 ZLUDA Cudnn 支持 防止错误
def enable_zluda_config():
    try:
        import torch
    except ModuleNotFoundError:
        # ZLUDA tuning is optional; do not make torch required for UI startup.
        return

    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print('Device name: ', device_name)
        print('Cuda is available: ', torch.cuda.is_available())
        print('Cuda version: ', torch.version.cuda)
        print('ZLUDA is available: ', zluda_available(device_name))

        if zluda_available(device_name):
            torch.backends.cudnn.enabled = False
            cuda_attr = torch.backends.cuda
            if hasattr(cuda_attr, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(False)
                print('Cuda enable flash sdp: ', False)
            if hasattr(cuda_attr, 'enable_math_sdp'):
                torch.backends.cuda.enable_math_sdp(True)
                print('Cuda enable math sdp: ', True)
            if hasattr(cuda_attr, 'enable_mem_efficient_sdp'):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                print('Cuda enable mem efficient sdp: ', False)
            if hasattr(cuda_attr, 'enable_cudnn_sdp'):
                torch.backends.cuda.enable_cudnn_sdp(False)
                print('Cuda enable cudnn sdp: ', False)
