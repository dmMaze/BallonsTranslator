import sys

if sys.platform == "win32":
    from .win_frameless_window import AcrylicWindow
    from .win_frameless_window import WindowsFramelessWindow as FramelessWindow
    from ..win32_utils import WindowsMoveResize as FramelessMoveResize
    from .win_frameless_window import WindowsWindowEffect as WindowEffect
elif sys.platform == "darwin":
    raise Exception(f'Please update to PySide6/PyQt6')
else:
    from .linux import LinuxFramelessWindow as FramelessWindow
    from .linux import LinuxWindowEffect as WindowEffect
    from .utils.linux_utils import LinuxMoveResize as FramelessMoveResize

    AcrylicWindow = FramelessWindow
