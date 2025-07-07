import sys

if sys.platform == "win32":
    from .win_frameless_window import AcrylicWindow
    from .win_frameless_window import WindowsFramelessWindow as FramelessWindow
    from .win_frameless_window import WindowsWindowEffect as WindowEffect
elif sys.platform == "darwin":
    # from .mac import AcrylicWindow
    from .mac_frameless_window import MacFramelessWindow as FramelessWindow
    from ..mac_window_effect import MacWindowEffect as WindowEffect
else:
    from .linux import LinuxFramelessWindow as FramelessWindow
    from .linux import LinuxWindowEffect as WindowEffect

    AcrylicWindow = FramelessWindow
