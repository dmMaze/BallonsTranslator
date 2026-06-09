from ballontranslator.modules.package_import_names import PACKAGE_IMPORT_NAMES
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.py_package_manager import PyPackageManager


def create_package_manager() -> PyPackageManager:
    """Create a package manager from the current persistent package settings.

    >>> manager = create_package_manager()  # doctest: +SKIP
    >>> manager.package_import_names['pyyaml']
    ['yaml']
    """

    pmcfg = pcfg.package_manager
    return PyPackageManager(
        backend=pmcfg.installer_backend,
        extra_args=pmcfg.extra_install_args,
        package_import_names=PACKAGE_IMPORT_NAMES,
    )
