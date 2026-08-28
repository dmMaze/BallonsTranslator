import os
import sys
import glob
import re
import hashlib
import subprocess
from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QMessageBox,
    QSizePolicy,
    QSpacerItem,
)
from qtpy.QtCore import Qt, QTimer

from ballontranslator.utils.logger import logger as LOGGER


def get_photoshop_paths():
    """Detect Photoshop installation directory and presets/scripts directory via Windows registry."""
    ps_dir = None
    scripts_dir = None
    exe_path = None

    if sys.platform == 'win32':
        import winreg

        reg_keys = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\Photoshop.exe"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\Photoshop.exe"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths\Photoshop.exe"),
        ]
        for hive, subkey in reg_keys:
            try:
                with winreg.OpenKey(hive, subkey) as key:
                    try:
                        p, _ = winreg.QueryValueEx(key, "Path")
                        if os.path.isdir(p):
                            ps_dir = os.path.normpath(p)
                    except OSError:
                        pass
                    try:
                        exe, _ = winreg.QueryValueEx(key, "")
                        exe_cleaned = exe.strip().strip('"')
                        if os.path.isfile(exe_cleaned):
                            exe_path = exe_cleaned
                            if not ps_dir:
                                ps_dir = os.path.dirname(exe_cleaned)
                    except OSError:
                        pass
                    if ps_dir:
                        break
            except OSError:
                pass

        if not ps_dir:
            for pattern in [
                r"C:\Program Files\Adobe\Adobe Photoshop *",
                r"D:\Program Files\Adobe\Adobe Photoshop *",
                r"E:\Program Files\Adobe\Adobe Photoshop *",
                r"F:\Program Files\Adobe\Adobe Photoshop *",
            ]:
                for match in glob.glob(pattern):
                    if os.path.isdir(match):
                        ps_dir = os.path.normpath(match)
                        break
                if ps_dir:
                    break

        if ps_dir:
            cand_scripts = os.path.join(ps_dir, "Presets", "Scripts")
            if os.path.isdir(cand_scripts):
                scripts_dir = cand_scripts
            if not exe_path:
                cand_exe = os.path.join(ps_dir, "Photoshop.exe")
                if os.path.isfile(cand_exe):
                    exe_path = cand_exe

    return ps_dir, scripts_dir, exe_path


def extract_jsx_version(jsx_path: str) -> str:
    """Read version metadata from a JSX script header."""
    if not jsx_path or not os.path.isfile(jsx_path):
        return ""
    try:
        with open(jsx_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(40):
                line = f.readline()
                if not line:
                    break
                m = re.search(r'(?:Version:\s*|BT_BRIDGE_VERSION\s*=\s*["\'])([\d\.]+)', line)
                if m:
                    return m.group(1)
    except Exception:
        pass
    return ""


def get_file_md5(file_path: str) -> str:
    """Calculate MD5 hash of file content for exact equality check."""
    if not file_path or not os.path.isfile(file_path):
        return ""
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


class PhotoshopBridgeDialog(QDialog):
    def __init__(self, parent=None, project=None):
        super().__init__(parent)
        self.project = project
        self.setObjectName("PhotoshopBridgeDialog")
        self.setWindowTitle(self.tr("Photoshop Bridge"))
        self.setMinimumWidth(480)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowCloseButtonHint
        )

        self.init_ui()
        self.refresh_status()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        self.setStyleSheet("""
            QDialog#PhotoshopBridgeDialog QLabel,
            QDialog#PhotoshopBridgeDialog QGroupBox {
                background-color: transparent;
            }
            QPushButton#OpenPSBtn {
                background-color: rgb(30, 147, 229);
                color: #ffffff;
                font-weight: bold;
                border: 1px solid rgb(25, 125, 195);
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton#OpenPSBtn:hover {
                background-color: rgb(45, 160, 245);
            }
            QPushButton#OpenPSBtn:pressed {
                background-color: rgb(20, 120, 190);
            }
            QPushButton#OpenPSBtn:disabled {
                background-color: rgba(30, 147, 229, 60);
                color: rgba(255, 255, 255, 120);
                border: 1px solid transparent;
            }
        """)

        # 1. Status Group
        status_group = QGroupBox(self.tr("Photoshop Integration Status"))
        status_group.setObjectName("PSStatusGroup")
        status_layout = QFormLayout(status_group)
        status_layout.setSpacing(8)
        status_layout.setContentsMargins(12, 12, 12, 12)

        lbl_ps = QLabel(self.tr("Photoshop:"))
        lbl_ps.setStyleSheet("background: transparent;")
        self.ps_status_label = QLabel(self.tr("Checking Photoshop installation..."))
        self.ps_status_label.setObjectName("PSStatusFieldLabel")
        self.ps_status_label.setStyleSheet("background: transparent;")
        self.ps_status_label.setWordWrap(True)
        status_layout.addRow(lbl_ps, self.ps_status_label)

        lbl_script = QLabel(self.tr("Bridge Script:"))
        lbl_script.setStyleSheet("background: transparent;")
        self.script_status_label = QLabel(self.tr("Checking Bridge script status..."))
        self.script_status_label.setObjectName("PSStatusFieldLabel")
        self.script_status_label.setStyleSheet("background: transparent;")
        self.script_status_label.setWordWrap(True)
        status_layout.addRow(lbl_script, self.script_status_label)

        self.check_update_btn = QPushButton(self.tr("Check Update"))
        self.check_update_btn.setFixedHeight(28)
        self.check_update_btn.clicked.connect(self.refresh_status)
        status_layout.addRow("", self.check_update_btn)

        layout.addWidget(status_group)

        # 2. Quick Actions Group
        actions_group = QGroupBox(self.tr("Bridge Quick Actions"))
        actions_group.setObjectName("PSActionsGroup")
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.setSpacing(8)
        actions_layout.setContentsMargins(12, 12, 12, 12)

        # Launch / Open in Photoshop button
        self.open_ps_btn = QPushButton(self.tr("Open This Project in Photoshop"))
        self.open_ps_btn.setObjectName("OpenPSBtn")
        self.open_ps_btn.setFixedHeight(36)
        self.open_ps_btn.clicked.connect(self.on_open_in_photoshop)
        actions_layout.addWidget(self.open_ps_btn)

        # Install / Update script button
        self.install_btn = QPushButton(self.tr("Install / Update Script in Photoshop"))
        self.install_btn.setFixedHeight(30)
        self.install_btn.clicked.connect(self.on_install_script)
        actions_layout.addWidget(self.install_btn)

        # Open Scripts folder in Explorer
        self.open_folder_btn = QPushButton(self.tr("Open Scripts Folder in Explorer"))
        self.open_folder_btn.setFixedHeight(30)
        self.open_folder_btn.clicked.connect(self.on_open_folder)
        actions_layout.addWidget(self.open_folder_btn)

        layout.addWidget(actions_group)

        # 3. Help info
        info_label = QLabel(
            self.tr(
                "Tip: Inside Photoshop, run the bridge via File -> Scripts -> BallonTranslator_PS_Bridge.\n"
                "You can export layers, edit texts/strokes, and sync changes back into BallonsTranslator."
            )
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("background: transparent; color: gray; font-size: 11px;")
        layout.addWidget(info_label)

    def refresh_status(self):
        ps_dir, scripts_dir, exe_path = get_photoshop_paths()
        source_jsx = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "scripts", "export to photoshop", "BallonTranslator_PS_Bridge.jsx"
        )

        source_ver = extract_jsx_version(source_jsx) or "2.5.2"
        source_hash = get_file_md5(source_jsx)

        installed_jsx = os.path.join(scripts_dir, "BallonTranslator_PS_Bridge.jsx") if scripts_dir else None
        is_installed = bool(installed_jsx and os.path.isfile(installed_jsx))

        if ps_dir:
            ps_name = os.path.basename(ps_dir)
            self.ps_status_label.setText(f"<span style='color: #4CAF50; font-weight: bold;'>{ps_name}</span>")
            self.install_btn.setEnabled(True)
            self.open_ps_btn.setEnabled(True)
            self.open_ps_btn.setToolTip("")
        else:
            self.ps_status_label.setText(f"<span style='color: #F44336; font-weight: bold;'>{self.tr('Not detected')}</span>")
            self.install_btn.setEnabled(False)
            self.open_ps_btn.setEnabled(False)
            self.open_ps_btn.setToolTip(self.tr("Photoshop installation not detected"))

        if is_installed:
            installed_ver = extract_jsx_version(installed_jsx) or "1.0.0"
            installed_hash = get_file_md5(installed_jsx)

            is_identical = (source_hash and installed_hash and source_hash == installed_hash)

            if is_identical or installed_ver == source_ver:
                self.script_status_label.setText(
                    f"<span style='color: #4CAF50; font-weight: bold;'>{self.tr('Installed')} (v{installed_ver})</span>"
                )
                self.install_btn.setText(self.tr("Reinstall Script in Photoshop"))
            else:
                self.script_status_label.setText(
                    f"<span style='color: #FFA000; font-weight: bold;'>{self.tr('Update available')} (v{installed_ver} -> v{source_ver})</span>"
                )
                self.install_btn.setText(self.tr("Update Script in Photoshop"))
        else:
            self.script_status_label.setText(
                f"<span style='color: #F44336; font-weight: bold;'>{self.tr('Not installed')} ({self.tr('v{version} available').format(version=source_ver)})</span>"
            )
            self.install_btn.setText(self.tr("Install Script in Photoshop"))

    def on_install_script(self):
        scripts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "scripts", "export to photoshop"
        )
        bat_file = os.path.join(scripts_dir, "install_ps_script.bat")

        if os.path.isfile(bat_file):
            try:
                subprocess.Popen(f'cmd.exe /c "{bat_file}"', shell=True, cwd=scripts_dir)
                QTimer.singleShot(2000, self.refresh_status)
                QTimer.singleShot(5000, self.refresh_status)
            except Exception as e:
                QMessageBox.warning(self, self.tr("Error"), f"{self.tr('Failed to start installer')}: {e}")
        else:
            QMessageBox.warning(self, self.tr("Error"), self.tr("Installer script not found."))

    def on_open_in_photoshop(self):
        _, scripts_dir, exe_path = get_photoshop_paths()
        target_jsx = os.path.join(scripts_dir, "BallonTranslator_PS_Bridge.jsx") if scripts_dir else None
        if not target_jsx or not os.path.isfile(target_jsx):
            target_jsx = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "scripts", "export to photoshop", "BallonTranslator_PS_Bridge.jsx"
            )

        if not os.path.isfile(target_jsx):
            QMessageBox.warning(self, self.tr("Error"), self.tr("Bridge JSX script not found."))
            return

        # Write Bridge Context for seamless automatic project pickup in Photoshop
        if self.project and getattr(self.project, 'proj_path', None) and os.path.isfile(self.project.proj_path):
            try:
                import json
                import tempfile
                import time
                context_file = os.path.join(tempfile.gettempdir(), "bt_ps_bridge_context.json")
                ctx_data = {
                    "project_path": os.path.abspath(self.project.proj_path).replace("\\", "/"),
                    "active_page": getattr(self.project, "curr_imgname", "") or "",
                    "timestamp": time.time(),
                }
                with open(context_file, "w", encoding="utf-8") as f:
                    json.dump(ctx_data, f, ensure_ascii=False, indent=2)
            except Exception as ctx_err:
                LOGGER.warning(f"Failed to write Photoshop Bridge context: {ctx_err}")

        if exe_path and os.path.isfile(exe_path):
            try:
                subprocess.Popen([exe_path, "-r", target_jsx])
            except Exception as e:
                LOGGER.error(f"Failed to launch Photoshop with script: {e}")
                os.startfile(target_jsx)
        else:
            try:
                os.startfile(target_jsx)
            except Exception as e:
                QMessageBox.warning(self, self.tr("Error"), f"{self.tr('Failed to open Photoshop')}: {e}")

    def on_open_folder(self):
        scripts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "scripts", "export to photoshop"
        )
        if os.path.isdir(scripts_dir):
            if sys.platform == 'win32':
                subprocess.Popen(f'explorer.exe "{scripts_dir}"', shell=True)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', scripts_dir])
            else:
                subprocess.Popen(['xdg-open', scripts_dir])
