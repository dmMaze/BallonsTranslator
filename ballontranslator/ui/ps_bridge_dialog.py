import os
import sys
import glob
import re
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
import uuid
from typing import Any, List, Optional, Tuple

from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QGroupBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QMessageBox,
    QSizePolicy,
)
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QHideEvent, QShowEvent

from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.utils.proj_imgtrans import ProjImgTrans, TextBlkEncoder
from ballontranslator.utils.textblock import TextBlock


class PhotoshopBridgeUpdateError(ValueError):
    """Raised when a Photoshop update is stale or unsafe to apply."""


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, cls=TextBlkEncoder))


def _canonical_path(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _project_page_blocks(
    project: ProjImgTrans,
    page_name: str,
) -> Optional[List[TextBlock]]:
    blocks = project.pages.get(page_name)
    if blocks is None:
        blocks = getattr(project, "not_found_pages", {}).get(page_name)
    return blocks


def photoshop_bridge_block_snapshot(block: TextBlock) -> dict:
    block_data = _json_value(block.to_dict())
    fontformat = block_data.get("fontformat") or {}
    return {
        "translation": block_data.get("translation", ""),
        "rich_text": block_data.get("rich_text", ""),
        "font_size": fontformat.get("font_size"),
        "text": block_data.get("text", []),
        "xyxy": block_data.get("xyxy", []),
        "bounding_rect": block_data.get("_bounding_rect"),
    }


def project_state_matches_disk(project: ProjImgTrans) -> bool:
    """Return whether the live project still matches its canonical JSON file."""
    project_path = getattr(project, "proj_path", "")
    if not project_path or not os.path.isfile(project_path):
        return False
    try:
        with open(project_path, "r", encoding="utf-8") as project_file:
            disk_data = json.load(project_file)
        live_data = _json_value(project.to_dict())
        disk_data = _json_value(disk_data)
        # Page selection can change without dirtying project content and is
        # passed to Photoshop separately in the launch context.
        live_data.pop("current_img", None)
        disk_data.pop("current_img", None)
        return disk_data == live_data
    except (OSError, TypeError, ValueError):
        return False


def validate_photoshop_bridge_updates(
    project: ProjImgTrans,
    payload: dict,
) -> Tuple[str, List[Tuple[TextBlock, str, Optional[float]]]]:
    """Validate a complete bridge payload before any project state is changed.

    >>> from types import SimpleNamespace
    >>> from ballontranslator.utils.textblock import TextBlock
    >>> block = TextBlock(translation="old")
    >>> project = SimpleNamespace(
    ...     proj_path="project.json", pages={"page.png": [block]}
    ... )
    >>> payload = {
    ...     "version": 1,
    ...     "project_path": "project.json",
    ...     "page": "page.png",
    ...     "block_count": 1,
    ...     "updates": [{
    ...         "block_index": 0,
    ...         "translation": "new",
    ...         "font_size": None,
    ...         "base": photoshop_bridge_block_snapshot(block),
    ...     }],
    ... }
    >>> validate_photoshop_bridge_updates(project, payload)[0]
    'page.png'
    """
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise PhotoshopBridgeUpdateError("Unsupported Photoshop update format.")

    payload_path = payload.get("project_path")
    project_path = getattr(project, "proj_path", "")
    if (
        not isinstance(payload_path, str)
        or not project_path
        or _canonical_path(payload_path) != _canonical_path(project_path)
    ):
        raise PhotoshopBridgeUpdateError(
            "The Photoshop update belongs to a different project."
        )

    page_name = payload.get("page")
    if not isinstance(page_name, str):
        raise PhotoshopBridgeUpdateError("The Photoshop update has no valid page.")
    blocks = _project_page_blocks(project, page_name)
    if blocks is None:
        raise PhotoshopBridgeUpdateError(
            f"Page '{page_name}' is no longer in this project."
        )
    block_count = payload.get("block_count")
    if (
        not isinstance(block_count, int)
        or isinstance(block_count, bool)
        or block_count != len(blocks)
    ):
        raise PhotoshopBridgeUpdateError(
            f"The blocks on page '{page_name}' changed after it was opened in Photoshop."
        )

    raw_updates = payload.get("updates")
    if not isinstance(raw_updates, list):
        raise PhotoshopBridgeUpdateError("The Photoshop update list is invalid.")

    validated = []
    seen_indices = set()
    for update in raw_updates:
        if not isinstance(update, dict):
            raise PhotoshopBridgeUpdateError("A Photoshop block update is invalid.")
        block_index = update.get("block_index")
        if (
            not isinstance(block_index, int)
            or isinstance(block_index, bool)
            or block_index < 0
            or block_index >= len(blocks)
            or block_index in seen_indices
        ):
            raise PhotoshopBridgeUpdateError(
                "A Photoshop layer has an invalid or duplicate block number."
            )
        seen_indices.add(block_index)

        block = blocks[block_index]
        if update.get("base") != photoshop_bridge_block_snapshot(block):
            raise PhotoshopBridgeUpdateError(
                f"Block #{block_index + 1} on page '{page_name}' changed after "
                "it was opened in Photoshop."
            )

        translation = update.get("translation")
        if not isinstance(translation, str):
            raise PhotoshopBridgeUpdateError(
                f"Block #{block_index + 1} has invalid text."
            )
        font_size = update.get("font_size")
        if font_size is not None and (
            not isinstance(font_size, (int, float))
            or isinstance(font_size, bool)
            or not math.isfinite(font_size)
            or font_size <= 0
        ):
            raise PhotoshopBridgeUpdateError(
                f"Block #{block_index + 1} has an invalid font size."
            )
        validated.append(
            (
                block,
                translation.replace("\r\n", "\n").replace("\r", "\n"),
                font_size,
            )
        )

    return page_name, validated


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
    def __init__(self, parent=None, project=None) -> None:
        super().__init__(parent)
        self.project = project
        self._bridge_session_id = ""
        self._pending_update_path = ""
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(2000)
        self._update_timer.timeout.connect(self._consume_bridge_update)
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

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if self._pending_update_path:
            self._update_timer.start()

    def hideEvent(self, event: QHideEvent) -> None:
        self._update_timer.stop()
        super().hideEvent(event)

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

        lbl_sync = QLabel(self.tr("Photoshop Changes:"))
        lbl_sync.setStyleSheet("background: transparent;")
        self.sync_status_label = QLabel(self.tr("No pending changes"))
        self.sync_status_label.setWordWrap(True)
        self.sync_status_label.setStyleSheet(
            "background: transparent; color: gray;"
        )
        status_layout.addRow(lbl_sync, self.sync_status_label)

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

    def refresh_status(self) -> None:
        ps_dir, scripts_dir, exe_path = get_photoshop_paths()
        source_jsx = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "scripts", "export to photoshop", "BallonTranslator_PS_Bridge.jsx"
        )

        source_ver = extract_jsx_version(source_jsx) or "2.5.3"
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

        self._consume_bridge_update()

    def _set_sync_status(self, message: str, color: str) -> None:
        self.sync_status_label.setText(message)
        self.sync_status_label.setStyleSheet(
            f"background: transparent; color: {color};"
        )

    def _bridge_update_paths(self) -> List[str]:
        paths = []
        if self._pending_update_path:
            paths.append(self._pending_update_path)
        project_path = getattr(self.project, "proj_path", "")
        if project_path:
            fallback_path = project_path + ".ps_bridge_updates.json"
            if fallback_path not in paths:
                paths.append(fallback_path)
        return paths

    def _consume_bridge_update(self) -> None:
        for update_path in self._bridge_update_paths():
            if not os.path.isfile(update_path):
                continue
            try:
                with open(update_path, "r", encoding="utf-8") as update_file:
                    payload = json.load(update_file)
            except (OSError, ValueError):
                # Photoshop may still be replacing the sidecar; retry instead
                # of treating a partial read as bridge data.
                self._set_sync_status(
                    self.tr("Waiting for Photoshop to finish writing changes..."),
                    "#FFA000",
                )
                return

            session_id = (
                payload.get("session_id") if isinstance(payload, dict) else None
            )
            if (
                update_path == self._pending_update_path
                and session_id != self._bridge_session_id
            ):
                self._update_timer.stop()
                self._set_sync_status(
                    self.tr("Ignored changes from an expired Photoshop session."),
                    "#F44336",
                )
                return

            apply_updates = getattr(
                self.parent(), "apply_photoshop_bridge_updates", None
            )
            if not callable(apply_updates):
                self._update_timer.stop()
                self._set_sync_status(
                    self.tr("BallonsTranslator cannot apply Photoshop changes."),
                    "#F44336",
                )
                return
            try:
                updated_count = apply_updates(payload)
                os.remove(update_path)
            except (OSError, PhotoshopBridgeUpdateError, RuntimeError) as error:
                self._update_timer.stop()
                LOGGER.warning("Photoshop Bridge update was not applied: %s", error)
                self._set_sync_status(str(error), "#F44336")
                return

            if update_path == self._pending_update_path:
                self._pending_update_path = ""
                self._bridge_session_id = ""
                self._update_timer.stop()
            self._set_sync_status(
                self.tr("Applied {count} Photoshop change(s).").format(
                    count=updated_count
                ),
                "#4CAF50",
            )
            return

    def on_install_script(self) -> None:
        ps_dir, scripts_dir, _ = get_photoshop_paths()
        source_jsx = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "scripts", "export to photoshop", "BallonTranslator_PS_Bridge.jsx"
        )
        if not ps_dir or not os.path.isfile(source_jsx):
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("Photoshop or the Bridge JSX script was not found."),
            )
            return

        scripts_dir = scripts_dir or os.path.join(ps_dir, "Presets", "Scripts")
        target_jsx = os.path.join(scripts_dir, os.path.basename(source_jsx))
        is_update = os.path.isfile(target_jsx)

        def _show_success() -> None:
            self.refresh_status()
            if is_update:
                QMessageBox.information(
                    self,
                    self.tr("Updated"),
                    self.tr("Bridge script updated. It is ready to use."),
                )
            else:
                QMessageBox.information(
                    self,
                    self.tr("Installed"),
                    self.tr("Bridge script installed. Restart Photoshop to refresh its Scripts menu."),
                )

        try:
            os.makedirs(scripts_dir, exist_ok=True)
            shutil.copy2(source_jsx, target_jsx)
        except PermissionError as error:
            LOGGER.warning("Photoshop Bridge requires manual installation: %s", error)
            QMessageBox.warning(
                self,
                self.tr("Manual Installation Required"),
                self.tr(
                    "BallonsTranslator cannot write to Photoshop's Scripts "
                    "folder without administrator permission.\n\n"
                    "Copy this file:\n{source}\n\n"
                    "To this folder:\n{destination}\n\n"
                    "Then restart Photoshop. Explorer will now open the source "
                    "file and destination location."
                ).format(
                    source=os.path.normpath(source_jsx),
                    destination=os.path.normpath(scripts_dir),
                ),
            )
            self._open_manual_install_folders(source_jsx, scripts_dir)
            return
        except OSError as error:
            LOGGER.warning("Failed to install Photoshop Bridge: %s", error)
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("Failed to install the Bridge script: {error}").format(
                    error=error
                ),
            )
            return

        _show_success()

    def _open_manual_install_folders(
        self,
        source_jsx: str,
        scripts_dir: str,
    ) -> None:
        if sys.platform != "win32":
            return
        destination_dir = scripts_dir
        while destination_dir and not os.path.isdir(destination_dir):
            parent_dir = os.path.dirname(destination_dir)
            if parent_dir == destination_dir:
                break
            destination_dir = parent_dir
        try:
            source_norm = os.path.normpath(source_jsx)
            subprocess.Popen(f'explorer.exe /select,"{source_norm}"')
            if destination_dir and os.path.isdir(destination_dir):
                subprocess.Popen(f'explorer.exe "{os.path.normpath(destination_dir)}"')
        except OSError as error:
            LOGGER.warning("Failed to open manual Photoshop install folders: %s", error)

    def on_open_in_photoshop(self) -> None:
        _, scripts_dir, exe_path = get_photoshop_paths()
        bundled_jsx = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "scripts", "export to photoshop", "BallonTranslator_PS_Bridge.jsx"
        )
        installed_jsx = (
            os.path.join(scripts_dir, "BallonTranslator_PS_Bridge.jsx")
            if scripts_dir else None
        )
        # The quick action must use the app-matched protocol even when an older
        # copy is still installed in Photoshop's Scripts menu.
        target_jsx = bundled_jsx if os.path.isfile(bundled_jsx) else installed_jsx

        if not target_jsx or not os.path.isfile(target_jsx):
            QMessageBox.warning(self, self.tr("Error"), self.tr("Bridge JSX script not found."))
            return

        context_written = False
        # Save through the application before Photoshop takes its conflict-check
        # snapshot, then give this launch a unique return sidecar.
        if self.project and getattr(self.project, 'proj_path', None) and os.path.isfile(self.project.proj_path):
            try:
                prepare_context = getattr(
                    self.parent(), "prepare_photoshop_bridge_context", None
                )
                if callable(prepare_context):
                    ctx_data = prepare_context()
                else:
                    ctx_data = {
                        "project_path": os.path.abspath(self.project.proj_path),
                        "active_page": getattr(
                            self.project, "current_img", ""
                        ) or "",
                    }
                session_id = uuid.uuid4().hex
                update_path = os.path.join(
                    tempfile.gettempdir(),
                    f"bt_ps_bridge_updates_{session_id}.json",
                )
                context_file = os.path.join(tempfile.gettempdir(), "bt_ps_bridge_context.json")
                ctx_data.update({
                    "project_path": os.path.abspath(
                        ctx_data["project_path"]
                    ).replace("\\", "/"),
                    "timestamp": time.time(),
                    "session_id": session_id,
                    "update_path": update_path.replace("\\", "/"),
                })
                context_tmp = context_file + ".tmp"
                with open(context_tmp, "w", encoding="utf-8") as context_stream:
                    json.dump(ctx_data, context_stream, ensure_ascii=False, indent=2)
                os.replace(context_tmp, context_file)
                self._bridge_session_id = session_id
                self._pending_update_path = update_path
                context_written = True
            except Exception as ctx_err:
                LOGGER.warning("Failed to prepare Photoshop Bridge context: %s", ctx_err)
                QMessageBox.warning(
                    self,
                    self.tr("Error"),
                    f"{self.tr('Failed to prepare Photoshop Bridge')}: {ctx_err}",
                )
                return

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
                return

        if context_written:
            self._set_sync_status(
                self.tr("Waiting for changes from Photoshop..."),
                "#FFA000",
            )
            self._update_timer.start()

    def on_open_folder(self):
        scripts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "scripts", "export to photoshop"
        )
        if os.path.isdir(scripts_dir):
            if sys.platform == 'win32':
                subprocess.Popen(['explorer.exe', scripts_dir])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', scripts_dir])
            else:
                subprocess.Popen(['xdg-open', scripts_dir])
