"""Generate deterministic offscreen evidence after GUI capture was unavailable."""

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyqt6")

from qtpy.QtCore import QPointF, QRectF, Qt, qVersion
from qtpy.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QTextCursor,
)
from qtpy.QtWidgets import (
    QApplication,
    QGraphicsPolygonItem,
    QGraphicsScene,
)

from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.text_advanced_format import TextAdvancedFormatPanel
from ballontranslator.ui.textitem import TextBlkItem
from ballontranslator.utils import shared as app_shared
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.proj_imgtrans import ProjImgTrans, TextBlkEncoder
from ballontranslator.utils.textblock import TextBlock


FEATURE_SHA = "f101479edbaf4b35606fe7503ae752846055d71d"
UPSTREAM_SHA = "6155f9b303033b24f57a2c025d2edbfed3eb847f"
ARTIFACT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.abspath(os.path.join(ARTIFACT_DIR, "..", "..", ".."))
BASELINE_DIR = r"F:\ballon\BallonsTranslator-baseline-6155f9b"
OVERLAY_DIR = r"F:\ballon\pr1238-baseline-overlay-6155f9b"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")


def run_checked(command, **kwargs):
    return subprocess.run(
        command,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        **kwargs,
    )


def run_recorded(label, command, *, cwd, environment, expected_exit_codes):
    process_env = os.environ.copy()
    process_env.update(environment)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=process_env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    record = {
        "label": label,
        "cwd": cwd,
        "environment_overrides": environment,
        "command": command,
        "exit_code": result.returncode,
        "expected_exit_codes": list(expected_exit_codes),
        "exit_code_as_expected": result.returncode in expected_exit_codes,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stdout_nonempty_line_count": sum(
            1 for line in result.stdout.splitlines() if line.strip()
        ),
    }
    if not record["exit_code_as_expected"]:
        raise AssertionError(
            f"{label} exit {result.returncode}, expected {expected_exit_codes}"
        )
    return record


def write_raw_log(path, records):
    with open(path, "w", encoding="utf-8") as output:
        for record in records:
            output.write(f"=== {record['label']} ===\n")
            output.write(f"cwd={record['cwd']}\n")
            output.write(
                "environment_overrides="
                + json.dumps(record["environment_overrides"], sort_keys=True)
                + "\n"
            )
            output.write(
                "command=" + json.dumps(record["command"], ensure_ascii=False) + "\n"
            )
            output.write(f"exit_code={record['exit_code']}\n")
            output.write(
                "expected_exit_codes="
                + json.dumps(record["expected_exit_codes"])
                + "\n"
            )
            output.write(
                f"exit_code_as_expected={record['exit_code_as_expected']}\n"
            )
            output.write(
                f"stdout_nonempty_line_count={record['stdout_nonempty_line_count']}\n"
            )
            output.write("--- stdout ---\n")
            output.write(record["stdout"])
            if record["stdout"] and not record["stdout"].endswith("\n"):
                output.write("\n")
            output.write("--- stderr ---\n")
            output.write(record["stderr"])
            if record["stderr"] and not record["stderr"].endswith("\n"):
                output.write("\n")
            output.write("=== end ===\n\n")


def git_command(*arguments, check=True):
    command = [
        "git",
        "-c",
        f"safe.directory={REPO_DIR.replace(os.sep, '/')}",
        "-C",
        REPO_DIR,
        *arguments,
    ]
    return subprocess.run(
        command,
        check=check,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )


def collect_git_provenance():
    head = git_command("rev-parse", "HEAD").stdout.strip()
    branch = git_command("branch", "--show-current").stdout.strip()
    status = git_command(
        "status", "--short", "--untracked-files=all"
    ).stdout.splitlines()
    feature_diff = git_command(
        "diff", "--quiet", "HEAD", "--", "ballontranslator", "tests", check=False
    )
    staged_feature_diff = git_command(
        "diff",
        "--cached",
        "--quiet",
        "HEAD",
        "--",
        "ballontranslator",
        "tests",
        check=False,
    )
    return {
        "expected_feature_sha": FEATURE_SHA,
        "git_head": head,
        "head_matches_expected_feature_sha": head == FEATURE_SHA,
        "branch": branch,
        "repo_dir": REPO_DIR,
        "git_status_porcelain": status,
        "tracked_feature_paths_clean": feature_diff.returncode == 0,
        "staged_feature_paths_clean": staged_feature_diff.returncode == 0,
        "scope_note": (
            "Untracked doc/artifact files are expected while this evidence bundle is "
            "being generated; production and test paths are compared to HEAD separately."
        ),
    }


def new_image(width, height, color=QColor(247, 249, 252)):
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(color)
    return image


def save_image(image, name):
    path = os.path.join(ARTIFACT_DIR, name)
    if not image.save(path, "PNG"):
        raise RuntimeError(f"could not save {path}")
    return path


def draw_header(image, title, subtitle=""):
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QColor(25, 34, 49))
    title_font = QFont("Arial", 18)
    title_font.setBold(True)
    painter.setFont(title_font)
    painter.drawText(QPointF(24, 32), title)
    if subtitle:
        painter.setFont(QFont("Arial", 9))
        painter.setPen(QColor(84, 96, 112))
        painter.drawText(QPointF(24, 51), subtitle)
    painter.end()


def render_scene(scene, width, height, title, subtitle="", source_rect=None):
    image = new_image(width, height)
    painter = QPainter(image)
    painter.setRenderHints(
        QPainter.RenderHint.Antialiasing
        | QPainter.RenderHint.TextAntialiasing
        | QPainter.RenderHint.SmoothPixmapTransform
    )
    target = QRectF(18, 62, width - 36, height - 80)
    scene.render(painter, target, source_rect or scene.sceneRect())
    painter.end()
    draw_header(image, title, subtitle)
    return image


def make_item(
    scene,
    rect,
    text,
    transform,
    *,
    vertical=False,
    rotation=0,
    font_size=34,
    stroke=0.0,
    shadow=0.0,
    shadow_offset=(0.0, 0.0),
    gradient=False,
    colors=([20, 45, 80], [225, 75, 95]),
):
    x, y, width, height = rect
    fontformat = FontFormat(
        font_family="Malgun Gothic",
        font_size=font_size,
        frgb=list(colors[0]),
        srgb=[19, 31, 48],
        vertical=vertical,
        stroke_width=stroke,
        shadow_radius=shadow,
        shadow_strength=0.78,
        shadow_color=[25, 29, 42],
        shadow_offset=list(shadow_offset),
        gradient_enabled=gradient,
        gradient_start_color=list(colors[0]),
        gradient_end_color=list(colors[1]),
        gradient_angle=24.0,
        gradient_size=0.72,
        horizontal_scale=transform[0],
        vertical_scale=transform[1],
        slant_angle=transform[2],
    )
    block = TextBlock(
        xyxy=[x, y, x + width, y + height],
        _bounding_rect=[x, y, width, height],
        translation=text,
        fontformat=fontformat,
    )
    item = TextBlkItem(block)
    scene.addItem(item)
    item.setRotation(rotation)
    outline = QGraphicsPolygonItem(item.visual_polygon_in_scene())
    pen = QPen(QColor(26, 123, 225, 190), 1.5, Qt.PenStyle.DashLine)
    pen.setCosmetic(True)
    outline.setPen(pen)
    outline.setBrush(QBrush(Qt.BrushStyle.NoBrush))
    outline.setZValue(100)
    scene.addItem(outline)
    item._evidence_outline = outline
    return item


def controls_evidence(app):
    original_register = getattr(app_shared, "register_view_widget", None)
    app_shared.register_view_widget = lambda *_args, **_kwargs: None
    try:
        panel = TextAdvancedFormatPanel(
            "Advanced Text Format",
            "text_advanced_format_panel",
            "expand_tadvanced_panel",
            lambda *_args: None,
        )
        panel.set_active_format(
            FontFormat(
                horizontal_scale=1.25,
                vertical_scale=0.8,
                slant_angle=17.0,
                shadow_radius=0.24,
                shadow_strength=0.7,
                shadow_offset=[-2.0, 3.0],
                gradient_enabled=True,
                gradient_start_color=[30, 90, 220],
                gradient_end_color=[235, 70, 130],
                gradient_angle=35,
                gradient_size=0.8,
            )
        )
        widget = panel.view_widget
        widget.resize(1120, 410)
        widget.show()
        app.processEvents()
        image = new_image(1160, 480)
        painter = QPainter(image)
        painter.translate(20, 58)
        widget.render(painter)
        painter.end()
        draw_header(
            image,
            "Advanced Text Format controls",
            "Committed H/V percentages and slant degrees; Enter/focus-out/drag are transactional.",
        )
        widget.close()
        save_image(image, "controls.png")
    finally:
        if original_register is None:
            del app_shared.register_view_widget
        else:
            app_shared.register_view_widget = original_register


def horizontal_vertical_evidence():
    scene = QGraphicsScene()
    scene.setSceneRect(0, 0, 1080, 520)
    make_item(
        scene,
        (80, 110, 390, 120),
        "Horizontal  가나  Latin  e\u0301  😀",
        (1.45, 0.72, 18.0),
        rotation=7,
        stroke=0.08,
        shadow=0.14,
        shadow_offset=(-0.12, 0.18),
        gradient=True,
    )
    make_item(
        scene,
        (680, 70, 145, 350),
        "세로 漢字 AB 😀",
        (0.78, 1.48, -16.0),
        vertical=True,
        rotation=-8,
        font_size=30,
        stroke=0.09,
        shadow=0.12,
        shadow_offset=(0.16, 0.12),
        gradient=True,
        colors=([24, 125, 82], [240, 165, 45]),
    )
    image = render_scene(
        scene,
        1120,
        600,
        "Horizontal and vertical use one item-local matrix",
        "Blue dashed quadrilaterals are the exact transformed logical bounds.",
    )
    save_image(image, "horizontal-vertical.png")


def extreme_effects_evidence():
    scene = QGraphicsScene()
    scene.setSceneRect(0, 0, 1500, 980)
    cases = (
        (
            (0.1, 4.0, 45.0),
            (215, 130, 260, 82),
            -7,
            "極 A",
            (75, 66),
            "H=0.1  V=4.0  slant=+45°  rotation=-7°",
        ),
        (
            (4.0, 0.1, -45.0),
            (930, 185, 82, 245),
            8,
            "縦 AB",
            (795, 66),
            "H=4.0  V=0.1  slant=-45°  rotation=+8°",
        ),
        (
            (2.4, 1.7, 32.0),
            (205, 665, 230, 76),
            14,
            "Effects",
            (75, 552),
            "H=2.4  V=1.7  slant=+32°  stroke/shadow/gradient",
        ),
        (
            (0.55, 2.5, -33.0),
            (990, 650, 100, 185),
            -12,
            "混合 Z",
            (795, 552),
            "H=0.55  V=2.5  slant=-33°  mixed vertical text",
        ),
    )
    card_pen = QPen(QColor(204, 213, 224), 1.2)
    card_brush = QBrush(QColor(255, 255, 255, 225))
    scene.addRect(QRectF(40, 38, 680, 450), card_pen, card_brush)
    scene.addRect(QRectF(760, 38, 680, 450), card_pen, card_brush)
    scene.addRect(QRectF(40, 524, 680, 420), card_pen, card_brush)
    scene.addRect(QRectF(760, 524, 680, 420), card_pen, card_brush)
    for index, (transform, rect, rotation, text, caption_pos, caption_text) in enumerate(cases):
        caption = scene.addText(caption_text, QFont("Arial", 12))
        caption.setDefaultTextColor(QColor(36, 50, 69))
        caption.setPos(*caption_pos)
        caption.setZValue(110)
        make_item(
            scene,
            rect,
            text,
            transform,
            vertical=index in (1, 3),
            rotation=rotation,
            font_size=27 if index else 24,
            stroke=0.18,
            shadow=0.32,
            shadow_offset=(-0.25 if index % 2 else 0.25, 0.28),
            gradient=True,
            colors=(
                [30 + index * 30, 70, 190 - index * 20],
                [235, 65 + index * 25, 75],
            ),
        )
    content_bounds = scene.itemsBoundingRect()
    padded_bounds = content_bounds.adjusted(-28, -28, 28, 28)
    scene.setSceneRect(padded_bounds)
    if not scene.sceneRect().contains(content_bounds):
        raise AssertionError("extreme/effects scene does not contain all evidence items")
    image = render_scene(
        scene,
        1560,
        1060,
        "Extreme transforms and effects",
        "Clamp boundaries, ±45° slant, rotation, stroke, shadow, gradient and clipping envelope.",
        source_rect=padded_bounds,
    )
    save_image(image, "extreme-effects.png")
    return {
        "all_items_inside_source_rect": padded_bounds.contains(content_bounds),
        "source_rect": [
            padded_bounds.x(),
            padded_bounds.y(),
            padded_bounds.width(),
            padded_bounds.height(),
        ],
        "items_bounds": [
            content_bounds.x(),
            content_bounds.y(),
            content_bounds.width(),
            content_bounds.height(),
        ],
    }


def editing_export_evidence():
    width, height = 620, 380
    canvas = Canvas()
    base = np.full((height, width, 4), 255, dtype=np.uint8)
    base[..., :3] = [248, 249, 252]
    canvas.imgtrans_proj = SimpleNamespace(
        img_valid=True,
        inpainted_valid=True,
        inpainted_array=base,
    )
    transparent = QPixmap(width, height)
    transparent.fill(Qt.GlobalColor.transparent)
    canvas.inpaintLayer.setPixmap(transparent)
    canvas.textLayer.setPixmap(transparent)
    canvas.baseLayer.setRect(QRectF(0, 0, width, height))
    canvas.setSceneRect(QRectF(0, 0, width, height))
    item = make_item(
        canvas,
        (105, 120, 350, 100),
        "Edit → Canvas → Export  한글 Latin",
        (1.7, 0.68, 21.0),
        rotation=-11,
        font_size=30,
        stroke=0.16,
        shadow=0.2,
        shadow_offset=(-0.18, 0.2),
        gradient=True,
    )
    # make_item added the item directly; Canvas text/export ownership is the
    # text layer, so move it without recreating document/layout state.
    item.setParentItem(canvas.textLayer)
    canvas.editor_index = 1
    canvas.txtblkShapeControl.blk_item = item
    item.startEdit()
    item.setSelected(True)
    cursor = item.textCursor()
    cursor.setPosition(4)
    cursor.setPosition(12, QTextCursor.MoveMode.KeepAnchor)
    item.setTextCursor(cursor)

    editing = new_image(width, height, QColor(248, 249, 252))
    painter = QPainter(editing)
    canvas.render(painter, QRectF(0, 0, width, height), QRectF(0, 0, width, height))
    painter.end()

    # The dashed polygon is evidence-only.  Keep it in the editing pane, then
    # remove it from the actual Canvas scene before invoking the production
    # export path so the right pane cannot accidentally include it.
    evidence_outline = item._evidence_outline
    canvas.removeItem(evidence_outline)
    outline_removed_before_export = evidence_outline.scene() is None
    if not outline_removed_before_export:
        raise AssertionError("evidence outline remained in the Canvas export scene")
    exported = canvas.render_result_img()

    combined = new_image(1280, 500)
    painter = QPainter(combined)
    painter.drawImage(QRectF(20, 90, 600, 368), editing)
    painter.drawImage(QRectF(660, 90, 600, 368), exported)
    painter.setPen(QColor(25, 34, 49))
    label_font = QFont("Arial", 11)
    label_font.setBold(True)
    painter.setFont(label_font)
    painter.drawText(QPointF(24, 78), "Editing / selection overlay")
    painter.drawText(QPointF(664, 78), "Actual Canvas.render_result_img() export")
    painter.end()
    draw_header(
        combined,
        "Editing, Canvas and export share the same live layout",
        "Export ends editing and omits cursor/selection while preserving fill, stroke, shadow and gradient.",
    )
    save_image(combined, "editing-canvas-export.png")
    return {
        "evidence_outline_removed_before_canvas_export": (
            outline_removed_before_export
        ),
        "export_method": "Canvas.render_result_img()",
    }


def reload_project_report(project_path):
    """Read one fixture from disk in a short-lived helper process."""
    fixture_dir = os.path.dirname(os.path.abspath(project_path))
    project = ProjImgTrans()
    returned_new_project = project.load(fixture_dir, json_path=project_path)
    page_name = "page.png"
    block = project.pages[page_name][0]
    with open(project_path, "r", encoding="utf-8") as source:
        raw_payload = json.load(source)
    encoded_block = json.loads(json.dumps(block, cls=TextBlkEncoder))
    return {
        "read_from_disk": True,
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "python_executable": sys.executable,
        "project_path": os.path.relpath(project_path, ARTIFACT_DIR),
        "project_sha256": sha256_file(project_path),
        "page_image_sha256": sha256_file(os.path.join(fixture_dir, page_name)),
        "load_returned_new_project": returned_new_project,
        "schema_version": raw_payload.get("text_transform_schema_version"),
        "stored_directory": raw_payload.get("directory"),
        "page_name": page_name,
        "text_transform": list(block.fontformat.text_transform),
        "block": encoded_block,
        "git": collect_git_provenance(),
    }


def save_reload_evidence():
    fixture_dir = os.path.join(ARTIFACT_DIR, "fixture-project")
    os.makedirs(fixture_dir, exist_ok=True)
    page_path = os.path.join(fixture_dir, "page.png")
    page = new_image(640, 320, QColor(242, 246, 250))
    page_painter = QPainter(page)
    page_painter.setPen(QColor(210, 220, 230))
    for x in range(0, 641, 40):
        page_painter.drawLine(x, 0, x, 320)
    for y in range(0, 321, 40):
        page_painter.drawLine(0, y, 640, y)
    page_painter.setPen(QColor(91, 105, 120))
    page_painter.setFont(QFont("Arial", 12))
    page_painter.drawText(QPointF(18, 28), "Portable on-disk project fixture")
    page_painter.end()
    if not page.save(page_path, "PNG"):
        raise RuntimeError(f"could not save {page_path}")

    original = TextBlock(
        xyxy=[60, 70, 440, 190],
        _bounding_rect=[60, 70, 380, 120],
        translation="Save → restart → reload  저장",
        rich_text="<p><b>Save → restart → reload</b> 저장</p>",
        fontformat=FontFormat(
            font_family="Arial",
            font_size=32,
            stroke_width=0.11,
            shadow_radius=0.17,
            shadow_strength=0.75,
            shadow_offset=[-0.14, 0.18],
            gradient_enabled=True,
            gradient_start_color=[35, 95, 210],
            gradient_end_color=[225, 65, 125],
            horizontal_scale=1.234568,
            vertical_scale=0.625,
            slant_angle=-14.5,
        ),
    )
    project_path = os.path.join(fixture_dir, "project.json")
    source = ProjImgTrans()
    # A relative directory keeps the committed fixture portable.  ProjImgTrans
    # load() receives the fixture directory explicitly, exactly as the app does.
    source.directory = "."
    source.proj_path = project_path
    source.pages = {"page.png": [original]}
    source.not_found_pages = {}
    source._image_info = {"page.png": {"finish_code": 0}}
    source.current_img = "page.png"
    source.save()

    with open(project_path, "r", encoding="utf-8") as saved_project:
        saved_payload = json.load(saved_project)
    saved_format = saved_payload["pages"]["page.png"][0]["fontformat"]
    saved_block = saved_payload["pages"]["page.png"][0]
    aliases_absent = (
        "italic_angle" not in saved_format
        and "horizontal_scale" not in saved_block
        and "vertical_scale" not in saved_block
        and "italic_angle" not in saved_block
        and "ballontranslator-logical-stretch-v1" not in saved_block["rich_text"]
    )
    if saved_payload.get("text_transform_schema_version") != 1:
        raise AssertionError("fixture did not emit text-transform schema v1")
    if saved_payload.get("directory") != ".":
        raise AssertionError("fixture project directory is not portable")
    if not aliases_absent:
        raise AssertionError("fixture contains a legacy transform alias")

    command = [sys.executable, os.path.abspath(__file__), "--reload-project", project_path]
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    children = [
        subprocess.Popen(
            command,
            cwd=REPO_DIR,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for _ in range(2)
    ]
    reports = []
    for index, child in enumerate(children, start=1):
        stdout, stderr = child.communicate(timeout=120)
        if child.returncode != 0:
            raise RuntimeError(
                f"fresh reload process {index} failed ({child.returncode}): {stderr}"
            )
        report = json.loads(stdout)
        report["invocation"] = [
            sys.executable,
            os.path.relpath(os.path.abspath(__file__), REPO_DIR),
            "--reload-project",
            os.path.relpath(project_path, REPO_DIR),
        ]
        report["stderr"] = stderr.splitlines()
        reports.append(report)
        write_json(
            os.path.join(fixture_dir, f"reload-process-{index}.json"), report
        )

    if len({report["process_id"] for report in reports}) != 2:
        raise AssertionError("reload attestations do not identify two fresh processes")
    if any(report["process_id"] == os.getpid() for report in reports):
        raise AssertionError("reload was not isolated from the generator process")
    if any(report["load_returned_new_project"] for report in reports):
        raise AssertionError("a reload process treated project.json as a new project")
    if any(not report["git"]["head_matches_expected_feature_sha"] for report in reports):
        raise AssertionError("reload process did not execute at the feature SHA")
    if any(not report["git"]["tracked_feature_paths_clean"] for report in reports):
        raise AssertionError("reload process saw production/test drift from feature SHA")

    once_transform = tuple(reports[0]["text_transform"])
    twice_transform = tuple(reports[1]["text_transform"])
    twice = TextBlock(**reports[1]["block"])

    scene = QGraphicsScene()
    scene.setSceneRect(0, 0, 1100, 380)
    for index, (label, block) in enumerate(
        (("before save", original), ("after two reloads", twice))
    ):
        block_copy = copy.deepcopy(block)
        block_copy.adjust_pos(index * 520, 0)
        item = TextBlkItem(block_copy)
        scene.addItem(item)
        outline = QGraphicsPolygonItem(item.visual_polygon_in_scene())
        outline.setPen(QPen(QColor(26, 123, 225, 190), 1.5, Qt.PenStyle.DashLine))
        outline.setZValue(100)
        scene.addItem(outline)
        caption = scene.addText(
            f"{label}: {block.fontformat.text_transform}", QFont("Arial", 10)
        )
        caption.setDefaultTextColor(QColor(36, 50, 69))
        caption.setPos(45 + index * 520, 275)

    if original.fontformat.text_transform != once_transform:
        raise AssertionError("first reload changed the canonical transform")
    if once_transform != twice_transform:
        raise AssertionError("second reload changed the canonical transform")
    image = render_scene(
        scene,
        1140,
        470,
        "Canonical save / restart / reload",
        "Two fresh Python processes read project.json from disk; schema v1 and H/V/A remain exact.",
    )
    save_image(image, "save-reload.png")
    return {
        "project_saved_with": "ProjImgTrans.save()",
        "portable_relative_directory": saved_payload["directory"] == ".",
        "real_page_image": os.path.relpath(page_path, ARTIFACT_DIR),
        "project_path": os.path.relpath(project_path, ARTIFACT_DIR),
        "schema_v1": saved_payload["text_transform_schema_version"] == 1,
        "legacy_aliases_absent": aliases_absent,
        "fresh_reload_process_ids": [report["process_id"] for report in reports],
        "fresh_processes_read_from_disk": all(
            report["read_from_disk"] for report in reports
        ),
        "feature_sha_attested_in_each_process": all(
            report["git"]["head_matches_expected_feature_sha"]
            for report in reports
        ),
        "transform_before": list(original.fontformat.text_transform),
        "transform_reload_1": list(once_transform),
        "transform_reload_2": list(twice_transform),
        "exact_transform_round_trip": (
            original.fontformat.text_transform
            == once_transform
            == twice_transform
        ),
    }


def copy_migration_fixtures():
    destination = os.path.join(ARTIFACT_DIR, "migration-fixtures")
    os.makedirs(destination, exist_ok=True)
    fixture_root = "tests/fixtures/text_transform"
    names = [
        path.strip()
        for path in git_command(
            "ls-tree", "-r", "--name-only", FEATURE_SHA, "--", fixture_root
        ).stdout.splitlines()
        if path.strip()
    ]
    expected_targets = {
        os.path.relpath(path, fixture_root).replace(os.sep, "/") for path in names
    }
    for root, _directories, files in os.walk(destination):
        for filename in files:
            existing = os.path.relpath(
                os.path.join(root, filename), destination
            ).replace(os.sep, "/")
            if existing not in expected_targets and existing != "manifest.json":
                os.remove(os.path.join(root, filename))
    manifest = {
        "feature_sha": FEATURE_SHA,
        "source": f"git object {FEATURE_SHA}:{fixture_root}",
        "files": {},
    }
    for repo_path in names:
        name = os.path.relpath(repo_path, fixture_root)
        if name.startswith(".."):
            raise AssertionError(f"fixture escaped source root: {repo_path}")
        blob_command = [
            "git",
            "-c",
            f"safe.directory={REPO_DIR.replace(os.sep, '/')}",
            "-C",
            REPO_DIR,
            "show",
            f"{FEATURE_SHA}:{repo_path}",
        ]
        blob = subprocess.run(blob_command, check=True, capture_output=True).stdout
        target = os.path.join(destination, name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as output:
            output.write(blob)
        blob_sha256 = hashlib.sha256(blob).hexdigest()
        copied_sha256 = sha256_file(target)
        if copied_sha256 != blob_sha256:
            raise AssertionError(f"copied fixture differs from committed blob: {repo_path}")
        manifest["files"][name.replace(os.sep, "/")] = {
            "repo_path": repo_path,
            "sha256": copied_sha256,
            "byte_count": len(blob),
            "matches_feature_sha_blob": True,
        }
    write_json(os.path.join(destination, "manifest.json"), manifest)
    return manifest


def junit_totals(path):
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for suite in suites:
        for key in totals:
            totals[key] += int(suite.attrib.get(key, 0))
    totals["passed_including_subtests"] = (
        totals["tests"]
        - totals["failures"]
        - totals["errors"]
        - totals["skipped"]
    )
    totals["sha256"] = sha256_file(path)
    return totals


def junit_failure_ids(path):
    root = ElementTree.parse(path).getroot()
    return sorted(
        f"{case.attrib.get('classname')}::{case.attrib.get('name')}"
        for case in root.iter("testcase")
        if case.find("failure") is not None or case.find("error") is not None
    )


def collect_environment():
    distributions = {}
    for name in (
        "numpy",
        "qtpy",
        "PyQt5",
        "PyQt5-Qt5",
        "PyQt6",
        "PyQt6-Qt6",
        "PySide6",
        "pytest",
        "pytest-subtests",
    ):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "working_directory": os.getcwd(),
        "qt_api": os.environ.get("QT_API"),
        "qt_qpa_platform": os.environ.get("QT_QPA_PLATFORM"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "qt_runtime_version": qVersion(),
        "distributions": distributions,
    }


def write_raw_verification_logs():
    result_dir = os.path.join(ARTIFACT_DIR, "test-results")
    feature_env = {
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "QT_QPA_PLATFORM": "offscreen",
        "QT_API": "pyqt6",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    changed_python = git_command(
        "diff",
        "--name-only",
        f"{UPSTREAM_SHA}...{FEATURE_SHA}",
        "--",
        "*.py",
    ).stdout.splitlines()
    changed_production = [
        path for path in changed_python if path.startswith("ballontranslator/")
    ]
    pycompile_code = """
import os
import py_compile
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix="pr1238-pycompile-") as output_dir:
    for index, path in enumerate(sys.argv[1:]):
        py_compile.compile(
            path,
            cfile=os.path.join(output_dir, f"{index}.pyc"),
            doraise=True,
        )
print("compiled", len(sys.argv) - 1, "changed production Python files")
print(*sys.argv[1:], sep="\\n")
""".strip()
    git_prefix = [
        "git",
        "-c",
        f"safe.directory={REPO_DIR.replace(os.sep, '/')}",
        "-C",
        REPO_DIR,
    ]
    forbidden_pattern = (
        "VisualLineTransform|LayoutScaleContract|TransformedPaintCache|"
        "effective_size_pt|effective_stretch|_source_to_visual|_visual_to_source|"
        "_normalize_doc_fonts_for_geometric_scale"
    )
    static_records = [
        run_recorded(
            "py_compile changed production Python",
            [sys.executable, "-c", pycompile_code, *changed_production],
            cwd=REPO_DIR,
            environment=feature_env,
            expected_exit_codes=(0,),
        ),
        run_recorded(
            "git diff --check exact upstream to feature",
            [*git_prefix, "diff", "--check", f"{UPSTREAM_SHA}...{FEATURE_SHA}"],
            cwd=REPO_DIR,
            environment=feature_env,
            expected_exit_codes=(0,),
        ),
        run_recorded(
            "forbidden identifier grep (exit 1 means zero matches)",
            ["rg", "-n", forbidden_pattern, "ballontranslator", "tests"],
            cwd=REPO_DIR,
            environment=feature_env,
            expected_exit_codes=(1,),
        ),
        run_recorded(
            "audit font geometry mutations in changed production files",
            [
                "rg",
                "-n",
                "setStretch|setFontStretch|pointSizeF.*\\*|setPointSizeF.*(scale|factor)",
                *changed_production,
            ],
            cwd=REPO_DIR,
            environment=feature_env,
            expected_exit_codes=(0, 1),
        ),
        run_recorded(
            "audit paint/effect document clone paths",
            [
                "rg",
                "-n",
                "QTextDocument\\(|clone\\(|drawContents\\(|toPixmap\\(",
                "ballontranslator/ui/scene_textlayout.py",
                "ballontranslator/ui/textitem.py",
            ],
            cwd=REPO_DIR,
            environment=feature_env,
            expected_exit_codes=(0, 1),
        ),
        run_recorded(
            "audit legacy serialization names",
            [
                "rg",
                "-n",
                "rich_text_transform_version|italic_angle",
                *changed_production,
            ],
            cwd=REPO_DIR,
            environment=feature_env,
            expected_exit_codes=(0, 1),
        ),
        run_recorded(
            "audit shear/tangent formulas",
            ["rg", "-n", "math\\.tan|\\.shear\\(", *changed_production],
            cwd=REPO_DIR,
            environment=feature_env,
            expected_exit_codes=(0,),
        ),
        run_recorded(
            "audit reconstruction and broad except sites",
            [
                "rg",
                "-n",
                "QTextDocument\\(|QGraphicsScene\\(|TextAdvancedFormatPanel\\(|except\\s*:",
                *changed_production,
            ],
            cwd=REPO_DIR,
            environment=feature_env,
            expected_exit_codes=(0, 1),
        ),
    ]
    if static_records[2]["stdout_nonempty_line_count"] != 0:
        raise AssertionError("forbidden identifiers were found")
    if static_records[6]["stdout_nonempty_line_count"] != 1:
        raise AssertionError("expected exactly one shear/tangent implementation")
    write_raw_log(os.path.join(result_dir, "static-checks.log"), static_records)

    doctest_code = """
import doctest
import sys
from ballontranslator.utils import fontformat
from ballontranslator.ui import text_transform

failed = 0
for module in (fontformat, text_transform):
    result = doctest.testmod(module, verbose=True)
    print(f"PACKAGE_AWARE_RESULT {module.__name__} attempted={result.attempted} failed={result.failed}")
    failed += result.failed
sys.exit(1 if failed else 0)
""".strip()
    doctest_record = run_recorded(
        "package-aware doctest for changed core modules",
        [sys.executable, "-c", doctest_code],
        cwd=REPO_DIR,
        environment=feature_env,
        expected_exit_codes=(0,),
    )
    for expected in (
        "ballontranslator.utils.fontformat attempted=4 failed=0",
        "ballontranslator.ui.text_transform attempted=5 failed=0",
    ):
        if expected not in doctest_record["stdout"]:
            raise AssertionError(f"missing doctest result: {expected}")
    write_raw_log(
        os.path.join(result_dir, "package-aware-doctest.log"), [doctest_record]
    )

    overlay_env = {
        "PYTHONPATH": rf"F:\ballon\.pr1238-test-deps;{BASELINE_DIR}",
        "QT_QPA_PLATFORM": "offscreen",
        "QT_API": "pyqt6",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    overlay_collection = run_recorded(
        "feature test overlay on exact upstream: full collection",
        [sys.executable, "-m", "pytest", "-q"],
        cwd=OVERLAY_DIR,
        environment=overlay_env,
        expected_exit_codes=(2,),
    )
    if "9 errors" not in overlay_collection["stdout"]:
        raise AssertionError("overlay full collection did not report 9 errors")
    write_raw_log(
        os.path.join(result_dir, "upstream-overlay-collection.log"),
        [overlay_collection],
    )
    overlay_collectable = run_recorded(
        "feature test overlay on exact upstream: collectable UI and shape files",
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "test_text_transform_panel_integration.py",
            "test_text_transform_shape_control.py",
        ],
        cwd=OVERLAY_DIR,
        environment=overlay_env,
        expected_exit_codes=(1,),
    )
    if "20 failed, 1 passed" not in overlay_collectable["stdout"]:
        raise AssertionError("overlay collectable run did not report 20 failed, 1 passed")
    write_raw_log(
        os.path.join(result_dir, "upstream-overlay-collectable.log"),
        [overlay_collectable],
    )
    return {
        "static_checks": {
            "changed_production_python_count": len(changed_production),
            "forbidden_identifier_matches": 0,
            "shear_tangent_formula_matches": 1,
        },
        "doctest": {"fontformat": "4/4", "text_transform": "5/5"},
        "overlay": {
            "full_collection_exit_code": overlay_collection["exit_code"],
            "collection_errors": 9,
            "collectable_exit_code": overlay_collectable["exit_code"],
            "collectable_result": "20 failed, 1 passed",
        },
    }


def write_verification_logs():
    result_dir = os.path.join(ARTIFACT_DIR, "test-results")
    provenance = collect_git_provenance()
    if not provenance["head_matches_expected_feature_sha"]:
        raise AssertionError("evidence generation HEAD is not the feature SHA")
    if not provenance["tracked_feature_paths_clean"]:
        raise AssertionError("production/test files differ from the feature SHA")
    environment = collect_environment()
    python_vv = run_checked([sys.executable, "-VV"]).stdout
    pip_freeze = run_checked([sys.executable, "-m", "pip", "freeze", "--all"]).stdout
    write_json(os.path.join(result_dir, "evidence-environment.json"), environment)
    write_json(os.path.join(result_dir, "git-provenance.json"), provenance)
    with open(os.path.join(result_dir, "python-vv.txt"), "w", encoding="utf-8") as output:
        output.write(python_vv.rstrip() + "\n")
    with open(os.path.join(result_dir, "pip-freeze.txt"), "w", encoding="utf-8") as output:
        output.write(pip_freeze.rstrip() + "\n")
    with open(
        os.path.join(result_dir, "evidence-generation-command.txt"),
        "w",
        encoding="utf-8",
    ) as output:
        output.write(f"cwd={os.getcwd()}\n")
        output.write(f"executable={sys.executable}\n")
        output.write("argv=" + json.dumps(sys.argv, ensure_ascii=False) + "\n")
    return {"environment": environment, "git": provenance}


def build_generation_report(assertions, verification):
    screenshots = (
        "controls.png",
        "horizontal-vertical.png",
        "editing-canvas-export.png",
        "extreme-effects.png",
        "save-reload.png",
    )
    junit_names = (
        "focused-pyqt5.xml",
        "focused-pyqt6.xml",
        "focused-pyside6.xml",
        "full-feature-pyqt6.xml",
        "full-upstream-pyqt6.xml",
    )
    junit = {
        name: junit_totals(os.path.join(ARTIFACT_DIR, "test-results", name))
        for name in junit_names
    }
    focused_expected = all(
        junit[name]["tests"] == 286
        and junit[name]["failures"] == 0
        and junit[name]["errors"] == 0
        and junit[name]["skipped"] == 0
        for name in junit_names[:3]
    )
    baseline_same_failure_count = (
        junit["full-feature-pyqt6.xml"]["failures"]
        == junit["full-upstream-pyqt6.xml"]["failures"]
        == 7
        and junit["full-feature-pyqt6.xml"]["errors"]
        == junit["full-upstream-pyqt6.xml"]["errors"]
        == 0
    )
    feature_failure_ids = junit_failure_ids(
        os.path.join(ARTIFACT_DIR, "test-results", "full-feature-pyqt6.xml")
    )
    upstream_failure_ids = junit_failure_ids(
        os.path.join(ARTIFACT_DIR, "test-results", "full-upstream-pyqt6.xml")
    )
    baseline_same_failure_ids = feature_failure_ids == upstream_failure_ids
    if not focused_expected:
        raise AssertionError("focused JUnit totals do not match the recorded result")
    if not baseline_same_failure_count:
        raise AssertionError("full-suite JUnit failure totals do not match baseline")
    if not baseline_same_failure_ids:
        raise AssertionError("full-suite JUnit failure case IDs differ from baseline")
    report = {
        "feature_sha": FEATURE_SHA,
        "generator": os.path.relpath(os.path.abspath(__file__), ARTIFACT_DIR),
        "assertions": {
            **assertions,
            "focused_junit_totals_match_record": focused_expected,
            "feature_and_upstream_have_same_full_suite_failure_count": (
                baseline_same_failure_count
            ),
            "feature_and_upstream_have_same_full_suite_failure_case_ids": (
                baseline_same_failure_ids
            ),
            "feature_sha_matches_git_head": verification["git"][
                "head_matches_expected_feature_sha"
            ],
            "production_and_tests_match_feature_sha": verification["git"][
                "tracked_feature_paths_clean"
            ],
        },
        "screenshots": {
            name: {"sha256": sha256_file(os.path.join(ARTIFACT_DIR, name))}
            for name in screenshots
        },
        "junit": junit,
        "full_suite_failure_case_ids": feature_failure_ids,
        "notes": [
            "JUnit 'tests' includes pytest-subtests cases; the prose report separates "
            "ordinary tests from passed subtests.",
            "Full-suite baseline identity is asserted by matching JUnit classname/name "
            "pairs; messages are reported separately and are not normalized here.",
            "GUI screenshots are offscreen automated evidence, not manual GUI passes.",
        ],
    }
    write_json(
        os.path.join(ARTIFACT_DIR, "test-results", "generation-report.json"),
        report,
    )
    return report


def main():
    app = QApplication.instance() or QApplication([])
    for font_path in (
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\malgun.ttf",
    ):
        QFontDatabase.addApplicationFont(font_path)
    app.setFont(QFont("Malgun Gothic", 9))
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    raw_verification = write_raw_verification_logs()
    verification = write_verification_logs()
    controls_evidence(app)
    horizontal_vertical_evidence()
    editing_assertions = editing_export_evidence()
    extreme_assertions = extreme_effects_evidence()
    reload_assertions = save_reload_evidence()
    migration_manifest = copy_migration_fixtures()
    report = build_generation_report(
        {
            "editing_export": editing_assertions,
            "extreme_effects": extreme_assertions,
            "save_reload": reload_assertions,
            "migration_fixtures": {
                "materialized_from_feature_sha": True,
                "committed_blob_count": len(migration_manifest["files"]),
                "all_destination_hashes_match": all(
                    item["matches_feature_sha_blob"]
                    for item in migration_manifest["files"].values()
                ),
            },
            "raw_verification_logs": raw_verification,
        },
        verification,
    )
    print(
        json.dumps(
            {
                "feature_sha": FEATURE_SHA,
                "artifact_dir": ARTIFACT_DIR,
                "generation_report": os.path.join(
                    ARTIFACT_DIR, "test-results", "generation-report.json"
                ),
                "all_assertions_passed": all(
                    bool(value) if isinstance(value, bool) else True
                    for value in report["assertions"].values()
                ),
                "screenshots": [
                    "controls.png",
                    "horizontal-vertical.png",
                    "editing-canvas-export.png",
                    "extreme-effects.png",
                    "save-reload.png",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--reload-project":
        # ASCII-only transport keeps the JSON lossless even when a Windows
        # console or pipe inherits a non-UTF-8 code page.
        print(json.dumps(reload_project_report(sys.argv[2]), ensure_ascii=True))
    else:
        main()
