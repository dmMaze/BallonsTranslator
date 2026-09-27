# Draw panel

Read this before changing drawing tools, canvas gestures, inpaint scheduling,
or drawing save/export behavior. Drawing shares the scene, cursors, and history
routing with the [text engine](text_engine.md); changes must preserve both modes.

## Ownership

| Concern | Owner |
| --- | --- |
| Tool selection, settings, accumulated masks, completion handling | [`DrawingPanel`](../../ballontranslator/ui/drawingpanel.py) |
| Input, gesture state, cancellation, previews, history routing | [`CustomGV` and `Canvas`](../../ballontranslator/ui/canvas.py) |
| Brush/shape rasterization and drawing composition | [`image_edit.py`](../../ballontranslator/ui/image_edit.py) |
| Reversible drawing and image/mask edits | [`drawing_commands.py`](../../ballontranslator/ui/drawing_commands.py) |
| Inpainter preparation, scheduling, request validity | [`ModuleManager`](../../ballontranslator/ui/module_manager.py) |
| Saved tool preferences | [`DrawPanelConfig`](../../ballontranslator/utils/config.py) |
| Page changes and saving | [`MainWindow`](../../ballontranslator/ui/mainwindow.py), [`ProjImgTrans`](../../ballontranslator/utils/proj_imgtrans.py) |

Extend these owners rather than adding parallel gesture, composition, or
scheduling paths.

## State transitions

The selected tool, active gesture, retained mask, and outstanding inpaint request
have different lifetimes. A completed Ctrl-held mask survives mouse release;
an inpaint request can finish after the user changes tools or pages. A selected
tool can also be temporarily disabled while work is pending. Do not infer one
state from another.

Review press, move, release, cancellation, and asynchronous completion together.
Small changes to one handler can otherwise affect a later, unrelated action.

| Boundary | Contract to preserve |
| --- | --- |
| Mouse release | Only the initiating button can finish a gesture. A late release must not reuse a retained mask or turn it into an erase operation. |
| Tool or brush-mode change | Clear the old gesture and incompatible pending state without forgetting physically held modifiers. Broad canvas cleanup is not interchangeable with tool cleanup. |
| Modifier release | Ctrl accumulates inpaint masks as well as constraining Shift segments. Releasing it during an active stroke must wait for mouse release before submission. |
| Escape, history changes, mode exit, hide/deactivation, page replacement | Cancel applicable gestures and invalidate their anchors so later input cannot commit stale work. Application shortcuts and scene handlers must agree on cancellation. |
| Inpaint completion or failure | Restore input only for the tool that is still selected and visible. A temporary busy state must not discard a valid line anchor; completion must not reactivate a tool the user left. |
| Leaving the view | Hide guides, and keep modifier-key events from recreating them outside the canvas. |

Keep brush, shape-fill, and text-creation gestures independent. Paint and eraser
anchors must not connect to each other, and magic wand must remain outside the
brush-segment path. Coordinate conversion must agree between mouse events,
keyboard-driven previews, and committed pixels at every zoom level.

## Asynchronous work

Drawing requests use their own inpainter/profile selection while sharing module
resources with the full-page pipeline. Keep requests on the existing scheduler:
freeze settings and input when submitted, then validate the source before both
starting and applying work. Page identity alone is insufficient because undo
can change pixels in the same image object.

Tool/page changes and late worker results must be safe in either order. Discard
obsolete results without terminating native inference. Preparation completion
is not necessarily thread completion; preserve the continuation path that waits
for the worker to become available.

Preview work has a separate lifetime from inference. Invalidate obsolete previews
when their inputs change, avoid copying entire pages on every pointer movement,
and keep workers independent of widget lifetime.

## History and persistence

Pen and shape edits belong to the drawing layer; inpaint and restoration belong
to the page image and mask. Keep those targets separate and route commands through
the canvas so undo and dirty-state accounting stay consistent. Shape fill must
not acquire an inpainting dependency.

Display and saving consume the same drawing composite. Changes must invalidate
it even when an existing image is edited in place. Guides must be excluded from
export, but hiding a guide does not cancel a gesture or remove its live pixels.

Review opacity must not alter saved pixels; intrinsic stroke/fill alpha must
survive saving. Check both the saved inpainted image and final result, and restore
temporary render state even when export fails. Persist tool preferences, not
anchors, pressed buttons, or previews.

## Verification

Follow [repository verification](../../AGENTS.md#verification). Exercise real
input sequences, including both release orders for overlapping mouse buttons,
held modifiers across tool changes, cancellation during a stroke, and late
inpaint completion after page/history changes. Include save/reload and export
failure paths when changing drawing pixels or render state.

| Area | Tests |
| --- | --- |
| Brush and shape gestures | `tests/test_brush_lines.py`, `tests/test_shape_fill.py` |
| Stroke cleanup and cursor handoff | `tests/test_canvas_stroke_lifecycle.py`, `tests/test_canvas_cursor_lifecycle.py` |
| Inpaint scheduling and drawing profiles | `tests/test_canvas_inpaint_lifecycle.py`, `tests/test_drawing_inpainter.py` |
| Wand selection and preview lifetime | `tests/test_inpaint_magic_wand.py`, `tests/test_magic_wand_canvas.py` |
| Configuration and save/export | `tests/test_drawpanel_config.py`, `tests/test_editing_layer_opacity.py` |

Shared scene-event changes also need coverage for path ordering, text alpha-mask
editing, and text-transform undo. Verify binding-sensitive behavior under both
PyQt5 and PyQt6; report any native behavior that offscreen tests cannot establish.
