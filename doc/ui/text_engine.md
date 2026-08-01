# Text engine

Start here before changing text layout, editing, effects, geometry, or export.
The code and tests are authoritative; this guide identifies ownership and the
cross-file invariants that are easiest to break. For Slant, Perspective,
Curvature, Grid, or Glyph Slant, continue with
[Composable text transforms](text_transforms.md).

## System and owners

```text
Project JSON
  -> TextBlock + FontFormat                 persistent state
  -> TextBlkItem + QTextDocument            live text and editing
  -> horizontal / vertical document layout shaping and placement
  -> TextEffectRenderer                     fill, stroke, shadow, gradient
  -> TextItemGeometryController             bounds and visual mapping
  -> QGraphicsScene                         interaction, view, export
```

| Concern | Owner | Main files |
| --- | --- | --- |
| Block text, logical rectangle, angle, metadata | `TextBlock` | [`utils/textblock.py`](../../ballontranslator/utils/textblock.py) |
| Persistent typography and transforms | `FontFormat` | [`utils/fontformat.py`](../../ballontranslator/utils/fontformat.py) |
| Live Qt integration | `TextBlkItem` | [`ui/textitem.py`](../../ballontranslator/ui/textitem.py) |
| Horizontal and vertical layout | `SceneTextLayout` subclasses | [`ui/scene_textlayout.py`](../../ballontranslator/ui/scene_textlayout.py) |
| Fill, stroke, shadow, gradient, raster bounds | `TextEffectRenderer` | [`ui/text_effects/renderer.py`](../../ballontranslator/ui/text_effects/renderer.py) |
| Derived geometry and visual/input mapping | `TextItemGeometryController` | [`ui/text_item_geometry.py`](../../ballontranslator/ui/text_item_geometry.py) |
| Scene geometry overlays | `TextBlkShapeControl`, `TextGridTransformControl` | [`ui/texteditshapecontrol.py`](../../ballontranslator/ui/texteditshapecontrol.py), [`ui/text_grid_control.py`](../../ballontranslator/ui/text_grid_control.py) |
| Scene items, paired editors, undo integration | `SceneTextManager` | [`ui/scenetext_manager.py`](../../ballontranslator/ui/scenetext_manager.py), [`ui/textedit_commands.py`](../../ballontranslator/ui/textedit_commands.py) |
| Formatting UI | `FontFormatPanel`, `TextAdvancedFormatPanel`, `TextTransformPanel` | [`ui/text_panel.py`](../../ballontranslator/ui/text_panel.py), [`ui/text_advanced_format.py`](../../ballontranslator/ui/text_advanced_format.py), [`ui/text_transform_panel.py`](../../ballontranslator/ui/text_transform_panel.py) |

`TextBlkItem` is the Qt-facing integration point, not the owner of every
subsystem. Keep Qt virtual methods and signals there, but extend the existing
layout, effect, geometry, or scene owner instead of adding a parallel path.

## State boundaries

There are three kinds of state:

- **Persistent model state:** `TextBlock` and `FontFormat`; this is what project
  JSON may store.
- **Live editing state:** `QTextDocument`, cursor, selection, IME preedit, and
  paired-editor synchronization.
- **Derived state:** layout records, effect padding, geometry mappings, preview
  values, pixmaps, and caches; never persist these.

Use `TextBlkItem.initTextBlock()` as the real construction path. It binds the
model, chooses writing mode, restores logical geometry, applies text and
formatting, then initializes derived owners. Tests should patch narrow instance
state rather than add constructor switches used only by tests.

Passive project loading follows the permissive recovery policy in `AGENTS.md`.
Live setters, rendering, and explicit saves may assume or require canonical
values.

Before save, page change, undo/redo, or scene replacement, resolve or cancel
pending edits and previews. Before removing items, release their effect, glyph,
and surface resources.

## Coordinate spaces

Name the space whenever geometry crosses an owner:

| Space | Meaning |
| --- | --- |
| Project/page | Persistent block rectangle on the page |
| Item-local logical | Text box before effects and visual transforms |
| Item-local source | Paint surface including effect padding |
| Item-local visual | Result after item-local visual mapping |
| Parent/scene | Position, rotation, and parent transforms applied |
| Device | View zoom, device scale, or export transform |

The persistent logical rectangle excludes effect padding. The source rectangle
includes it. Padding changes must preserve logical geometry while refreshing
layout surface, effects, dependent mappings, and scene bounds. Never write a
visual bounding box back as the logical model rectangle.

## Layout, painting, and interaction

Both writing modes share `SceneTextLayout` state such as available size,
effect padding, draw offsets, layout generation, optional delegated glyph
painting, and optional input mapping.

- Horizontal layout should continue using Qt shaping, wrapping, glyph runs, and
  cursor indices.
- Vertical layout owns its extra character orientation, punctuation, offsets,
  and column records.
- `VerticalTextDocumentLayout.reLayoutForResize()` has a width-only fast path:
  it translates settled columns when height and padding are unchanged. Height,
  padding, or minimum-width changes still require a full relayout.
- Shared layout changes must cover horizontal and vertical writing.

The conceptual paint order is:

```text
stroke / shadow background
  -> text fill or gradient
  -> editing UI: selection and cursor
  -> item selection and geometry guides
```

`TextEffectRenderer` owns stroke, shadow, gradient, and the padding they
require. Effect padding is derived layout state, not document content, and must
not create `QTextDocument` undo steps.

Qt's text control remains authoritative for shaping, cursor, selection, IME,
and ordinary hit testing. When source and visual geometry differ, adapt points
and rectangles at the geometry/layout boundary; do not implement a second text
editor.

Interactive rendering may use bounded lower-quality fallbacks to stay
responsive. Export must report an incomplete render instead of silently
omitting text.

## Editing, paired editors, and undo

Separate histories own separate state:

- `QTextDocument` owns content and rich-text edit steps.
- `TextEditCommand` and `TextItemEditCommand` bridge text edits to canvas
  history and the paired editor.
- Canvas `QUndoCommand`s own item geometry, formatting, and transforms.

Do not make an item-state command consume document undo steps, and do not send
paint-only or geometry-only changes to the paired editor as text edits.

One logical formatting action may emit several Qt signals. Wrap the internal
document work in one edit boundary, publish one user-visible command, and guard
undo/redo from recursively creating another command. IME preedit text remains
transient; only Qt's normal commit lifecycle should make it persistent.

## Geometry and transforms

Every `TextBlkItem`, including a neutral one, owns
`TextItemGeometryController`. It maintains the relationship among logical
rectangle, padded source rectangle, visual geometry, installed Qt transform,
input mapping, cache policy, and render resources.

`TextBlkShapeControl` owns resize and rotation. While one Grid transform on one
text block is selected, the global `TextGridTransformControl` replaces that
shape overlay and edits the selected stage's normalized control points. Both
read and write through `TextItemGeometryController`; transform features do not
create a parallel layout, renderer, or text editor. See
[Composable text transforms](text_transforms.md).

## Invalidation and performance

Update the owner whose input changed:

| Input | First derived owner |
| --- | --- |
| Plain/rich text or character format | Document and layout |
| Font metrics, spacing, writing mode | Layout |
| Stroke, shadow, gradient | Effect renderer |
| Effect extent | Padding and source rectangle, then geometry |
| Logical rectangle | Layout size and geometry controller |
| Position or rotation | Scene/item geometry |
| Visual transform parameters | Geometry controller |
| Item/page lifetime | Every item-owned cache |

Important rules:

- Make setters and refreshers idempotent; one settled change should not trigger
  duplicate layout, compilation, or cache rebuilds.
- Increment `layout_generation` only when layout geometry changes.
- Keep the neutral path native and cheap. Allocate specialized renderers only
  while their feature is active.
- Cache keys must include every input that changes output, but no unrelated
  state. Keep caches bounded by count and memory, namespaced by owner, and
  releasable at page/layout boundaries.
- Batch transient formatting or preview states and refresh once when settled.
- Keep interactive raster dimensions and quality bounded; use the appropriate
  settled/export tier afterward.
- Index lookup data and resolve it on demand rather than scanning all blocks,
  fragments, widgets, or cache entries for one item.

For rendering, preserve these invariants:

- Fill and effects use the same glyph geometry.
- Apply antialiasing on the painter that creates the actual vector, mask, or
  pixmap pixels; final smoothing cannot repair an aliased source.
- Interpolate transparent images in premultiplied-alpha form.
- Compose mappings first and resample the completed surface as few times as
  possible.
- Include ink, effect extent, antialiasing guard, and clear border in bounds.
- Preserve fragment-specific font, color, weight, and stroke parameters.
- Keep cursor, selection, and IME geometry visible over the final destination.

## Change workflow

Before editing this subsystem:

1. Trace the production construction, signal, and paint path.
2. Identify the persistent owner and each affected derived owner.
3. State the coordinate spaces crossing the change.
4. Preserve document history, canvas history, and paired-editor boundaries.
5. Check both writing modes and the neutral-to-active-to-neutral lifecycle.
6. Check editing, formatting, resize, rotation, effects, export, and page
   removal as applicable.
7. If performance can change, measure the complete user action, including Qt
   event delivery and cache warmth, not an isolated helper.

Prefer state, geometry, cache, and painter-boundary assertions over exact pixel
baselines, which vary with Qt, platform, fonts, scale, and warmed glyph caches.
Painting changes still need a themed-app visual pass.

```bash
python -m py_compile <touched-python-files>

QT_API=pyqt6 QT_QPA_PLATFORM=offscreen \
  /opt/miniconda3/envs/common/bin/python -m unittest \
  discover -s tests -p 'test_*.py'

git diff --check
```

Run the narrowest relevant test pattern first, then broaden in proportion to
the affected ownership boundaries.
