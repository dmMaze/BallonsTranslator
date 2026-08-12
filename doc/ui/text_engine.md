# Text engine

Start here before changing text layout, editing, effects, geometry, or export.
The code and tests are authoritative; this guide identifies ownership and the
cross-file invariants that are easiest to break. Continue with
[Text layout](text_layout.md) for shaping, wrapping, vertical flow, spacing,
alignment, cursor, and hit geometry. For Projective, Bend, Sine Wave, Grid, or
Glyph Slant, continue with [Composable text transforms](text_transforms.md).

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

The implementation lives under `ui/text_engine/`: engine-wide item, layout,
geometry, effect, and shape-control boundaries stay at its root; paired-editor
coordination lives in `editing/`; format commands and panels live in
`formatting/`; pixel and glyph work lives in `rendering/`; composable transform
math, UI, and selected transform controls live in `transforms/`.

| Concern | Owner | Main files |
| --- | --- | --- |
| Block text, logical rectangle, angle, metadata | `TextBlock` | [`utils/textblock.py`](../../ballontranslator/utils/textblock.py) |
| Persistent typography and transforms | `FontFormat` | [`utils/fontformat.py`](../../ballontranslator/utils/fontformat.py) |
| Live Qt integration | `TextBlkItem` | [`ui/text_engine/item.py`](../../ballontranslator/ui/text_engine/item.py) |
| Inline rich-text annotations | `QTextDocument` character formats plus the semantic HTML boundary | [`ui/text_engine/annotations.py`](../../ballontranslator/ui/text_engine/annotations.py) |
| Horizontal and vertical layout | Shared `SceneTextLayout` plus writing-mode subclasses | [`ui/text_engine/layout.py`](../../ballontranslator/ui/text_engine/layout.py), [`ui/text_engine/horizontal_layout.py`](../../ballontranslator/ui/text_engine/horizontal_layout.py), [`ui/text_engine/vertical_layout.py`](../../ballontranslator/ui/text_engine/vertical_layout.py) |
| Fill, stroke, shadow, gradient, raster bounds | `TextEffectRenderer` | [`ui/text_engine/effect_renderer.py`](../../ballontranslator/ui/text_engine/effect_renderer.py), [`ui/text_engine/rendering/`](../../ballontranslator/ui/text_engine/rendering/) |
| Derived geometry and visual/input mapping | `TextItemGeometryController` | [`ui/text_engine/geometry.py`](../../ballontranslator/ui/text_engine/geometry.py) |
| Scene geometry overlays | `TextBlkShapeControl`, selected-transform controls | [`ui/text_engine/shape_control.py`](../../ballontranslator/ui/text_engine/shape_control.py), [`ui/text_engine/transforms/grid_control.py`](../../ballontranslator/ui/text_engine/transforms/grid_control.py), [`ui/text_engine/transforms/projective_control.py`](../../ballontranslator/ui/text_engine/transforms/projective_control.py) |
| Scene items, paired editors, undo integration | `SceneTextManager` | [`ui/text_engine/editing/manager.py`](../../ballontranslator/ui/text_engine/editing/manager.py), [`ui/text_engine/editing/commands.py`](../../ballontranslator/ui/text_engine/editing/commands.py), [`ui/text_engine/editing/widgets.py`](../../ballontranslator/ui/text_engine/editing/widgets.py) |
| Formatting UI | `FontFormatPanel`, `TextAdvancedFormatPanel`, `TextTransformPanel` | [`ui/text_engine/formatting/`](../../ballontranslator/ui/text_engine/formatting/), [`ui/text_engine/transforms/panel.py`](../../ballontranslator/ui/text_engine/transforms/panel.py) |

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

### Inline annotations

Qt character-format user properties are the live source of truth for emphasis,
tate-chu-yoko, character spacing, and future range-bound features such as ruby.
`TextBlock.rich_text` remains an HTML string. `annotations.py` extends Qt's
ordinary HTML with semantic inline `span` markup. Standard CSS carries
emphasis, tate-chu-yoko, and approximate external character spacing;
application-owned `data-*` attributes carry only the exact spacing multiplier
and tate group identity that CSS cannot represent. Extended output uses the
HTML5 doctype so those attributes remain standard HTML. HTML written before
this layer has none of these attributes and follows the same load path;
malformed optional values are discarded without losing the base document.

Tate-chu-yoko stores a stable group ID as well as the `all` enabled value. One
application or insertion-format session therefore remains one group even when
inherited character styling creates several Qt fragments, while adjacent
independent applications remain separate. CSS stores
`text-combine-upright: all`; `data-btrans-text-combine-id` stores that stable
identity. Its cell, overflow, cursor, hit-test, and paint behavior is specified
in [Text layout](text_layout.md).

Keep one shared inline range boundary: reserve stable live property IDs,
coalesce equal extension values, emit one semantic span per resulting text
segment, and restore all supported properties after Qt loads the ordinary
HTML. Give the existing layout/render owners each feature's metrics and paint
hook; do not introduce a parallel editable document model or a position side
table. Internal clipboard data uses the same extended representation, with
ordinary HTML and plain-text fallbacks.

Character spacing uses the same selection/insertion behavior as other inline
formats. Qt imports but does not export CSS `letter-spacing`, and the existing
value is a per-glyph-width multiplier rather than CSS's additive length. Each
saved range therefore uses standard `letter-spacing` in `em` as the closest
external representation plus `data-btrans-letter-spacing` for the exact
multiplier. `FontFormat.letter_spacing` remains the item-wide compatibility and
default value: rich text without the exact inline attribute is seeded from it
on load and gains explicit spans on the next save. The attribute's absence is
the pre-feature compatibility signal; there is no separate rich-text version.
Effect-document clones and internal clipboard insertion load the same inline
representation. Writing-mode behavior is specified in
[Text layout](text_layout.md).

`TextBlock.text_layout_version` versions item-wide layout semantics separately
from the versionless inline HTML extensions. Missing/version-zero vertical
blocks migrate to right alignment because that was their effective placement
before vertical alignment was implemented; current blocks preserve their saved
left, center, or right alignment. See [Text layout](text_layout.md).

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
painting, and optional input mapping. Horizontal layout keeps Qt shaping and
wrapping; vertical layout owns its additional orientation, punctuation, column,
whitespace, cursor, and hit-test records. Their detailed contract lives in
[Text layout](text_layout.md).

Keep only genuinely shared metrics and interaction-range helpers in
`layout.py`. Writing-mode flow, placement, and private records belong in their
concrete modules; the shared module must not import those subclasses back.

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

`TextBlkShapeControl` owns resize and item rotation. While one Grid or
Projective transform on one text block is selected, its global transform
controller replaces that shape overlay. Grid edits normalized control points;
Projective exposes a fixed-device-size 3D rotation gizmo. All controllers read
and write through `TextItemGeometryController`; transform features do not create
a parallel layout, renderer, or text editor. See
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
- `TextEffectRenderer` owns the device-scale-aware stroke/shadow raster cache
  for neutral and transformed text alike. Items with effects or delegated
  glyph painting bypass Qt's outer device-coordinate cache so the renderer
  sees the real view/export scale instead of stretching a hidden 1x cache.
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
