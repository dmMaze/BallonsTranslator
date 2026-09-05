# Text engine

Start here before changing text layout, editing, effects, geometry, or export.
The code and tests are authoritative; this guide records ownership and the
cross-file contracts that are easy to break. Continue with
[Text layout](text_layout.md) for writing-mode behavior,
[Text effects](text_effects.md) for the effect stack, masks, and raster
lifecycle, and [Composable text transforms](text_transforms.md) for
Projective, Bend, Sine Wave, Grid, and Glyph Slant.

## Architecture and ownership

```text
Project JSON
  -> TextBlock + FontFormat.text_effects    persistent state
               + TextBlock.text_alpha_mask
  -> TextBlkItem + QTextDocument            live text and editing
  -> horizontal / vertical document layout shaping and placement
  -> TextEffectRenderer                     generated/filter stack + foreground paints
  -> TextItemGeometryController             bounds and visual mapping
  -> QGraphicsScene                         interaction, view, export
```

| Concern | Owner | Main files |
| --- | --- | --- |
| Block content, logical rectangle, angle, metadata, alpha mask | `TextBlock` | [`utils/textblock.py`](../../ballontranslator/utils/textblock.py), [`utils/text_alpha_mask.py`](../../ballontranslator/utils/text_alpha_mask.py) |
| Persistent typography, transforms, and immutable effect stack | `FontFormat` | [`utils/fontformat.py`](../../ballontranslator/utils/fontformat.py), [`utils/text_effects.py`](../../ballontranslator/utils/text_effects.py) |
| Live Qt integration | `TextBlkItem` and `QTextDocument` | [`ui/text_engine/item.py`](../../ballontranslator/ui/text_engine/item.py) |
| Rich-text import/export and annotations | `annotations.py` | [`ui/text_engine/annotations.py`](../../ballontranslator/ui/text_engine/annotations.py) |
| Shaping and placement | `SceneTextLayout` and its writing-mode subclasses | [`ui/text_engine/layout.py`](../../ballontranslator/ui/text_engine/layout.py), [`ui/text_engine/horizontal_layout.py`](../../ballontranslator/ui/text_engine/horizontal_layout.py), [`ui/text_engine/vertical_layout.py`](../../ballontranslator/ui/text_engine/vertical_layout.py) |
| Foreground paint, effects, raster bounds | `TextEffectRenderer` and effect helpers | [`ui/text_engine/effects/`](../../ballontranslator/ui/text_engine/effects/) |
| Reusable content-addressed raster assets | `ProjImgTrans` | [`utils/proj_imgtrans.py`](../../ballontranslator/utils/proj_imgtrans.py) |
| Bounds, transforms, and input mapping | `TextItemGeometryController` | [`ui/text_engine/geometry.py`](../../ballontranslator/ui/text_engine/geometry.py), [`ui/text_engine/transforms/`](../../ballontranslator/ui/text_engine/transforms/) |
| Alpha-mask brush input, preview, and undo | Canvas-owned `TextAlphaMaskEditSession` | [`ui/text_engine/effects/alpha_mask_edit_session.py`](../../ballontranslator/ui/text_engine/effects/alpha_mask_edit_session.py), [`ui/text_engine/editing/commands.py`](../../ballontranslator/ui/text_engine/editing/commands.py) |
| Paired editors and undo integration | `SceneTextManager` | [`ui/text_engine/editing/`](../../ballontranslator/ui/text_engine/editing/) |
| Formatting UI | Formatting panels and commands | [`ui/text_engine/formatting/`](../../ballontranslator/ui/text_engine/formatting/) |

`TextBlkItem` is the Qt-facing integration point, not the owner of every
subsystem. Extend the existing annotation, layout, effect, geometry, or scene
owner instead of adding a parallel path.

## State boundaries

- **Persistent state:** `TextBlock` and `FontFormat`; only this belongs in
  project JSON. Project-only Image effects live in each concrete item's
  `FontFormat.text_effects`; the immutable alpha-mask history is block-owned.
  Global formatting and presets strip project raster values.
- **Live editing state:** `QTextDocument`, cursor, selection, IME state, and
  paired-editor synchronization.
- **Derived state:** layout records, padding, visual mappings, previews,
  pixmaps, and caches; never persist it.

`TextBlkItem.initTextBlock()` is the real construction path. Passive project
loading follows the permissive recovery policy in `AGENTS.md`; live setters and
explicit writes may require canonical values. Before save, page replacement,
or item removal, settle pending edits and release item-owned render resources.

Font weights use the CSS/Qt 6 scale from `100` to `900`. Normalize legacy Qt 5
values only at the Qt/HTML boundary. Pass saved or UI-provided family names
through `qfont_with_family()` so Qt-unsafe names can use runtime aliases without
changing persisted names.

합성 굵기는 효과 카드가 아닌 `FontFormat.synthetic_bold`로 관리한다.
`none`, `rect`, `ellipse` 중 하나를 선택하고 `synthetic_bold_offset`으로
X/Y 확장량을 각각 글자 크기의 0~50%로 지정한다. 직사각형은 두 축을 분리해
확장하고, 타원은 타원형 범위를 사용한다. 두 축의 확장량이 같은 타원 모드는
원형 외곽선을 사용한다.
확장은 Stroke, Shadow, Glow, Filter, 마스크와 변환보다 먼저 정규 글리프
윤곽에 적용하며 레이아웃 치수와 선택한 글꼴은 바꾸지 않는다.

## Rich text and CSS extensions

This layer extends `QTextDocument` rich text; it is not Qt Style Sheets (QSS)
and is not a general browser CSS engine. `QTextDocument` remains the editable
model and shaper. `annotations.py` preserves the supported semantic HTML/CSS,
then restores application properties that Qt cannot represent directly.
Qt's limited HTML parser and serializer are a compatibility boundary: new
extensions must prove stable round-tripping under both supported Qt bindings.

| Feature | HTML/CSS representation | Application-only data |
| --- | --- | --- |
| Font weight | `font-weight` | None; Qt 5 values are normalized at the boundary |
| Emphasis | `text-emphasis-*` | None |
| Tate-chu-yoko | `text-combine-upright` | Stable group identity |
| Character spacing | `letter-spacing` | Exact multiplier |
| Paragraph line spacing | `line-height` | Exact distance-mode value |
| Ligatures and oldstyle figures | `font-variant-ligatures`, `font-variant-numeric` | None |
| Ruby/furigana | `<ruby>`, `<rt>`, `ruby-position` | Runtime container/unit identities are regenerated |

Keep all supported inline properties on one import/export path:

- `QTextDocument` character and block formats are the live source of truth;
  `TextBlock.rich_text` is the persisted HTML.
- Use standard markup where it can preserve the meaning. Add `data-btrans-*`
  only for exact behavior that CSS cannot express.
- Parse optional extensions defensively. Invalid or unsupported annotation data
  must not discard the base document.
- Clipboard copy/paste uses the same extended representation, with ordinary
  HTML and plain-text fallbacks.
- New CSS-backed features still need a live Qt property and, when Qt cannot
  render them, integration with the existing layout and rendering owners.

Ruby is annotation text, not editable document content. Group Ruby is one
indivisible base/reading unit; mono Ruby may break only between base/reading
pairs. Older releases do not understand this extension and may flatten `<rt>`
readings into editable text when they open and resave a project.

Character spacing is range-bound. Line spacing is paragraph-bound: a caret
formats its current paragraph, a selection formats the paragraphs it intersects,
and Enter inherits the current block format. `FontFormat` retains item-wide
defaults for old or empty rich text.

Common ligatures work on Qt 5 and Qt 6. Discretionary/contextual ligatures and
oldstyle figures require Qt 6.11's font-feature API; Qt 5 preserves those CSS
states without applying them. Letter spacing can suppress optional ligatures,
so shaping policy belongs with the layout rather than the UI or serializer.

Automatic tate-chu-yoko is a pipeline formatting pass owned by
[`pipeline_formatting.py`](../../ballontranslator/ui/text_engine/pipeline_formatting.py).
The translation and manual project actions trigger that same pass; they should
not duplicate its formatting rules.

## Layout, painting, and geometry

Horizontal and vertical layouts share engine-wide metrics and lifecycle state,
but each writing mode owns its flow and placement records. Keep only genuinely
shared helpers in `layout.py`; do not make it import its concrete subclasses.
See [Text layout](text_layout.md) for the detailed behavior contract.

Use these coordinate-space names consistently:

| Space | Meaning |
| --- | --- |
| Project/page | Persistent block rectangle on the page |
| Item-local logical | Text box before effects and visual transforms |
| Item-local source | Paint surface including effect padding |
| Item-local visual | Result after item-local visual mapping |
| Parent/scene | Item position, rotation, and parent transforms |
| Device | View zoom, device scale, or export transform |

The persistent logical rectangle excludes effect padding and visual overflow.
Never write a source or visual bounding box back into the model. Layout-owned
placement must be shared by fill, effects, annotations, cursor, selection, and
hit testing; adapting only one consumer creates visible drift or broken editing.

`FontFormat.text_effects` owns one immutable effect stack;
`TextBlock.text_alpha_mask` separately owns the item-specific Eraser. The effect
renderer builds the structural Gradient/Texture foreground, walks movable
Image/generated/Filter cards in panel application order around one canonical
glyph source, then applies Eraser and overall Opacity before handing one padded
surface to the geometry owner. Effects compose inside the item surface, never
against the page backdrop.

Project-only Texture and Image values share immutable `RasterAssetRef` data and
the `ProjImgTrans` import, validation, resolution, and bounded decode cache.
Passive loading preserves valid-but-missing references; interactive rendering
warns and bypasses them, while strict export fails rather than silently omitting
output. Global formatting and reusable presets strip project-only raster values
because they have no project asset registry.

Effect padding expands source paint bounds only. Ordinary hit testing stays on
the logical box plus layout-owned ink overhang. Qt remains authoritative for
shaping and editing feedback; selection and the deferred caret paint after
effects. Image is suppressed during native editing in both writing modes, while
ordinary Filters remain active.

Committed, preview, and export raster namespaces are separate, bounded derived
state. Requested-quality previews may promote on commit; Faster Preview uses a
non-promotable 0.5x scratch surface. Reshape omits effects during pointer motion
and rebuilds once settled. See [Text effects](text_effects.md) for ordering,
sources, persistence, preview/undo, mask editing, assets, caches, and extension
rules; see [Text filters](text_filters.md) for plug-in and tile contracts.

`TextItemGeometryController` owns the relationship among logical, source, and
visual geometry, installed transforms, input mapping, caches, and render
resources. Transform tools must use that owner rather than creating another
layout, renderer, or editor.

## Editing and undo

Separate histories own separate state:

- `QTextDocument` owns content and rich-text edit steps.
- `TextEditCommand` and `TextItemEditCommand` bridge document edits to canvas
  history and the paired editor.
- Canvas `QUndoCommand`s own item geometry, formatting, and transforms.

One logical action should publish one user-visible command. Guard undo/redo
against recursively creating commands, and never send paint-only or
geometry-only changes to the paired editor as text changes.

Qt positions and removal lengths are UTF-16 code units. Replay Qt's
`(position, charsRemoved, insertedText)` contract directly; do not infer ranges
from Python string length or glyph count. IME preedit remains transient until
Qt commits it.

## Invalidation and performance

Refresh from the first owner whose input changed:

| Input | First derived owner |
| --- | --- |
| Text, character format, paragraph format | Document and layout |
| Metrics, spacing, writing mode | Layout |
| Typed effect stack, paint, Image, or Overall Opacity | Effect renderer |
| Filter-only parameter preview | Effect renderer below-filter prefix and retained canonical/Stroke coverage |
| TextBlock alpha-mask replacement | Effect renderer mask generation |
| Effect extent or logical rectangle | Geometry controller after layout/effect update |
| Visual transform parameters | Geometry controller |
| Item/page lifetime | Every item-owned cache |

Keep refreshers idempotent, rebuild derived layout state as one settled
generation, and keep caches bounded and releasable. The neutral path stays
native and cheap. Batch previews and transient formatting so they refresh once
when settled. Optional acceleration must preserve the working fallback's
coordinates, rounding, and output, and must not compile on the Qt thread.

Dynamic Text Effect and Text Transform cards keep their natural height inside
`PanelArea.scrollContent`. Their panels may advertise a capped, content-driven
preferred height, but must not raise their minimum height with that content;
the containing window's current allocation takes precedence and overflow
scrolls. Mark newly inserted cards visible before measuring them, then use the
shared scroll-content height synchronizer so a constrained viewport neither
flattens cards nor enlarges the main window. Leave horizontal sizing to the
resizable scroll area so cards follow the viewport when a splitter moves.

Formatting ownership includes ordinary child controls and the parent chain of
their top-level Qt popups, such as nested `QMenu` and combo-box dropdowns.
Transient canvas-selection signals while those widgets are active must retain
the current text item and its card-local drafts; an actual target change still
settles pending edits and projects the new item's persisted state.

Multi-item Text Effect projection uses one explicit primary selection anchor
without reordering Transform targets. Matching is derived from committed stacks,
never persisted as a merged stack or Mixed values; preview and commit still
replace complete per-item stacks through one canvas undo command. See
[Text effects](text_effects.md) for the occurrence-matching contract.

## Change workflow

Before editing this subsystem:

1. Trace the production construction, signal, and paint path.
2. Identify the persistent owner and affected derived owners.
3. Name every coordinate-space boundary the change crosses.
4. Preserve document history, canvas history, and paired-editor behavior.
5. Check both writing modes and the neutral-to-active-to-neutral lifecycle.
6. Test the affected editing, geometry, effects, export, and cleanup paths.

Prefer state and relationship assertions over font-dependent pixel baselines.
Run the narrowest relevant tests first, then broaden in proportion to the
affected ownership boundaries. Layout lifetime, glyph-run, cursor, or painting
changes should be checked under both PyQt5 and PyQt6.

```bash
python -m py_compile <touched-python-files>
QT_QPA_PLATFORM=offscreen /opt/miniconda3/envs/common/bin/python -m unittest \
  discover -s tests -p 'test_*.py'
git diff --check
```
