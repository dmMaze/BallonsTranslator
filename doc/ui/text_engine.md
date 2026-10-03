# Text engine

Read this before changing text layout, editing, effects, geometry, or export.
It owns cross-subsystem contracts; detailed behavior belongs to these guides:

- [Text layout](text_layout.md): shaping, writing modes, spacing, and placement.
- [Text effects](text_effects.md): composition, masks, assets, and raster lifecycle.
- [Text filters](text_filters.md): filter plug-ins and tile contracts.
- [Text transforms](text_transforms.md): mapping, interaction, and surface warping.
- [Runtime font refresh](font_refresh.md): font identity, discovery, and invalidation.

## Architecture and ownership

```text
TextBlock + FontFormat                    persisted project state
  -> TextBlkItem + QTextDocument           live text and editing
  -> SceneTextLayout                      shaping and placement
  -> TextEffectRenderer                   padded source surface
  -> TextItemGeometryController           bounds and visual mapping
  -> QGraphicsScene                       interaction, view, export
```

| Concern | Owner |
| --- | --- |
| Content, logical rectangle, angle, and alpha mask | [`TextBlock`](../../ballontranslator/utils/textblock.py), [`text_alpha_mask.py`](../../ballontranslator/utils/text_alpha_mask.py) |
| Typography, transforms, and immutable effect stack | [`FontFormat`](../../ballontranslator/utils/fontformat.py), [`text_effects.py`](../../ballontranslator/utils/text_effects.py) |
| Qt integration and real construction path | [`TextBlkItem.initTextBlock()`](../../ballontranslator/ui/text_engine/item.py) |
| Rich-text import/export and annotations | [`annotations.py`](../../ballontranslator/ui/text_engine/annotations.py) |
| Shaping and placement | [`layout.py`](../../ballontranslator/ui/text_engine/layout.py), [`horizontal_layout.py`](../../ballontranslator/ui/text_engine/horizontal_layout.py), [`vertical_layout.py`](../../ballontranslator/ui/text_engine/vertical_layout.py) |
| Composition and raster bounds | [`effects/`](../../ballontranslator/ui/text_engine/effects/) |
| Content-addressed raster assets | [`ProjImgTrans`](../../ballontranslator/utils/proj_imgtrans.py) |
| Bounds, transforms, and input mapping | [`geometry.py`](../../ballontranslator/ui/text_engine/geometry.py), [`transforms/`](../../ballontranslator/ui/text_engine/transforms/) |
| Alpha-mask input and preview | [`TextAlphaMaskEditSession`](../../ballontranslator/ui/text_engine/effects/alpha_mask_edit_session.py) |
| Paired editors and canvas undo | [`editing/`](../../ballontranslator/ui/text_engine/editing/) |
| Formatting controls and commands | [`formatting/`](../../ballontranslator/ui/text_engine/formatting/) |

Extend the existing owner; `TextBlkItem` integrates these systems rather than
absorbing their implementation or creating parallel models/renderers.

## State and geometry

Only committed `TextBlock` and `FontFormat` state belongs in project JSON.
`QTextDocument`, cursor, selection, and IME belong to live editing. Layout records,
padding, mappings, previews, pixmaps, and caches are derived and never persisted.
Passive loading follows [AGENTS.md](../../AGENTS.md#changes-and-data-safety).
Scene-to-project saves publish the complete block list in one slice assignment,
preserving list identity. Translation workers snapshot that membership before
reading sources and translations; they must never observe a partly rebuilt list.

Use coordinate spaces explicitly:

| Space | Meaning |
| --- | --- |
| Project/page | Persistent block rectangle |
| Item-local logical | Text box before effects and visual transforms |
| Item-local source | Paint surface including effect padding |
| Item-local visual | Result of item-local visual mapping |
| Parent/scene | Item position, rotation, and parent transforms |
| Device | View zoom, device scale, or export transform |

The logical rectangle excludes effect padding and visual overflow. Never write
source/visual bounds into the persistent rectangle. Layout placement is shared
by paint, effects, annotations, cursor, selection, and hit testing. The geometry
controller owns mapping between spaces; tools must use it rather than adapting
only one consumer.

## Rich text and CSS extensions

`QTextDocument` remains the editable model and shaper. This is its rich-text
extension layer, not QSS or a general browser CSS engine. `annotations.py`
preserves supported semantic markup and restores properties Qt cannot represent.

| Feature | HTML/CSS representation | Application-only data |
| --- | --- | --- |
| Font weight | `font-weight` | Qt 5 normalization at the boundary |
| Emphasis | `text-emphasis-*` | None |
| Tate-chu-yoko | `text-combine-upright` | Stable group identity |
| Character spacing | `letter-spacing` | Exact multiplier |
| Paragraph spacing | `line-height` | Exact distance-mode value |
| Ligatures and oldstyle figures | `font-variant-ligatures`, `font-variant-numeric` | None |
| Ruby/furigana | `<ruby>`, `<rt>`, `ruby-position` | Regenerated runtime identities |

Document formats are the live source of truth; `TextBlock.rich_text` stores the
HTML. Prefer standard markup and reserve `data-btrans-*` for behavior CSS cannot
express. Clipboard and project persistence share this representation, with
ordinary HTML/plain-text clipboard fallbacks. Invalid optional annotations must
not discard the base document. Extensions need a Qt property, layout/render
integration where necessary, and round-trip checks under both bindings.

Ruby readings are annotations, not editable document text. Older releases may
flatten `<rt>` readings when resaving; layout behavior belongs to the layout
guide. [`pipeline_formatting.py`](../../ballontranslator/ui/text_engine/pipeline_formatting.py)
owns automatic tate-chu-yoko for both translation and manual project actions.

## Editing, preview, and undo

`QTextDocument` owns content and rich-text history. `TextEditCommand` and
`TextItemEditCommand` bridge it to the canvas and paired editor. Canvas commands
own geometry, formatting, effects, and transforms. One logical action creates
one user-visible command; undo/redo must not recursively create commands or
report paint/geometry changes as text changes.

Edit sessions own transient drafts and previews. Preview must not mutate
committed project values or consume document history; cancel restores committed
state, and commit publishes one command for its targets. Resolve pending work
before structural changes, save, undo/redo, target/page replacement, or teardown.
Commands must invoke the formatting panel's pending-edit resolver **before**
capturing or mutating state; waiting until stack insertion is too late.

`Canvas` owns save-state accounting through each `QUndoStack` clean marker.
Replacing an undo branch must invalidate its saved state. Edits merged by
`QTextDocument` into an existing command invalidate a clean marker at or beyond
that command; source, translation, and canvas editors share this rule. A merge
into an older document command below another canvas command conservatively
invalidates the saved marker until the next save. History
clearing preserves unsaved state unless the caller explicitly resets it for a
page/project replacement. Document owners reset their undo-step baseline when
Qt clears document history. Whole-style replay suppresses user-edit capture
while keeping layout and painting live; it must not push nested commands.
Text commands that also modify image pixels mark drawing dirty independently
on undo and redo; they must not retain commands owned by the drawing stack.

| Session | Specific boundary |
| --- | --- |
| [`move_session.py`](../../ballontranslator/ui/text_engine/editing/move_session.py) | Position-only movement; no transform stage. Cancel before zoom changes or competing geometry capture. |
| [`size_edit_session.py`](../../ballontranslator/ui/text_engine/formatting/size_edit_session.py) | Whole-item drags preview geometry, then commit rich-text sizes and bounds together. Text-selection edits remain numeric drafts; held drags cancel at save/target/history changes. |
| Effect/transform sessions | Complete-stack snapshots; matching targets and commands are defined in their owning guides. |

Formatting focus includes child controls and the parent chain of top-level Qt
popups. Transient selection signals while these are active retain the target and
drafts; actual target changes settle pending edits. Multi-item panels project the
primary selection without merging persisted state or reordering targets.

Qt positions and removal lengths are UTF-16 units. Replay
`(position, charsRemoved, insertedText)` rather than infer ranges from Python
string length or glyph count. IME preedit stays transient until Qt commits;
reset native composition before detaching the paired editor.

## Invalidation and verification

Refresh from the first owner whose input changed:

| Input | First derived owner |
| --- | --- |
| Text or document formats | Document and layout |
| Metrics, spacing, writing mode | Layout |
| Effect stack or alpha mask | Effect renderer |
| Effect extent or logical rectangle | Geometry after layout/effect update |
| Visual transform parameters | Geometry |
| Item/page lifetime | Every item-owned cache |

Publish one settled generation, batch transient edits, and keep refreshers
idempotent and caches bounded/releasable. Return to neutral must restore the
native path and release active-only resources. Optional acceleration preserves
fallback coordinates, rounding, and output and never compiles on the Qt thread.
Dynamic cards use `PanelArea` sizing so content scrolls instead of enlarging the
main window.

Follow [repository verification](../../AGENTS.md#verification). Trace construction,
signals, paint, persistent owners, and coordinate boundaries before editing.
Select focused suites from the topic guides; check both writing modes, paired
editors, undo, export, cleanup, and neutral/active transitions as affected. Prefer
state/relationship assertions over font-dependent pixel baselines. Layout
lifetime, shaping, cursor, or painting changes require both PyQt5 and PyQt6;
rendering/interaction changes also need a themed-app pass or an explicit limitation.
