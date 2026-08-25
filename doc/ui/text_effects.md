# Text effects

Read [Text engine](text_engine.md) first. This guide is an orientation to the
text-effect subsystem: its fixed composition model, ownership boundaries, and
the lifecycle and cache contracts that are easy to break. The code and focused
tests are authoritative for individual controls and raster algorithms.

## Mental model

```text
QTextDocument + settled SceneTextLayout
  -> canonical glyph source, including Glyph Slant when active
  -> exterior Shadow / Long Shadow / Outer Glow
  -> Center and Outside Stroke
  -> canonical foreground or Gradient, unless Hollow
  -> Inside Stroke and interior Shadow / Inner Glow
  -> TextBlock-owned alpha mask
  -> Overall Opacity
  -> ordered global Text Transform
  -> item position and rotation
```

`TextEffectStack` is an ordered value, but it is not an arbitrary layer graph.
Each effect belongs to a fixed compiler phase. Order is retained within a
phase, with the first card visually topmost; cards cannot move across phase
boundaries. This keeps bounds, tiling, Hollow, and source-alpha behavior
deterministic.

## Owners

| Concern | Owner |
| --- | --- |
| Immutable effects, paints, stack helpers, and tolerant typed loading | [`utils/text_effects.py`](../../ballontranslator/utils/text_effects.py) |
| Style persistence, legacy migration, and compatibility views | [`utils/fontformat.py`](../../ballontranslator/utils/fontformat.py) |
| Block-owned alpha-mask persistence | [`utils/text_alpha_mask.py`](../../ballontranslator/utils/text_alpha_mask.py), [`utils/textblock.py`](../../ballontranslator/utils/textblock.py) |
| Composition, padding, raster policy, preview namespaces, and caches | [`ui/text_engine/effect_renderer.py`](../../ballontranslator/ui/text_engine/effect_renderer.py), [`ui/text_engine/rendering/`](../../ballontranslator/ui/text_engine/rendering/) |
| Selection-scoped effect preview and commit | [`ui/text_engine/effect_edit_session.py`](../../ballontranslator/ui/text_engine/effect_edit_session.py) |
| Panel projection and card controls | [`ui/text_engine/formatting/effects.py`](../../ballontranslator/ui/text_engine/formatting/effects.py), [`ui/text_engine/formatting/gradient_editor.py`](../../ballontranslator/ui/text_engine/formatting/gradient_editor.py) |
| Canvas brush input and mask undo | [`ui/text_engine/alpha_mask_edit_session.py`](../../ballontranslator/ui/text_engine/alpha_mask_edit_session.py) |
| Effect and mask undo commands | [`ui/text_engine/editing/commands.py`](../../ballontranslator/ui/text_engine/editing/commands.py) |
| Source/visual bounds and the final global mapping | [`ui/text_engine/geometry.py`](../../ballontranslator/ui/text_engine/geometry.py), [`ui/text_engine/transforms/`](../../ballontranslator/ui/text_engine/transforms/) |

The panel projects model values and emits edit requests; it does not own effect
state. `TextBlkItem` is the Qt boundary, while `TextEffectRenderer` owns the
completed effect surface. Add behavior at those existing owners instead of
introducing a second effect model, renderer, or edit session.

Effect and Transform cards share `BottomBorderComboBox`, the compact selector
used by the Run dialog. Stroke Position and Shadow/Glow Type follow the card
title. Fill stays in the left control column; Solid shows its swatch opposite,
while Gradient expands the shared stop editor below it. The editor orders its
square selected-stop color swatch and stop strip on one row, Opacity/Position
row, then angle-dial/numeric Angle and Scale row. Dragging the dial pointer uses
the renderer's screen-angle direction and retains the numeric editor for exact
input. The gradient band and square selected-stop picker are both 24 px high.
Compact Add/Remove icon buttons stack beside the complete strip-and-stops area.
Effect control rows use a consistent 8 px vertical gap. Angle directly controls
direction; its dial is 36 px, and
two-column parameter labels align to the left edge of their column. Labels keep
their natural width and a fixed local gap, while editors/selectors absorb spare
column width. Two-column rows use equal columns and an 8 px inter-column gap.
There is no redundant Flip action.

## Model and fixed phases

`FontFormat.text_effects` contains one immutable `TextEffectStack`. Its
`overall_opacity` applies to the completed item, while `effects` preserves the
semantic order of typed immutable values. Multiple Stroke, Shadow, and Glow
entries are allowed. Hollow and Gradient are unique.

| Effect | Phase and source | Important semantics |
| --- | --- | --- |
| Stroke | Stroke phase | Width is relative to font size. Center splits the full band across the glyph edge; Outside and Inside clip a full-width outline to the corresponding side. New and migrated legacy strokes default to Outside. |
| Shadow | Exterior for Drop and Long; interior for Inner | Exterior Shadow uses the Stroke-inclusive silhouette. Inner Shadow uses canonical glyph alpha and is suppressed by Hollow. |
| Glow | Exterior for Outer; interior for Inner | Outer Glow uses the Stroke-inclusive silhouette. Inner Glow uses canonical glyph alpha and is suppressed by Hollow. |
| Gradient | Foreground, unique | Overwrites canonical foreground RGB and multiplies foreground alpha by stop opacity. Stroke and generated effects continue using their earlier canonical/source alpha. There is no separate effect opacity. |
| Hollow | Foreground modifier, unique | Removes the canonical face, Gradient, and interior phase while retaining Stroke and exterior output. It is a toggle, not an independent painted layer. |

Stroke, Shadow, and Glow use either `SolidPaint` or `LinearGradientPaint`.
Gradient always uses a linear gradient. A gradient contains two to 32
ordered stops; RGB and stop opacity interpolate independently. Its angle and
scale are defined against the complete unpadded logical text rectangle, not
each glyph, effect layer, or tile, so writing modes and render paths agree.
An enabled Gradient remains active when every stop is transparent because
that state intentionally erases the foreground face.

Only the `normal` blend mode is valid today. Do not expose another mode until
its destination is specified: blending inside the isolated text surface and
blending against the page backdrop are different persistence, preview, tiling,
and export contracts.

Neutral effects stay in model and panel state but are skipped by rendering.
Keep that neutral test explicit when adding a type; disabled, zero-opacity, or
zero-extent values must not allocate effect surfaces when they leave output
unchanged. An enabled transparent Gradient is the intentional exception above.

## Composition and render paths

The renderer captures canonical glyph pixels once, including the current
layout-owned placement and any Glyph Slant renderer. It then compiles the
effective stack as follows:

1. Build canonical foreground alpha and, when exterior effects and Stroke are
   both active, a Stroke-inclusive exterior silhouette.
2. Paint exterior Shadow and Glow entries back-to-front within their retained
   stack order.
3. For Hollow, remove canonical face coverage from that exterior output.
4. Paint Center and Outside Stroke entries.
5. Paint canonical rich foreground or its Gradient unless Hollow is
   active.
6. Paint Inside Stroke, then interior Shadow and Glow over the foreground.
7. Multiply the completed Normal composite by the block-owned alpha mask.
8. Apply group Overall Opacity and hand the completed source to the global
   transform path described in [Composable text transforms](text_transforms.md).

On coverage-producing layers, effect opacity and gradient-stop opacity each
multiply coverage exactly once. Gradient overwrites foreground RGB and
multiplies canonical foreground alpha by stop opacity, so a transparent stop
does not reveal the original solid foreground. Preserve straight-RGBA rounding
at the paint boundary; changing the order can square coverage or make clipped
Inside/Outside pixels visible.

| Effective state | Render path |
| --- | --- |
| No active pixel effects and no active mask | Native Qt foreground and group opacity; no effect raster state |
| Solid Center Stroke without a completed-surface consumer | Native/direct Stroke and foreground fast path |
| Positioned or gradient Stroke, Shadow/Glow, Gradient, Hollow, or active mask | One bounded completed effect surface, full or tiled |
| Effects or mask plus an active nonlinear transform | Their completed source surface is warped once by the geometry owner |
| Strict export | Independent exact-quality namespace; incomplete output is reported rather than silently omitted |

Interactive allocation failure may fall back to a bounded tile or compatible
direct-Stroke path for the frame. Export must not turn that degradation into a
successful but incomplete image.

## Coordinates, padding, and interaction

Effects operate between the item-local logical and item-local source spaces
defined in [Text engine](text_engine.md). The persistent text rectangle never
includes effect padding or transformed overflow.

- Stroke, Shadow, and Glow geometry is relative to the maximum font size.
- Inside Stroke and interior effects do not expand source bounds. Center Stroke
  contributes its outside half; Outside Stroke contributes its full reach.
- Exterior effects expand source padding from the visible source silhouette.
  Glyph-distorting paths use layout-owned ink bounds; ordinary paths use the
  conservative symmetric padding calculation.
- Every linear gradient remains anchored to the unpadded logical rectangle,
  even when a tile renders only part of the effect surface.
- Alpha-mask points are relative to the unpadded logical origin. They may erase
  effect overflow, but the mask itself never expands bounds.

Effect padding is paint geometry, not interaction geometry. Ordinary shape,
click, double-click, selection, and resize behavior stays on the logical box
plus layout-owned ink overhang. Qt remains authoritative for shaping, cursor,
selection, and IME. Selection is composited after the completed effect surface,
and the caret is deferred and painted last, so neither is erased by Hollow,
masked, shadowed, or included in cached effect pixels.

## State, persistence, and migration

| State | Persistent owner | Transfer behavior |
| --- | --- | --- |
| `TextEffectStack` | `FontFormat` | Reusable typography style; presets and multi-selection formatting may copy it |
| `TextAlphaMask` history | `TextBlock` | Item-specific structural alpha; never copy it through a style or preset |
| Preview stacks, preview masks, pixmaps, alpha planes, padding, and cache keys | Runtime owners only | Never serialize |

Live effect values are strict and typed. Passive project/config loading is
permissive: malformed top-level fields fall back independently, and an invalid
effect, mask stroke, or mask point is warned about and discarded without
losing valid siblings or replacing the surrounding project.

Compatibility rules are intentional:

- Presence of `text_effects` is authoritative, including an explicitly empty
  or malformed typed payload; do not revive older flat settings behind it.
- Legacy flat `opacity`, `stroke_width`, and `srgb` migrate to Overall Opacity and
  a solid Outside primary Stroke. Legacy Shadow and Gradient fields are ignored
  with one warning per load owner.
- Typed Stroke payloads written before `position` existed load as Center,
  preserving their old appearance. New Stroke values default to Outside.
- Typed Shadow payloads written before shared paints existed migrate their bare
  RGB `color` to `SolidPaint`.
- Typed Gradient payloads ignore the removed effect-level `opacity`; stop
  opacity remains authoritative and new serialization omits the old field.
- Serialization keeps the typed stack authoritative and dual-writes neutralized
  legacy fields plus compatible opacity/primary-Stroke views for older readers.

Pipeline formatting also respects this split. A Run width or Stroke-color
override inserts or updates the primary Stroke without replacing its other
parameters or sibling effects. The general effect override copies non-Stroke
style state from the global format while retaining target-owned Stroke entries.

## Preview, commit, undo, and mask editing

```text
panel control
  -> TextEffectEditSession complete-stack preview
     -> cancel: restore committed state, no command
     -> commit: one SetTextEffectStackCommand for all targets

canvas mask brush
  -> TextAlphaMaskEditSession complete-mask preview
     -> cancel incomplete stroke: restore committed mask, no command
     -> release: one SetTextAlphaMaskCommand for the target
```

An effect preview replaces the complete effective stack at the item boundary;
it never mutates committed `FontFormat`, project JSON, `QTextDocument` history,
or the paired editor. A parameter switch first cancels an incompatible active
preview. Structural add, remove, move, or Hollow changes settle pending inputs
and cancel transient previews before changing indices. Reordering is allowed
only within the same compiler phase.

Multiple selected items may edit matching indices only when their effect-type
sequences agree. Values at those indices may still be mixed. One committed
action snapshots the complete before/after stack for every target, so undo does
not reconstruct state from individual controls. With no item targets, the same
session updates the global format directly because there is no canvas state to
put on the item undo stack.

Pixel-changing effect previews use a non-promotable 0.5x physical scratch
surface. Commit and export rerender at the requested quality. Overall Opacity
is native group state and does not require rebuilding effect pixels. Reshape
temporarily omits effects, invalidates geometry-sensitive caches, and rebuilds
the effective namespace once geometry settles.

The Canvas owns the one global alpha-mask edit session. It activates only for
one eligible selected text item, freezes the scene-to-source mapper and logical
origin for the duration of a brush stroke, and keeps raw samples session-owned.
First activation inserts an empty enabled mask as its own undo step. Each
completed Erase or Restore stroke adds one immutable history entry and one undo
command. Escape, selection/page/scene teardown, or starting another effect edit
discards an incomplete stroke.

Before save, undo/redo, page replacement, or scene teardown, resolve pending
numeric/gradient edits and previews at their owning session. Do not let a
preview survive while stack indices or target items change underneath it.

## Cache boundaries and performance

Effect caches are item-owned, bounded, and split by what changes:

| Cached work | Reuse and invalidation boundary |
| --- | --- |
| Final full surface or visible tiles | Separate committed, preview, and export namespaces; key by effective effects, mask generation, document/layout/render state, transform, writing mode, bounds, and quality tier |
| Complete pre-mask surface | At most two entries per namespace; reuse across mask-only previews while upstream effects and geometry match |
| Canonical glyph pixmap and lazy alpha | At most two entries; reuse across paint-only edits while document, layout, source geometry, transform state, and render scale match |
| Positioned Stroke coverage | At most two read-only alpha planes; key by canonical-source inputs plus Stroke width and position, excluding paint and opacity |
| Gradient compiled kernel | Runtime acceleration only; the byte-identical NumPy path remains the pre-warm, unavailable, and quality-oracle fallback |

Within one composite, reuse the same colored positioned Stroke band for its
visible pass and exterior silhouette. Paint-only edits must not rerasterize the
native glyph outline. Shadow and Glow alpha are intentionally generated on
demand; their measured cost does not justify another invalidation surface.

Keep antialiasing enabled. The live 0.5x tier provides the useful preview
tradeoff; disabling painter antialiasing did not materially reduce native
outline cost. Large surfaces use the shared bounded full/tile policy instead of
unbounded allocation. Every namespace must release pixmaps and arrays on
reshape invalidation, item/page removal, and return to an inactive path.

Numba gradient warmup is queued after the main window is constructed and runs
outside the Qt thread. Painting before it is ready stays on NumPy and must not
import or compile Numba synchronously on the UI thread.

## Adding or changing an effect

1. Add or extend an immutable typed value with a stable `effect_type`, strict
   live validation, serialization, exact neutral state, and tolerant passive
   loading.
2. Choose its existing fixed phase and source alpha. If it does not fit one,
   revisit the composition contract before adding renderer branches.
3. Define its external reach and whether Hollow, Stroke-inclusive exterior
   alpha, canonical alpha, Gradient, and the block mask affect it.
4. Extend the panel and `TextEffectEditSession` using the same typed fields and
   complete-stack preview/commit boundary. Keep new visible strings translated.
5. Audit full and tiled output, source coordinates, cache keys, reshape,
   neutral activation/deactivation, strict export, and nonlinear transforms.
6. Cover migration and malformed optional data without weakening strict live
   values. Preserve one-command undo and paired-editor isolation.

Do not add a generic effect graph, another raster owner, or a generated-layer
cache without a measured problem. If a new blend mode needs the page backdrop,
design that cross-layer/export boundary explicitly before adding its UI.

## Focused verification

Start with the suites matching the ownership changed:

- `tests/test_text_effect_domain.py`
- `tests/test_text_effect_persistence.py`
- `tests/test_text_effect_panel.py`
- `tests/test_text_effect_preview.py`
- `tests/test_typed_text_effect_renderer.py`
- `tests/test_text_alpha_mask.py`
- `tests/test_text_alpha_mask_renderer.py`
- `tests/test_text_alpha_mask_edit_session.py`
- `tests/test_effect_paint_numba.py`

Broaden to `tests/test_text_transform_undo.py`,
`tests/test_rich_text_annotations.py`, and the relevant Run-pipeline tests when
the change crosses those boundaries.

```bash
QT_API=pyqt6 QT_QPA_PLATFORM=offscreen \
  /opt/miniconda3/envs/common/bin/python -m pytest -q \
  tests/test_text_effect_domain.py tests/test_text_effect_persistence.py \
  tests/test_text_effect_preview.py tests/test_typed_text_effect_renderer.py

git diff --check
```

Rendering, interaction, or panel changes still need a themed-app pass covering
horizontal and vertical text, Hollow with selection/caret, mask editing,
neutral and nonlinear transforms, preview/cancel/commit, undo/redo, zoom, and
strict export as applicable.
