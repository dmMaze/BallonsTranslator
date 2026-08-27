# Text effects

Read [Text engine](text_engine.md) first. This guide is an orientation to the
text-effect subsystem: its ordered composition model, ownership boundaries, and
the lifecycle and cache contracts that are easy to break. The code and focused
tests are authoritative for individual controls and raster algorithms.

## Mental model

```text
QTextDocument + settled SceneTextLayout
  -> canonical glyph source, including Glyph Slant when active
  -> canonical rich foreground or repeatable Text Fill base group, unless Hollow
  -> TextBlock-owned Rendered Image base (Replace or Overlay)
  -> ordered Stroke / Shadow / Glow / Filter cards, panel top-to-bottom
  -> TextBlock-owned alpha mask
  -> Overall Opacity
  -> ordered global Text Transform
  -> item position and rotation
```

`TextEffectStack` is an ordered value, but it is not an arbitrary layer graph.
For persistence compatibility its tuple remains topmost-first; the panel
projects movable cards in application order, so the first visible card runs
first. A Filter transforms the base and generated layers accumulated by cards
above it, while cards below it run afterward. Generated effects still derive
high-quality geometry from canonical glyph/Stroke alpha; order does not turn a
later Stroke into a raster outline of filtered pixels. Text Fill and Hollow
remain structural base state, independent of their serialized position. Text
Fills can be reordered only with other Text Fills; Hollow is not reordered.

New movable effects and Text Fills are appended to their visible panel areas
and therefore execute last within their respective stacks. This is stored at
tuple index zero; existing persisted tuples are neither rewritten nor migrated
for the presentation remapping.

## Owners

| Concern | Owner |
| --- | --- |
| Immutable effects, paints, stack helpers, and tolerant typed loading | [`utils/text_effects.py`](../../ballontranslator/utils/text_effects.py) |
| Generic immutable raster references and their serialization validation | [`utils/raster_assets.py`](../../ballontranslator/utils/raster_assets.py) |
| Content-addressed raster import and safe project-relative resolution | [`utils/proj_imgtrans.py`](../../ballontranslator/utils/proj_imgtrans.py) |
| Style persistence, legacy migration, and compatibility views | [`utils/fontformat.py`](../../ballontranslator/utils/fontformat.py) |
| Block-owned alpha-mask persistence | [`utils/text_alpha_mask.py`](../../ballontranslator/utils/text_alpha_mask.py), [`utils/textblock.py`](../../ballontranslator/utils/textblock.py) |
| Block-owned Rendered Image persistence | [`utils/rendered_image.py`](../../ballontranslator/utils/rendered_image.py), [`utils/textblock.py`](../../ballontranslator/utils/textblock.py) |
| Lazy filter metadata, active runtime resolution, and built-ins | [`ui/text_engine/effects/filters/`](../../ballontranslator/ui/text_engine/effects/filters/), [Text filters](text_filters.md) |
| Composition, padding, raster policy, preview namespaces, and caches | [`ui/text_engine/effects/renderer.py`](../../ballontranslator/ui/text_engine/effects/renderer.py), [`ui/text_engine/effects/`](../../ballontranslator/ui/text_engine/effects/) |
| Selection-scoped effect preview and commit | [`ui/text_engine/effects/edit_session.py`](../../ballontranslator/ui/text_engine/effects/edit_session.py) |
| Panel projection and card controls | [`ui/text_engine/effects/panel.py`](../../ballontranslator/ui/text_engine/effects/panel.py), [`ui/text_engine/effects/gradient_editor.py`](../../ballontranslator/ui/text_engine/effects/gradient_editor.py) |
| Canvas brush input and mask undo | [`ui/text_engine/effects/alpha_mask_edit_session.py`](../../ballontranslator/ui/text_engine/effects/alpha_mask_edit_session.py) |
| Effect and mask undo commands | [`ui/text_engine/editing/commands.py`](../../ballontranslator/ui/text_engine/editing/commands.py) |
| Source/visual bounds and the final global mapping | [`ui/text_engine/geometry.py`](../../ballontranslator/ui/text_engine/geometry.py), [`ui/text_engine/transforms/`](../../ballontranslator/ui/text_engine/transforms/) |

The panel projects model values and emits edit requests; it does not own effect
state. `TextBlkItem` is the Qt boundary, while `TextEffectRenderer` owns the
completed effect surface. Add behavior at those existing owners instead of
introducing a second effect model, renderer, or edit session.

Effect and Transform cards share `BottomBorderComboBox`, the compact selector
used by the Run dialog. Stroke Position and Shadow/Glow Type follow the card
title. Fill stays in the left control column; Solid shows its swatch opposite,
Gradient expands the shared stop editor below it, and Text Fill additionally
offers Texture with an image chooser plus Fill/Fit/Crop/Tile mapping. The
Stroke, Shadow, Glow, and Text Fill cards expose only Normal, Darken, and
Lighten in their Blend selector; mixed selections show Mixed. Text Fill puts
Opacity and Blend in the compact row below its paint row. The
Texture choice exists only for concrete project-item selections; global and
itemless formatting never offers it. For a mixed Texture selection, asset,
mapping, and scale compare independently. The chooser remains enabled when
assets differ and selecting a file changes only the asset, retaining each
item's mapping and scale. The file dialog, synchronous import, and any error
message keep the formatting panel pinned for their complete lifetime. The
editor orders its
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

Rendered Image is a fixed item-specific base card after globally movable cards,
above the fixed Text Fill area and before Eraser. It exists only for one
concrete project TextBlock and therefore never appears in global formatting,
presets, or multi-selection. Its Image and
Replace/Overlay controls use the same two equal-column, natural-label-spacing
card rows and shared pinned project-image chooser. The card tooltip explains
that the layer is intentionally hidden during native text editing.

## Model, sources, and ordered composition

`FontFormat.text_effects` contains one immutable `TextEffectStack`. Its
`overall_opacity` applies to the completed item, while `effects` preserves the
semantic order of typed immutable values. Stroke, Shadow, Glow, Text Fill, and
Filter are repeatable. Hollow is unique.

| Effect | Phase and source | Important semantics |
| --- | --- | --- |
| Stroke | Ordered generated layer | Width is relative to font size. Center splits the full band across the glyph edge; Outside and Inside clip a full-width outline to the corresponding side. A completed surface caches raw outline coverage for every position, then clips Center paint outside canonical face alpha unless Hollow needs the full band. This matches the direct path's later foreground repaint without putting Hollow in the geometry-cache key. New and migrated legacy strokes default to Outside. |
| Shadow | Ordered generated layer; exterior source for Drop/Long, interior source for Inner | Exterior Shadow uses the canonical Stroke-inclusive silhouette but clips output only outside the canonical face. It therefore cannot tint foreground, while global order still decides whether a higher Shadow covers a lower Stroke. Inner Shadow uses canonical glyph alpha and is suppressed by Hollow. |
| Glow | Ordered generated layer; exterior source for Outer, interior source for Inner | Outer Glow uses the canonical Stroke-inclusive silhouette. Inner Glow uses canonical glyph alpha and is suppressed by Hollow. |
| Text Fill | Structural base-fill sub-stack, repeatable | Enabled renderable fills compose in visible order on one transparent face group. Solid or Gradient paints the logical rectangle; Texture maps one managed raster over it. The completed group is clipped once by canonical glyph coverage and replaces the rich foreground as a group. Paint alpha and effect Opacity each multiply alpha once. Stroke and generated effects continue using their earlier canonical/source alpha. |
| Hollow | Foreground modifier, unique | Removes the canonical face, Text Fill, and interior phase while retaining Stroke and exterior output. It is a toggle, not an independent painted layer. |
| Filter | Ordered pixel transform, repeatable | Transforms the base and generated layers accumulated above its visible card. Consecutive Filters execute panel top-to-bottom through one RGBA bridge. Alpha is non-expanding by default; an explicitly declared expander is halo-bounded and adds matching effect padding. |

`RenderedImageLayer` is not a `TextEffectStack` node or paint. It is one
immutable versioned `TextBlock` value containing `enabled`, a generic
`RasterAssetRef`, and `mode=replace|overlay`. Replace clears the entire isolated
text/effect surface, including padded Stroke/Shadow overflow, then maps the
image exactly into the unpadded logical text rectangle. Transparent source
pixels remain transparent. Overlay source-over composites above Text Fill and
below the movable stack. Neither mode expands padding.

Stroke, Shadow, and Glow use either `SolidPaint` or `LinearGradientPaint`.
Text Fill uses `SolidPaint`, `LinearGradientPaint`, or `TexturePaint`. A
gradient contains two to 32
ordered stops; RGB and stop opacity interpolate independently. Its angle and
scale are defined against the complete unpadded logical text rectangle, not
each glyph, effect layer, or tile, so writing modes and render paths agree.
If no enabled Text Fill can render, the canonical rich foreground remains. If
at least one can render, the transparent fill group replaces that foreground.
An enabled transparent or zero-Opacity Text Fill is renderable and therefore
can intentionally erase the face. A missing or invalid Texture is bypassed
interactively; it does not cause replacement by itself, but valid sibling
fills still do. Strict export reports the missing asset instead.

`TexturePaint` stores one generic immutable `RasterAssetRef` from
`utils/raster_assets.py`. Project import snapshots the selected source once in
the validated assets directory, hashes and fully decodes that same snapshot,
then atomically installs it as `assets/<sha256>.<actual-format>`. An existing
destination must match its content digest. Persistence stores only a relative
reference plus its display name. Fill stretches to the logical rectangle, Fit
contains and centers the whole image, Crop covers and center-crops it, and Tile
repeats at the selected scale from the unpadded logical top-left. Full surfaces,
visible tiles, both writing modes, and downstream text transforms therefore
sample the same logical point from the same texture point.

Stroke, Shadow, Glow, and Text Fill accept exactly `normal`, `darken`, and
`lighten`. QPainter applies these native modes while composing generated
layers into the isolated text surface and fills into their transparent group.
The destination is always earlier output in that local stack, never the page
backdrop. This same destination and order apply to full, tiled, preview, and
export rendering; no extra draw or RGBA blend bridge is used for these modes.

Neutral effects stay in model and panel state but are skipped by rendering.
Keep that neutral test explicit when adding a type; disabled, zero-opacity, or
zero-extent values must not allocate effect surfaces when they leave output
unchanged. An enabled transparent Gradient is the intentional exception above.

## Composition and render paths

The renderer captures canonical glyph pixels once, including the current
layout-owned placement and any Glyph Slant renderer. It then compiles the
effective stack as follows:

1. Unless Hollow is active, paint the canonical rich foreground when no
   enabled Text Fill can render. Otherwise compose enabled renderable Text
   Fills in visible order on a transparent surface, apply their paint/effect
   alpha, clip the group once with cached canonical glyph coverage, and use the
   group in place of canonical foreground.
2. Apply the optional block-owned Rendered Image base. Overlay sits above Text
   Fill; valid Replace skips canonical and generated-layer raster work.
3. Walk movable cards top-to-bottom as displayed by the panel (the reverse of
   their compatibility-preserved tuple order). Batch adjacent Stroke/Shadow/Glow
   cards in one painter segment and adjacent Filters in one straight-RGBA8
   bridge.
4. Generate every typed layer from its canonical source: exterior Shadow/Glow
   use the complete canonical Stroke-inclusive silhouette, while interior
   effects use canonical glyph alpha. Drop/Long Shadow clip outside the
   canonical face, while Outer Glow clips outside its full source silhouette,
   at generation time. Hollow therefore only suppresses the base and interior
   layers; it needs no extra clipping pass.
5. Multiply the completed composite by the block-owned alpha mask.
6. Apply group Overall Opacity and hand the completed source to the global
   transform path described in [Composable text transforms](text_transforms.md).

On coverage-producing layers, effect opacity and gradient-stop opacity each
multiply coverage exactly once. Text Fill paint alpha and effect Opacity are
composed before the shared canonical mask, so transparent fill output does not
reveal the original rich foreground. Preserve straight-RGBA rounding at the
paint boundary; changing the order can square coverage or make clipped
Inside/Outside pixels visible.

| Effective state | Render path |
| --- | --- |
| No active pixel effects and no active mask | Native Qt foreground and group opacity; no effect raster state |
| Solid Center Stroke without a completed-surface consumer | Native/direct Stroke and foreground fast path |
| Positioned or gradient Stroke, Shadow/Glow, Text Fill, Hollow, Rendered Image, Filter, or active mask | One bounded completed effect surface, full or tiled |
| Effects or mask plus an active nonlinear transform | Their completed source surface is warped once by the geometry owner |
| Strict export | Independent exact-quality namespace; incomplete output is reported rather than silently omitted |

Interactive allocation failure may fall back to a bounded tile or compatible
direct-Stroke path for the frame. Export must not turn that degradation into a
successful but incomplete image.

A missing optional Texture or Rendered Image asset is warned about and visibly
bypassed during interactive rendering. A missing Texture leaves canonical
foreground when no sibling fill renders; a missing Rendered Image leaves its
upstream surface in place. Its card shows the missing filename. Strict export
fails the render transaction instead of silently exporting the bypass. Strict export
SHA-256-verifies the file even when interactive rendering already populated the
decoded cache. The strict hash and matching cache reuse or decode share one
before/after file-signature bracket, so replacement during verification fails
the render transaction. Every interactive cache reuse still checks file
existence and containment. Unchanged warm entries avoid digest hashing, while
cold or stat-changed files are SHA-256-verified before decode; corrupt bytes and
a different valid image at the content-addressed path therefore bypass visibly.
Missing and failed interactive reads are not cached; after deletion plus
invalidation the rich foreground is visible, and after restore plus invalidation
the texture returns without a page reload.

Raster imports and first loads are bounded to 32 MiB of source bytes, 64 Mpx,
and 256 MiB of decoded RGBA8 storage. Supported 8-bit Pillow modes decode through
an owned immutable RGBA8 copy; wider integer and floating-point modes are
rejected instead of truncated. A successful user import also warms the shared
project cache, keeping decode out of subsequent paint. Opening an older project
may perform one bounded synchronous decode on the first uncached use; v1 does
not add a worker or asset registry for that narrow case.

Texture and Rendered Image scaling interpolate premultiplied RGBA in absolute
logical coordinates, then return to straight RGBA for composition. Rendered
Image samples only the logical intersection of each bounded full/tile surface
and draws the synchronous `QImage` directly, avoiding transparent-edge color
fringes and a redundant padded pixmap. The project decode cache lazily retains
an immutable premultiplied companion only for transparent assets, so full and
tiled sampling do not copy and premultiply the complete source on each tile. It
retains at most two entries and 512 MiB of unique array storage; opaque assets
share their straight representation.

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
- Every linear gradient and texture remains anchored to the unpadded logical
  rectangle, even when a tile renders only part of the effect surface.
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
| `TextEffectStack` | `FontFormat` | Reusable typography style; presets and multi-selection formatting may copy portable values, while project Texture fills remain item/project-only |
| `RenderedImageLayer` | `TextBlock` | Item-specific full-RGBA replacement/overlay; copied only with the TextBlock, never through formatting or presets |
| `TextAlphaMask` history | `TextBlock` | Item-specific structural alpha; never copy it through a style or preset |
| Preview stacks, preview masks, pixmaps, alpha planes, padding, and cache keys | Runtime owners only | Never serialize |

Live effect values are strict and typed. Passive project/config loading is
permissive: malformed top-level fields fall back independently, and an invalid
effect, mask stroke, or mask point is warned about and discarded without
losing valid siblings or replacing the surrounding project.
An unknown or malformed `blend_mode` on a supported passive effect is warned
about and replaced with Normal without discarding the effect; live constructors
and explicit writes remain strict.

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
- Typed `gradient` and `gradient_overlay` payloads migrate losslessly to
  `TextFillEffect` with `LinearGradientPaint`; their removed effect-level
  `opacity` is deliberately ignored. Older `text_fill` payloads default their
  newly supported effect Opacity to `1.0`. New serialization writes repeatable
  `text_fill` entries with Opacity and Blend.
- Malformed Raster asset data discards only the optional Text Fill or Rendered
  Image value that owns it on passive load. Valid-but-missing files remain
  referenced so the project can recover when its `assets/` contents are
  restored.
- Application-global formatting and reusable presets have no project asset
  registry in v1. They strip only `TextFillEffect(TexturePaint)` on passive
  load, edit, update, and save boundaries while preserving every other effect.
  Project TextBlocks retain valid Texture refs, and the absence of Text Fill
  keeps the original rich foreground.
- Unknown filter IDs, schemas, and opaque scalar params remain portable through
  passive project/config/preset loading without importing plug-in code. Active
  resolution passes only declared params; invalid known values fall back to
  metadata defaults without mutating the preserved payload.
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

Rendered Image card
  -> chooser / mode / eye / delete
     -> one SetRenderedImageLayerCommand for the target; no preview session
```

An effect preview replaces the complete effective stack at the item boundary;
it never mutates committed `FontFormat`, project JSON, `QTextDocument` history,
or the paired editor. A parameter switch first cancels an incompatible active
preview. Structural add, remove, move, or Hollow changes settle pending inputs
and cancel transient previews before changing indices. Reordering swaps
adjacent movable Stroke/Shadow/Glow/Filter cards while skipping structural Text
Fill and Hollow values. A Text Fill reorder separately swaps adjacent fills,
even when generated entries occupy raw tuple positions between them; it never
changes generated-layer order.

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

While native text editing is active, Rendered Image is intentionally omitted
for both writing modes so the editable source, selection, caret, IME, and
annotations remain coherent. Ending the edit restores the exact settled layer;
strict export still includes it. Selection and the deferred caret remain after
all completed pixel phases. Starting a Rendered Image chooser or discrete card
edit also deactivates the Eraser brush session.
Ordinary Filters remain active during horizontal and vertical native editing;
selection, caret, and IME feedback are painted afterward and never enter their
input pixels.

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

Effect output caches are item-owned and bounded. Decoded immutable raster data
is instead shared at the project asset boundary:

| Cached work | Reuse and invalidation boundary |
| --- | --- |
| Final full surface or visible tiles | Separate committed, preview, and export namespaces; key by effective effects, active Rendered Image, mask generation, document/layout/render state, transform, writing mode, bounds, and quality tier |
| Complete pre-mask surface | At most two entries per namespace; reuse across mask-only previews while upstream effects and geometry match |
| Complete below-filter prefix | At most two entries per namespace; filter-only previews reuse the fixed base and generated nodes below the bottom Filter. The key includes that boundary and canonical Stroke dependencies of cached exterior layers. Upper generated layers are cheaply recomposited from retained canonical/coverage caches. |
| Canonical glyph pixmap and lazy alpha | At most two entries; reuse across paint-only, Fill Opacity, and Blend previews while document, layout, source geometry, transform state, and render scale match. Repeated fills share this source and its mask rather than rerasterizing glyphs. |
| Positioned Stroke coverage | At most two read-only alpha planes; key by canonical-source inputs plus Stroke width and position, excluding paint and opacity |
| Gradient compiled kernel | Runtime acceleration only; the byte-identical NumPy path remains the pre-warm, unavailable, and quality-oracle fallback |
| Decoded project raster | At most two positive entries per project, shared by Texture and Rendered Image and keyed by immutable relative ref; successful import prewarms it, project reload clears it, and failures are never cached |

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
2. Choose its canonical source alpha and whether it is a movable generated
   layer, pixel Filter, or fixed structural value.
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
