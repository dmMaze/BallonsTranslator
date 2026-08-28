# Text effects

Read [Text engine](text_engine.md) first. This guide is an orientation to the
text-effect subsystem: its ordered composition model, ownership boundaries, and
the lifecycle and cache contracts that are easy to break. The code and focused
tests are authoritative for individual controls and raster algorithms.

## Mental model

```text
QTextDocument + settled SceneTextLayout
  -> canonical glyph source, including Glyph Slant when active
  -> canonical rich foreground or repeatable Gradient/Texture group,
     unless Hollow
  -> ordered Image / Stroke / Shadow / Glow / Filter cards, panel top-to-bottom
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
later Stroke into a raster outline of filtered pixels. The UI exposes fixed
**Gradient** and **Texture** cards; both reuse the internal `TextFillEffect`
value and renderer. Inline font color remains the sole solid foreground source.
Foreground paints and Hollow remain structural base state, independent of their
serialized position. Foreground paints can be reordered only with each other;
Hollow is not reordered.

New movable effects and foreground paints normally append to their visible
panel areas and therefore execute last within their respective stacks. This is
stored at tuple index zero; existing persisted tuples are neither rewritten nor
migrated for the presentation remapping. During multi-selection, a new effect
is instead inserted after the occurrences of the same structural identity that
are common to every target. That makes the newly added occurrence matched while
leaving unmatched surplus occurrences in place.

## Owners

| Concern | Owner |
| --- | --- |
| Immutable effects, paints, stack helpers, and tolerant typed loading | [`utils/text_effects.py`](../../ballontranslator/utils/text_effects.py) |
| Generic immutable raster references and their serialization validation | [`utils/raster_assets.py`](../../ballontranslator/utils/raster_assets.py) |
| Content-addressed raster import and safe project-relative resolution | [`utils/proj_imgtrans.py`](../../ballontranslator/utils/proj_imgtrans.py) |
| Style persistence, legacy migration, and compatibility views | [`utils/fontformat.py`](../../ballontranslator/utils/fontformat.py) |
| Block-owned alpha-mask persistence | [`utils/text_alpha_mask.py`](../../ballontranslator/utils/text_alpha_mask.py), [`utils/textblock.py`](../../ballontranslator/utils/textblock.py) |
| Lazy filter metadata, active runtime resolution, and built-ins | [`ui/text_engine/effects/filters/`](../../ballontranslator/ui/text_engine/effects/filters/), [Text filters](text_filters.md) |
| Composition, padding, raster policy, preview namespaces, and caches | [`ui/text_engine/effects/renderer.py`](../../ballontranslator/ui/text_engine/effects/renderer.py), [`ui/text_engine/effects/`](../../ballontranslator/ui/text_engine/effects/) |
| Selection-scoped effect preview and commit | [`ui/text_engine/effects/edit_session.py`](../../ballontranslator/ui/text_engine/effects/edit_session.py) |
| Image-generation request, logical crop, and background job | [`ui/text_engine/effects/image_generation.py`](../../ballontranslator/ui/text_engine/effects/image_generation.py), [`modules/llm_image.py`](../../ballontranslator/modules/llm_image.py) |
| Panel projection and card controls | [`ui/text_engine/effects/panel.py`](../../ballontranslator/ui/text_engine/effects/panel.py), [`ui/text_engine/effects/gradient_editor.py`](../../ballontranslator/ui/text_engine/effects/gradient_editor.py) |
| Canvas brush input and mask undo | [`ui/text_engine/effects/alpha_mask_edit_session.py`](../../ballontranslator/ui/text_engine/effects/alpha_mask_edit_session.py) |
| Effect and mask undo commands | [`ui/text_engine/editing/commands.py`](../../ballontranslator/ui/text_engine/editing/commands.py) |
| Source/visual bounds and the final global mapping | [`ui/text_engine/geometry.py`](../../ballontranslator/ui/text_engine/geometry.py), [`ui/text_engine/transforms/`](../../ballontranslator/ui/text_engine/transforms/) |

The panel projects model values and emits edit requests; it does not own effect
state. `TextBlkItem` is the Qt boundary, while `TextEffectRenderer` owns the
completed effect surface. Add behavior at those existing owners instead of
introducing a second effect model, renderer, or edit session.

The fixed top row is Add, Eraser, Hollow, a spacer, then Opacity. The opt-in
Faster Preview checkbox lives in the Text Effect expander header. It is a
runtime panel preference, not a project/style value or an undoable effect edit.

Effect and Transform cards share `BottomBorderComboBox`, the compact selector
used by the Run dialog. The Blend control uses the same transparent
bottom-border treatment on a native menu button so its families remain
keyboard-accessible: Normal is direct, Darken contains Darken, Multiply, Color
Burn, Linear Burn, and Darker Color, and Lighten contains Lighten, Screen,
Color Dodge, Linear Dodge (Add), and Lighter Color. The primary item's Blend
leaf stays checked during multi-selection. These modes add no mode-specific
parameter row; each effect's existing Opacity remains independent. Stroke
Position and Shadow/Glow Type follow the card title. Gradient expands the
shared stop editor and puts Opacity and Blend in a compact two-column row below
it. Texture has Image and Opacity on row one,
Mapping and Blend on row two, and exposes Scale on a third row only for Tile
mapping. Texture uses the same blank Glossary-style raster field and embedded
file picker as Image. An Empty Texture remains blank and keeps its
Fill/Fit/Crop/Tile mapping; adding Texture does not open the chooser. Texture exists only for
concrete project-item selections;
global and itemless formatting never offers it. For a mixed Texture selection,
asset, mapping, and scale compare independently. The chooser remains enabled
when assets differ and selecting a file changes only the asset, retaining each
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
column width. Editor and current-selector values are centered. Two-column rows
use equal columns and an 8 px inter-column gap. There is no redundant Flip
action.

The panel projects Gradient/Texture structural cards first, followed by the complete
movable Image/Stroke/Shadow/Glow/Filter execution order and finally Eraser.
Image is project-only, repeatable, reorderable, and available for single-item
project selections; global formatting and presets strip it. During
multi-selection, existing Image cards remain primary-item reference cards and
Add Image is disabled because Image is excluded from occurrence matching. Its
Image and In Front/Behind controls use the same compact card rows and pinned
project-image chooser. Adding a card leaves its asset unset and its
Glossary-style image field blank without opening that chooser, and defaults its
eventual placement to In Front.
The card tooltip explains that Image is intentionally hidden during native
text editing. Each Image card also has a draft-only Generate section. Its
Model, Context, and Prompt controls update only the live card's pending recipe;
they do not change the effect stack, create undo history, or invalidate/render
the text item. The hierarchical Model menu runs as one synchronous popup
transaction whose formatting-owner pin covers the complete close notification;
it has no timer or focus-restoration dependency. Other child controls and their
top-level popup parent chains share the same owner. A real target change
projects the new item's persisted recipe. Generate reads the pending fields,
and only a successful request commits the generated asset and saved recipe
together.

## Model, sources, and ordered composition

`FontFormat.text_effects` contains one immutable `TextEffectStack`. Its
`overall_opacity` applies to the completed item, while `effects` preserves the
semantic order of typed immutable values. Stroke, Shadow, Glow, Gradient/Texture,
Image, and Filter are repeatable. Hollow is unique.

| Effect | Phase and source | Important semantics |
| --- | --- | --- |
| Stroke | Ordered generated layer | Width is relative to font size and retains the historical native-outline meaning for every position. Center splits that outline across the glyph edge; Outside and Inside clip the same outline to the corresponding side without doubling it. A completed surface caches raw outline coverage for every position, then clips Center paint outside canonical face alpha unless Hollow needs the full band. This matches the direct path's later foreground repaint without putting Hollow in the geometry-cache key. New and migrated legacy strokes default to Outside. |
| Shadow | Ordered generated layer; exterior source for Drop/Long, interior source for Inner | Angle uses the same screen-space convention as Gradient and Distance is relative to maximum font size. Exterior Shadow uses the canonical Stroke-inclusive silhouette but clips output only outside the canonical face. It therefore cannot tint foreground, while global order still decides whether a higher Shadow covers a lower Stroke. Inner Shadow uses canonical glyph alpha and is suppressed by Hollow. |
| Glow | Ordered generated layer; exterior source for Outer, interior source for Inner | Outer Glow uses the canonical Stroke-inclusive silhouette. Inner Glow uses canonical glyph alpha and is suppressed by Hollow. |
| Gradient/Texture | Structural foreground sub-stack, repeatable | Enabled renderable paints compose in visible order on one transparent face group. Gradient paints the logical rectangle; Texture maps one managed raster over it. The completed group is clipped once by canonical glyph coverage and replaces the rich foreground as a group. Paint alpha and effect Opacity each multiply alpha once. Stroke and generated effects continue using their earlier canonical/source alpha. Both cards persist through the internal `TextFillEffect` type. |
| Hollow | Foreground modifier, unique | Removes the canonical face, Gradient/Texture group, and interior phase while retaining Stroke and exterior output. It is a toggle, not an independent painted layer. |
| Image | Ordered project-raster layer, repeatable | Empty or disabled is neutral. In Front source-over composites above accumulated pixels, while Behind destination-over composites behind them so existing text remains visible. It does not change generated layers' canonical glyph sources. |
| Filter | Ordered pixel transform, repeatable | Transforms the base and generated layers accumulated above its visible card. Consecutive Filters execute panel top-to-bottom through one RGBA bridge. Alpha is non-expanding by default; an explicitly declared expander is halo-bounded and adds matching effect padding. |

`ImageEffect` is an immutable `TextEffectStack` node containing `enabled`, an
optional generic `RasterAssetRef`, `mode=foreground|background`, and an
optional regeneration recipe. The recipe stores a backend identity, profile,
model, context, and prompt; rendering depends only on the asset, so a removed
backend never hides an existing image. `mode=foreground|background` stores the
In Front or Behind placement. Both keep accumulated output and map the image
exactly into the unpadded logical text rectangle without expanding padding;
later cards, including Filters, still process the result.

Stroke, Shadow, and Glow use either `SolidPaint` or `LinearGradientPaint`.
`TextFillEffect` accepts only `LinearGradientPaint` or `TexturePaint`. A
gradient contains two to 32
ordered stops; RGB and stop opacity interpolate independently. Its angle and
scale are defined against the complete unpadded logical text rectangle, not
each glyph, effect layer, or tile, so writing modes and render paths agree.
If no enabled foreground paint can render, the canonical rich foreground
remains. If at least one can render, the transparent foreground group replaces
that foreground.
An enabled transparent or zero-Opacity Gradient is renderable and therefore
can intentionally erase the face. A Texture with no selected asset is neutral.
A missing or invalid Texture is bypassed
interactively; it does not cause replacement by itself, but valid sibling
paints still do. Strict export reports the missing asset instead.

`TexturePaint` stores an optional generic immutable `RasterAssetRef` from
`utils/raster_assets.py`. Project import snapshots the selected source once in
the validated assets directory, hashes and fully decodes that same snapshot,
then atomically installs it as `assets/<sha256>.<actual-format>`. An existing
destination must match its content digest. Persistence stores only a relative
reference plus its display name. Fill stretches to the logical rectangle, Fit
contains and centers the whole image, Crop covers and center-crops it, and Tile
repeats at the selected scale from the unpadded logical top-left. Full surfaces,
visible tiles, both writing modes, and downstream text transforms therefore
sample the same logical point from the same texture point.

Stroke, Shadow, Glow, Gradient, and Texture persist one flat leaf from the UI
families:
`normal`; `darken`, `multiply`, `color_burn`, `linear_burn`, `darker_color`;
or `lighten`, `screen`, `color_dodge`, `linear_dodge`, `lighter_color`. Family
names and submenus are presentation only. Normal, Darken, Multiply, Color Burn,
Lighten, Screen, and Color Dodge use native QPainter composition. Linear Burn,
Darker Color, Linear Dodge (Add), and Lighter Color use one exact
straight-RGBA8 source-over bridge per custom layer; row-chunked arithmetic
bounds working memory and preserves full/tiled rounding, but the pixmap readback
and replacement make these modes more expensive than native leaves. They do not
rerasterize glyphs or add work to native layers.

The first renderable foreground paint is source identity over its transparent
group, so it is source-copied without entering native or custom blend dispatch.
A single foreground paint therefore incurs no custom bridge regardless of its
saved leaf; a custom second paint incurs exactly one. The completed group is
still clipped once by canonical glyph coverage. Generated layers and later paints
blend with earlier output in their isolated local stack, never the page
backdrop. The same destination, order, and arithmetic apply to full, tiled,
preview, and export rendering.

Neutral effects stay in model and panel state but are skipped by rendering.
Keep that neutral test explicit when adding a type; disabled, zero-opacity, or
zero-extent values must not allocate effect surfaces when they leave output
unchanged. An enabled transparent Gradient is the intentional exception above.

## Composition and render paths

The renderer captures canonical glyph pixels once, including the current
layout-owned placement and any Glyph Slant renderer. It then compiles the
effective stack as follows:

1. Unless Hollow is active, paint the canonical rich foreground when no
   enabled foreground paint can render. Otherwise compose enabled renderable
   Gradient/Texture paints in visible order on a transparent surface, apply
   their paint/effect alpha, clip the group once with cached canonical glyph
   coverage, and use the group in place of canonical foreground.
2. Walk movable cards top-to-bottom as displayed by the panel (the reverse of
   their compatibility-preserved tuple order). Batch adjacent Stroke/Shadow/Glow
   or Image cards in one painter segment and adjacent Filters in one
   straight-RGBA8 bridge. In Front and Behind compose at their exact positions.
3. Generate every typed layer from its canonical source: exterior Shadow/Glow
   use the complete canonical Stroke-inclusive silhouette, while interior
   effects use canonical glyph alpha. Drop/Long Shadow clip outside the
   canonical face, while Outer Glow clips outside its full source silhouette,
   at generation time. Hollow therefore only suppresses the base and interior
   layers; it needs no extra clipping pass.
4. Multiply the completed composite by the block-owned alpha mask.
5. Apply group Overall Opacity and hand the completed source to the global
   transform path described in [Composable text transforms](text_transforms.md).

On coverage-producing layers, effect opacity and gradient-stop opacity each
multiply coverage exactly once. Foreground paint alpha and effect Opacity are
composed before the shared canonical mask, so transparent output does not
reveal the original rich foreground. Preserve straight-RGBA rounding at the
paint boundary; changing the order can square coverage or make clipped
Inside/Outside pixels visible.

| Effective state | Render path |
| --- | --- |
| No active pixel effects and no active mask | Native Qt foreground and group opacity; no effect raster state |
| Solid Center Stroke without a completed-surface consumer | Native/direct Stroke and foreground fast path |
| Positioned or gradient Stroke, Shadow/Glow, Gradient/Texture, Hollow, Image, Filter, or active mask | One bounded completed effect surface, full or tiled |
| Effects or mask plus an active nonlinear transform | Their completed source surface is warped once by the geometry owner |
| Strict export | Independent exact-quality namespace; incomplete output is reported rather than silently omitted |

Interactive allocation failure may fall back to a bounded tile or compatible
direct-Stroke path for the frame. Export must not turn that degradation into a
successful but incomplete image.

An Empty Texture or Image performs no lookup and remains valid during strict
export. A nonempty reference whose file is missing is warned about and visibly
bypassed during interactive rendering. A missing Texture leaves canonical
foreground when no sibling fill renders; a missing Image leaves its upstream
surface in place. Its card shows the missing filename. Strict export fails the
render transaction instead of silently exporting the bypass and SHA-256-verifies
the file even when interactive rendering already populated the decoded cache.
Raster decode, allocation, array-bridge, and OpenCV failures are contained
inside Qt paint/bounds callbacks. Strict export records the failure there and
raises it immediately after the Canvas returns to its Python boundary.
The strict hash and matching cache reuse or decode share one before/after
file-signature bracket, so replacement during verification fails the render
transaction. Every interactive cache reuse still checks file existence and
containment. Unchanged warm entries avoid digest hashing, while cold or
stat-changed files are SHA-256-verified before decode; corrupt bytes and a
different valid image at the content-addressed path therefore bypass visibly.
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

Texture and Image scaling interpolate premultiplied RGBA in absolute
logical coordinates, then return to straight RGBA for composition. Image
samples only the logical intersection of each bounded full/tile surface
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
- Inside Stroke and interior effects do not expand source bounds. Center and
  Outside Stroke contribute half of the same native outline width.
- Exterior effects expand source padding from the visible source silhouette.
  Glyph-distorting paths use layout-owned ink bounds; ordinary paths use the
  conservative symmetric padding calculation.
- Image contributes the unpadded logical rectangle to accumulated distorted
  bounds. A later alpha-expanding Filter grows that rectangle even for empty
  horizontal or vertical text; generated Shadow/Glow sources remain glyph-
  and Stroke-owned rather than sampling Image pixels.
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
| `TextEffectStack` | `FontFormat` | Reusable typography style plus project-only Image/Texture values on concrete items; global formatting and presets strip project raster entries |
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
  `text_fill` entries with Opacity and Blend. A solid `text_fill` is no longer a
  live value; passive loading warns and discards only that entry without
  migrating it, because inline font color owns solid foreground.
- Malformed Image fields recover independently on passive load: an invalid
  asset becomes Empty, invalid mode/enabled/recipe fields use defaults, and
  unknown backend identity remains saved without affecting a valid asset.
  Malformed Texture data still discards only that optional entry.
  Valid-but-missing files remain referenced so the project can recover when
  its `assets/` contents are restored.
- Application-global formatting and reusable presets have no project asset
  registry in v1. They strip `TextFillEffect(TexturePaint)` and `ImageEffect`
  on passive load, edit, update, and save boundaries while preserving every
  portable effect.
  Project TextBlocks retain valid Texture refs, and the absence of Gradient/Texture
  keeps the original rich foreground.
- The removed singleton `TextBlock.rendered_image` payload is ignored on
  passive load. It is not migrated or exposed through a compatibility model;
  `ImageEffect` is the only live Image owner.
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

Image card
  -> add Empty / chooser / mode / eye / reorder / delete
     -> same complete-stack session and one SetTextEffectStackCommand

  -> Generate draft (model / context / prompt)
     -> background request: committed ImageEffect stays unchanged
     -> success: import managed RGBA + recipe in one stack undo command
     -> failure / Stop / stale target: discard result, preserve draft and asset
```

An effect preview replaces the complete effective stack at the item boundary;
it never mutates committed `FontFormat`, project JSON, `QTextDocument` history,
or the paired editor. A parameter switch first cancels an incompatible active
preview. Structural add, remove, move, or Hollow changes settle pending inputs
and cancel transient previews before changing indices. Reordering swaps
adjacent movable Stroke/Shadow/Glow/Image/Filter cards while skipping structural
`TextFillEffect` and Hollow values. A foreground-paint reorder separately swaps
adjacent Gradient/Texture values, even when generated entries occupy raw tuple
positions between them; it never changes generated-layer order.

For multiple selected items, the panel always projects every card and exact
parameter value from the primary item. The latest direct canvas click or paired-
list selection anchor is primary; marquee and programmatic selections fall back
to the stable final selected item. Cards never synthesize Mixed parameter
values.

Batch editing uses a derived occurrence map rather than a merged or persistent
stack. Starting from the primary cards, it pairs each non-Image structural
identity in panel-visible occurrence order and intersects the available count
across every other item. Filter identity includes ID and schema, and Gradient
and Texture remain distinct. Relative order among unrelated effect types does
not affect a match. A card mapped across every item has a 2 px pink selection
border. Its absolute edits fan out the chosen value, while numeric label-drag
deltas remain relative to each target's own mapped value. An unmatched card,
and every Image card, edits only the primary item. Overall Opacity and Hollow
remain all-selected controls; Eraser remains single-item.

Adding Stroke, Shadow, Glow, Gradient, Texture, or Filter inserts a new value in
every selected stack. It lands after the occurrences of that structural
identity already common to every target, so the new primary card is matched
immediately even when one stack has unmatched surplus occurrences. Add Image is
disabled during multi-selection. Deleting a matched card removes every mapped
occurrence; deleting an unmatched or Image card changes only the primary stack.
Reordering a matched card fans out only when the relevant visible movable or
foreground-paint structural sequences align exactly; otherwise its controls
are disabled. An unmatched card, including Image, may still reorder within the
primary stack only. One
committed action snapshots the complete before/after stack for every target, so
undo does not reconstruct state from individual controls. With no item targets,
the same session updates the global format directly because there is no canvas
state to put on the item undo stack.

Pixel-changing effect previews use requested quality by default, allowing a
valid full-quality preview surface, including a higher device-scale tier, to
promote on commit. Requested-quality preview work is deferred to the next scene
paint so it renders once at the painter's actual scale. A checked 0.5x preview
also defers its exact committed rebuild to that paint instead of rebuilding at
an intermediate 1x tier. The runtime-only,
opt-in Faster Preview toggle selects a non-promotable 0.5x physical scratch
surface instead. The same choice controls the final nonlinear preview surface;
unchecked effect previews remain uncapped there, while transform and Eraser
previews retain their independent responsive 0.5x policy.
Commit and export always use the requested quality. Overall Opacity is native
group state and does not require rebuilding effect pixels. Reshape temporarily
omits effects, invalidates geometry-sensitive caches, and rebuilds the effective
namespace once geometry settles.

While native text editing is active, every Image node is intentionally omitted
for both writing modes so the editable source, selection, caret, IME, and
annotations remain coherent. Ending the edit restores the exact settled layer;
strict export still includes it. Selection and the deferred caret remain after
all completed pixel phases. Starting an Image chooser or discrete card
edit also deactivates the Eraser brush session.
Effective Image visibility is part of both the completed-surface and nonlinear
source cache identities, so editing cannot reuse settled Image pixels and
leaving edit cannot reuse the Image-free editing surface.
Ordinary Filters remain active during horizontal and vertical native editing;
selection, caret, and IME feedback are painted afterward and never enter their
input pixels.

Image generation is available only for exactly one concrete selected text
item. One request may run per panel. Source and Inpainted use the exact logical
pre-transform item rectangle from the corresponding page image; bounds must be
finite, positive, and fully inside the page, with no clamping. Lettered starts
from the Inpainted crop and draws only that item's logical horizontal or
vertical text, without effects, global transforms, selection, caret, or other
items. None bypasses crop validation and sends a prompt-only request. The LLM
adapter reuses the saved LLM Inpaint network policy without selecting or
loading the global inpainter; the request boundary can later accept a local
image-edit backend without changing crop, card, worker, or undo ownership.
LLMInpaint keeps its request-parameter schema as SafeEval-compatible literal
metadata, so this shared transport does not break lazy module discovery.

Generate runs on one retained parentless QObject backed by one daemon thread;
explicitly queued signals marshal results to Qt. Stop interrupts retry waits
cooperatively and keeps the panel in Stopping until an in-flight provider call
returns. The current synchronous in-flight HTTP call is not forcibly
interrupted; its eventual output is discarded. Selection,
history, page, scene, structure, or card changes detach the request target and
discard stale output. Mode and eye changes do not retarget generation; success
preserves their latest values. Application shutdown marks jobs abandoned and
does not wait for the provider timeout.

LLM image throttling is shared across short-lived Image generation backends and
LLM Inpaint. Delay and rolling requests-per-minute reservations are global;
waiting remains cooperative, and Stop is checked again after reservation before
provider dispatch.

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
| Final full surface or visible tiles | Separate committed, preview, and export namespaces; key by effective effects, mask generation, document/layout/render state, transform, writing mode, bounds, and quality tier |
| Complete pre-mask surface | At most two entries per namespace; reuse across mask-only previews while upstream effects and geometry match |
| Complete below-filter prefix | At most two entries per namespace; filter-only previews reuse the fixed base and retained nodes below the first effective Filter. The key follows exact node order and canonical Stroke dependencies of cached exterior layers. Upper generated layers are cheaply recomposited from retained canonical/coverage caches. |
| Canonical glyph pixmap and lazy alpha | At most two entries; reuse across paint-only, Fill Opacity, and Blend previews while document, layout, source geometry, transform state, and render scale match. Repeated fills share this source and its mask rather than rerasterizing glyphs. |
| Positioned Stroke coverage | At most two read-only alpha planes; key by canonical-source inputs plus Stroke width and position, excluding paint and opacity |
| Gradient compiled kernel | Runtime acceleration only; the byte-identical NumPy path remains the pre-warm, unavailable, and quality-oracle fallback |
| Decoded project raster | At most two positive entries per project, shared by Texture and Image and keyed by immutable relative ref; successful import prewarms it, project reload clears it, and failures are never cached. One top-level paint shares one asset-keyed Image lookup map through its gates and full/visible-tile composite. |

Within one composite, reuse the same colored positioned Stroke band for its
visible pass and exterior silhouette. Paint-only edits must not rerasterize the
native glyph outline. Shadow and Glow alpha are intentionally generated on
demand; their measured cost does not justify another invalidation surface.
Keep antialiasing enabled. Faster Preview's 0.5x tier provides the optional
speed/quality tradeoff; disabling painter antialiasing did not materially
reduce native outline cost. Large surfaces use the shared bounded full/tile
policy instead of unbounded allocation. Every namespace must release pixmaps
and arrays on reshape invalidation, item/page removal, and return to an inactive
path.

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
- `tests/test_image_effect.py`
- `tests/test_image_effect_rendering.py`
- `tests/test_image_generation.py`
- `tests/test_llm_inpaint.py`
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
