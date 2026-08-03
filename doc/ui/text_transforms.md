# Composable text transforms

Read [Text engine](text_engine.md) first. This guide records the transform
contracts that span persistence, UI transactions, geometry, rendering, and
interaction. The code and focused tests remain authoritative.

## Core model

There are two transform layers:

```text
QTextDocument + SceneTextLayout
  -> Glyph Slant around each shaped glyph baseline
  -> fill, stroke, shadow, gradient
  -> ordered global stack: Projective / Bend / Sine Wave / Grid
  -> item-local visual geometry
  -> QGraphicsItem position and rotation
```

Glyph Slant is a pre-stack typography effect and is not reorderable. The global
stack transforms the completed text/effect box, preserves order, and allows
duplicate types.

The compiler produces one global mapping:

- Matrix-only stages collapse to one native `QTransform`.
- If any active stage is nonlinear, every active stage enters one
  `CompositeTextTransformMapper` and the completed source surface is warped
  once.

Never install matrix stages natively as well as inside a composite mapper.
Item position and built-in rotation remain outside the stack.

## Owners

| Concern | Owner |
| --- | --- |
| Immutable values, stack, state, persistence | [`utils/fontformat.py`](../../ballontranslator/utils/fontformat.py) |
| Stage math, matrix adapter, composite mapper | [`ui/text_engine/transforms/mapping.py`](../../ballontranslator/ui/text_engine/transforms/mapping.py) |
| Variant registry and stack compiler | [`ui/text_engine/transforms/registry.py`](../../ballontranslator/ui/text_engine/transforms/registry.py) |
| Item transform and render lifecycle | [`ui/text_engine/geometry.py`](../../ballontranslator/ui/text_engine/geometry.py) |
| Selection-scoped preview and commit transactions | [`ui/text_engine/transforms/editor.py`](../../ballontranslator/ui/text_engine/transforms/editor.py) |
| Expandable transform panel, cards, and controls | [`ui/text_engine/transforms/panel.py`](../../ballontranslator/ui/text_engine/transforms/panel.py), [`ui/text_engine/transforms/controls.py`](../../ballontranslator/ui/text_engine/transforms/controls.py) |
| Transform undo command | [`ui/text_engine/editing/commands.py`](../../ballontranslator/ui/text_engine/editing/commands.py) |
| Glyph Slant | [`ui/text_engine/rendering/glyph_slant.py`](../../ballontranslator/ui/text_engine/rendering/glyph_slant.py) |
| Bend mapping and final surface warp | [`ui/text_engine/transforms/bend.py`](../../ballontranslator/ui/text_engine/transforms/bend.py), [`ui/text_engine/rendering/surface.py`](../../ballontranslator/ui/text_engine/rendering/surface.py) |
| Sine Wave mapping | [`ui/text_engine/transforms/sine.py`](../../ballontranslator/ui/text_engine/transforms/sine.py) |
| Grid mapping and selected-stage overlay | [`ui/text_engine/transforms/grid.py`](../../ballontranslator/ui/text_engine/transforms/grid.py), [`ui/text_engine/transforms/grid_control.py`](../../ballontranslator/ui/text_engine/transforms/grid_control.py) |
| Projective matrix and selected-stage overlay | [`ui/text_engine/transforms/mapping.py`](../../ballontranslator/ui/text_engine/transforms/mapping.py), [`ui/text_engine/transforms/projective_control.py`](../../ballontranslator/ui/text_engine/transforms/projective_control.py) |
| Resize/rotation overlay | [`ui/text_engine/shape_control.py`](../../ballontranslator/ui/text_engine/shape_control.py) |

New variants extend the model, registry, and stage factory. Do not add
transform-type branches to `TextBlkItem` or `TextItemGeometryController`.

## Persistence and editing state

`TextTransform` subclasses are frozen canonical values with a stable
`transform_type`, normalized fields, `is_neutral()`, and runtime-only
`is_nonlinear` capability metadata.

Current global variants are:

- `ProjectiveTextTransform`: horizontal/vertical scale and sequential slant,
  X/Y/Z planar rotation, and normalized perspective, compiled together into
  one centered native `QTransform`;
- `BendTextTransform`: nonlinear and mapper-based;
- `SineTextTransform`: nonlinear horizontal and vertical sine shears. Integer
  frequencies count half-waves from 0 to 64; phase and perpendicular-box
  amplitude use 0 to 1. The x-axis wave runs first so the combined map
  has an exact inverse;
- `GridTextTransform`: nonlinear free-form deformation with normalized control
  points, 1 to 32 horizontal and vertical cell divisions, and Straight or
  Smooth interpolation. A 1 by 1 grid has four corner handles. Their canonical
  interpolation values are `bilinear` and `catmull_rom`.

`TextTransformStack` is an immutable ordered tuple. Neutral entries remain in
the model for stable UI structure and persistence but are skipped at runtime.
`TextTransformState` combines the stack with `glyph_slant_angle`, so one undo
operation captures the complete visible state.

Project JSON stores only canonical values:

```json
{
  "text_transform": [
    {"transform_type":"projective","horizontal_scale":1.1,
     "vertical_scale":1.0,"horizontal_slant":8.0,"vertical_slant":0.0,
     "rotation_x":15.0,"rotation_y":-20.0,"rotation_z":5.0,
     "perspective":0.4},
    {"transform_type":"bend","bend":0.35}
  ],
  "glyph_slant_angle": 10.0
}
```

Do not serialize preview values, matrices, mappers, bounds, layout generations,
or raster caches. Passive project loading discards invalid optional entries
according to `AGENTS.md`; live model and compiler boundaries use typed canonical
values.

### UI and undo

The Text Transform panel generates transform cards from
`TEXT_TRANSFORM_VARIANTS`.
Add appends and selects the new entry, delete removes one indexed entry, and
move swaps adjacent entries.
Typed values commit on Return/focus-out; label dragging previews transient state
and commits one command on release; Escape cancels the preview. Integer labels
move one step per eight pixels, and every drag saturates at the shared valid range
without accumulating hidden overshoot.

Cards have one selected index. Clicking a card or manually interacting with one
of its parameters selects it; selecting another card replaces that selection,
and deleting the selected entry clears it. Selecting a Grid or Projective card
for exactly one text block binds its global overlay and hides the normal shape
overlay. The two transform overlays are mutually exclusive and follow the same
selection, deletion, item-switch, and text-editing lifecycle.
Circle handles use Ctrl or Shift to toggle selection, rubber-band selection can
start with either mouse button inside the grid or on the surrounding canvas,
and dragging any selected handle moves the selected set in one preview and one
undo transaction. Grid-owned right-button selection is consumed without
opening the Canvas context menu.

The Canvas owns one scene-space rubber-band gesture and visual. Normal canvas
selection applies it to scene items while an active Grid delegates completion
to the Grid controller's handle-selection rule; Grid does not keep a second
rectangle or gesture state.

With at least one Grid handle selected, `G`, `R`, and `S` start reusable modal
Move, Rotate, and Scale operations. The initial mouse-to-selection offset is
preserved. Left click commits one undo command; right click or Escape restores
the operation-start points without creating a command. Starting another modal
operation also restores those points before switching modes. During Move, `X`
or `Y` restores the start points and constrains subsequent movement to the
corresponding canvas axis. Rotate and Scale use the selected-handle center and
show a dotted origin-to-pointer guide; constrained movement shows its active
axis line.

The Projective controller is fixed in device pixels and follows the selected
stage's fixed matrix pivot without growing with text geometry or canvas zoom.
Its three mutually perpendicular X, Y, and Z rings use one small display-only
X/Y tilt so no axis collapses under the fixed front view. Perspective previews
must not recenter the controller from their asymmetric visual bounding box.
Direct ring dragging edits that rotation axis.
`R` defaults to Z; pressing X, Y, or Z restores the operation-start transform
and constrains rotation to that axis. `S` scales both dimensions; X or Y resets
and constrains the corresponding pre-rotation scale. Switching between `R` and
`S` also restores the start transform. Left click commits one undo command;
right click or Escape cancels.

The selected Grid overlay batch-maps handle coordinates through the compiled
stage suffix. Its guide lines are one transient raster warped by that same
mapper instead of thousands of scalar scene-path mappings; the overlay remains
global UI state and is never included in text export.

Grid control points are stored as normalized coordinates, so font, spacing,
writing-mode, and text-box geometry changes rebuild the stage against settled
logical bounds instead of leaving the controller attached to stale pixels.
Straight interpolation is bilinear within each cell. Smooth interpolation uses a
tensor-product Catmull-Rom interpolation: it passes through every handle while
neighboring handles curve the coordinates between them. A 1 by 1 grid produces
the same result in either mode because it has no interior neighbors to create
additional bending.

Before a structural edit, `TextTransformEditSession` commits pending typed
values and cancels previews so indices cannot move under an active control.
Save, undo/redo, page change, and scene replacement also resolve transient
state.

For selected items, one user action creates one `SetTextTransformCommand` with
complete before/after states for every target. That command owns transform
state and overlay refresh only. It must not consume `QTextDocument` history or
modify paired-editor text; `TextEditCommand`, `TextItemEditCommand`, and
`SceneTextManager` own those paths.

Multi-selection exposes indexed controls only when all targets share the same
sequence of transform types. Matching indices may have mixed values. When
stack shapes differ, existing indices are not reinterpreted; append remains
safe.

## Compiler and installation

`compile_text_transform_stack()` is the policy boundary. For every active
operation in forward order it:

1. builds context from current logical bounds, padded source bounds, and
   writing mode;
2. asks the registered stage factory for a matrix or mapper;
3. validates matrix finiteness, invertibility, and projective horizon safety;
4. folds adjacent matrix stages;
5. advances bounds only when a later stage needs them.

Later stages see earlier output bounds, so reordering is semantically visible.

### Matrix-only stack

```text
active matrix stages
  -> combined QTransform
  -> CompiledTextTransform.native_matrix
  -> no surface mapper
```

The matrix is installed through the existing item transform path. One
Projective stage computes its complete homography when its parameters or input
bounds change; painting never assembles its scale, slant, rotation, or depth
components. `compensated_native_transform_matrix()` preserves the intended order of
item-local stack transform followed by Qt's built-in item rotation. Keep its
identity and zero-rotation fast paths exact; floating residue in a neutral
matrix can activate unnecessary custom geometry and cache behavior.

### Stack containing a nonlinear stage

```text
all active stages
  -> MatrixTransformMapper / nonlinear mapper stages
  -> one CompositeTextTransformMapper
  -> identity native stack matrix
  -> one final inverse-sampled surface
```

The controller compiles by immutable stack, writing mode, logical rectangle,
and padded source rectangle. Rich-text formatting can emit intermediate sizes,
so compilation is deferred until its edit block settles. The settled flush
publishes `visual_geometry_changed` after installing the rebuilt mapper, which
keeps shape and Grid overlays synchronized for every transform variant and
formatting setter. Reusing compiled output still requires reinstalling it
because page/layout lifecycle code may have detached the mapper or changed Qt's
matrix.

## Mapper and rendering contract

A nonlinear stage must provide stable point and vectorized raster inversion:

```python
forward_point(source)
forward_arrays(source_x, source_y)
inverse_point(visual, previous_source=None, *, extrapolate=False)
inverse_arrays(visual_x, visual_y, *, return_valid=False)
visual_bounds(source_rect=None)
geometry_key
```

- Forward mapping drives outlines, handles, bounds, and composition.
- Array forward mapping batches dense controller and overlay geometry.
- Point inversion drives hit testing and resize.
- `previous_source` preserves branch continuity near seams.
- `extrapolate=True` is reserved for reshape beyond the visible mapped surface.
- Array inversion is the raster hot path and returns a validity mask when
  requested.
- Visual bounds must include interior extrema, not only the mapped outer edge.
- `geometry_key` includes every input that changes mapping.

The composite applies forward stages in order and inverse stages in reverse.
Painting, outline geometry, hit testing, cursor mapping, and resize must share
this same mapping boundary.

The effective render paths are:

| State | Layout paint | Global geometry | Final warp |
| --- | --- | --- | --- |
| Neutral | Native | Identity | No |
| Matrix-only stack | Native | Combined matrix | No |
| Glyph Slant, no nonlinear stage | Custom glyph layout | Identity or matrix | No |
| Any active nonlinear stage | Native or custom glyph layout | Composite mapper | Once |

For nonlinear output, `TextItemGeometryController.paint_item()` captures fill,
gradient, stroke, and shadow into one padded source pixmap, inverse-maps it in
bounded row bands, draws it once, then overlays the mapped caret. Sampling uses
premultiplied alpha to avoid colored fringes. Settled text uses cubic sampling
at a raster tier no smaller than the device scale when the bounded allocation
policy permits; live parameter and resize previews retain the cheaper bilinear
path.

Cache keys include mapping geometry, layout generation, Glyph Slant render
state, effect/background generation, document revision, and live selection
state. The renderer separately caches inverse remap coordinates by mapper
geometry, source/destination rectangles, and render scale. Text, selection,
and IME changes regenerate surface pixels through that existing map; mapper
geometry changes and item/page release discard it. Editing retains the final
surface: caret-only repaints reuse it and run a transparent source-layout probe
to preserve Qt's native blink visibility, selection changes select a new
surface key, and each IME event explicitly invalidates transient preedit
pixels. Resize and parameter previews remain uncached. Raster size and quality
remain bounded; interactive allocation failure may degrade with a warning,
while export failure is reported after the Qt paint callback.

Grid's dense Newton inverse uses separately compiled bilinear and Catmull-Rom
Numba kernels after an asynchronous launch warm-up. Numba owns cache validation and
stores the signatures under `.btrans_cache/numba`, which survives application
updates; a missing, stale, or incompatible entry is compiled in the background.
Until warm-up succeeds, Grid keeps using the NumPy inverse so the Qt thread
never waits for compilation.

## Glyph Slant and effects

Glyph Slant uses Qt-produced glyph runs and leans each glyph around its own
baseline while preserving line breaks, advances, and cursor indices. A box
shear cannot provide those semantics. Outline glyphs use paths; pathless/color
glyphs use the bounded raster fallback. For vertical characters that Qt lays
out with a rotated orientation, that orientation is applied before the
item-space slant so the visible glyph, rather than its unrotated source axes,
is slanted.

The fixed order is:

```text
Qt shaping -> Glyph Slant -> fill/stroke/shadow -> global stack
```

Do not make Glyph Slant reorderable without redesigning shaping, cursor
geometry, effect masks, UI, undo, and persistence. Fill and effects must reuse
the same slanted glyph geometry.

Effect padding participates in the source rectangle and may grow or shrink. It
is derived layout state and must not create document history. Returning Glyph
Slant or the stack to neutral must restore native effects, refresh gradient
geometry, clear transformed surfaces, and release transformed-only owners.

## Interaction invariants

### Hit testing, selection, and caret

Qt's text control remains the editor. Visual input is inverse-mapped to source
layout coordinates; `inputMethodQuery()` maps point and rectangle results back
to visual space.

Qt normally invalidates source-local dirty rectangles. A nonlinear warp can
move selection and caret pixels outside them, so editing changes request a full
`TextBlkItem.update()` while a surface warp is active.

During source capture, both layouts defer caret painting and retain the cursor
position. After the warp, the controller uses the layout-owned horizontal
caret for vertical text and Qt's native rectangle for horizontal text, maps it
through the composite, and paints it over the completed destination with
`RasterOp_NotDestination`. Selection remains part of source capture.

### Resize

Every resize event is derived from one frozen drag-start coordinate frame.
`TextBlkShapeControl.beginResize()` stores the initial logical/absolute
rectangles, opposite scene anchor, source handle, and a frozen
scene-to-source mapper. Each pointer sample maps through that frozen transform,
updates a rectangle relative to the initial one, and restores the opposite
visual handle to its original scene position.

Only this reshape mapper uses branch-aware `extrapolate=True`. Ordinary text
hit testing retains bounded seam behavior. Mapping through geometry changed by
the previous event creates feedback and can collapse an outward bend drag.

## Lifecycle and optimization rules

- Skip neutral entries but retain them in model/UI state.
- Fold matrix stages and keep the native path when no nonlinear stage is active.
- With nonlinear stages, compose mappings and warp the completed surface once.
- Compile once per distinct stack, writing mode, logical rect, and source rect.
- Keep preview and committed glyph caches separate and bounded.
- Increment layout generation only for layout geometry changes.
- Do not retain final-surface caches during active interaction.
- Release glyph, effect, and surface namespaces when items/pages are removed.
- Treat activation and return to neutral as symmetric lifecycle transitions.
- Test horizontal and vertical writing with effects, editing, resize, rotation,
  and export.

## Adding a variant

1. Add a frozen canonical `TextTransform` subclass and stable
   `transform_type` in `fontformat.py`; define exact neutral state and set
   `is_nonlinear` only when `QTransform` cannot represent it.
2. Register the model type and a `TextTransformVariantSpec` with localized
   controls.
3. Implement one stage factory returning a validated matrix or a complete
   mapper using the incoming stage bounds.
4. Keep model and runtime registry keys equal and preserve permissive project
   loading of invalid optional entries.
5. Test persistence, neutral/active states, order, duplicates, matrix/nonlinear
   composition, preview/cancel/commit, multi-selection undo, both writing modes,
   effects, input mapping, resize, and export.

A nonlinear variant without stable point inversion and vectorized array
inversion does not fit this architecture.

## Focused verification

[`tests/test_text_transform_undo.py`](../../tests/test_text_transform_undo.py)
covers persistence, registry/UI structure, multi-item undo and paired-editor
isolation, compiler composition, rendering paths, cache lifecycle,
cursor/selection mapping, and frozen-coordinate resize.

```bash
python -m py_compile \
  ballontranslator/utils/fontformat.py \
  ballontranslator/ui/text_engine/geometry.py \
  ballontranslator/ui/text_engine/transforms/mapping.py \
  ballontranslator/ui/text_engine/transforms/registry.py \
  ballontranslator/ui/text_engine/transforms/editor.py \
  ballontranslator/ui/text_engine/transforms/projective_control.py

QT_API=pyqt6 QT_QPA_PLATFORM=offscreen \
  /opt/miniconda3/envs/common/bin/python -m unittest \
  discover -s tests -p 'test_text_transform_undo.py'

git diff --check
```

Painting changes also need a themed-app pass covering both writing modes,
neutral/matrix/nonlinear stacks, Glyph Slant with effects, typing and selection,
resize/rotation, zoomed interaction, and export.
