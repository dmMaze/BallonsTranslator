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
  -> ordered global stack: Slant / Perspective / Curvature
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
| Stage math, matrix adapter, composite mapper | [`ui/text_transform.py`](../../ballontranslator/ui/text_transform.py) |
| Variant registry and stack compiler | [`ui/text_transform_variants.py`](../../ballontranslator/ui/text_transform_variants.py) |
| Item transform and render lifecycle | [`ui/text_item_geometry.py`](../../ballontranslator/ui/text_item_geometry.py) |
| Selection-scoped preview and commit transactions | [`ui/text_transform_editor.py`](../../ballontranslator/ui/text_transform_editor.py) |
| Advanced-panel cards and controls | [`ui/text_transform_controls.py`](../../ballontranslator/ui/text_transform_controls.py), [`ui/text_advanced_format.py`](../../ballontranslator/ui/text_advanced_format.py) |
| Transform undo command | [`ui/textedit_commands.py`](../../ballontranslator/ui/textedit_commands.py) |
| Glyph Slant | [`ui/text_effects/transform_layout.py`](../../ballontranslator/ui/text_effects/transform_layout.py) |
| Curvature mapping and final surface warp | [`ui/text_effects/curvature.py`](../../ballontranslator/ui/text_effects/curvature.py) |
| Resize/rotation overlay | [`ui/texteditshapecontrol.py`](../../ballontranslator/ui/texteditshapecontrol.py) |

New variants extend the model, registry, and stage factory. Do not add
transform-type branches to `TextBlkItem` or `TextItemGeometryController`.

## Persistence and editing state

`TextTransform` subclasses are frozen canonical values with a stable
`transform_type`, normalized fields, `is_neutral()`, and runtime-only
`is_nonlinear` capability metadata.

Current global variants are:

- `SlantTextTransform`: affine box scale and shear;
- `PerspectiveTextTransform`: projective, but representable by `QTransform`;
- `CurvatureTextTransform`: nonlinear and mapper-based.

`TextTransformStack` is an immutable ordered tuple. Neutral entries remain in
the model for stable UI structure and persistence but are skipped at runtime.
`TextTransformState` combines the stack with `glyph_slant_angle`, so one undo
operation captures the complete visible state.

Project JSON stores only canonical values:

```json
{
  "text_transform": [
    {"transform_type":"slant","horizontal_scale":1.1,
     "vertical_scale":1.0,"slant_angle":8.0},
    {"transform_type":"curvature","curvature":0.35}
  ],
  "glyph_slant_angle": 10.0
}
```

Do not serialize preview values, matrices, mappers, bounds, layout generations,
or raster caches. Passive project loading discards invalid optional entries
according to `AGENTS.md`; live model and compiler boundaries use typed canonical
values.

### UI and undo

The Advanced panel generates transform cards from `TEXT_TRANSFORM_VARIANTS`.
Add appends, delete removes one indexed entry, and move swaps adjacent entries.
Typed values commit on Return/focus-out; label dragging previews transient state
and commits one command on release; Escape cancels the preview.

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

The matrix is installed through the existing item transform path.
`compensated_box_transform_matrix()` preserves the intended order of
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
so compilation is deferred until its edit block settles. Reusing compiled
output still requires reinstalling it because page/layout lifecycle code may
have detached the mapper or changed Qt's matrix.

## Mapper and rendering contract

A nonlinear stage must provide stable point and vectorized raster inversion:

```python
forward_point(source)
inverse_point(visual, previous_source=None, *, extrapolate=False)
inverse_arrays(visual_x, visual_y, *, return_valid=False)
geometry_key
```

- Forward mapping drives outlines, handles, bounds, and composition.
- Point inversion drives hit testing and resize.
- `previous_source` preserves branch continuity near seams.
- `extrapolate=True` is reserved for reshape beyond the visible mapped surface.
- Array inversion is the raster hot path and returns a validity mask when
  requested.
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
premultiplied alpha to avoid colored fringes.

Cache keys include mapping geometry, layout generation, Glyph Slant render
state, effect/background generation, and document revision. Idle settled
output may be cached. Editing, resize, and parameter previews do not retain a
final-surface cache. Raster size and quality remain bounded; interactive
allocation failure may degrade with a warning, while export failure is
reported after the Qt paint callback.

## Glyph Slant and effects

Glyph Slant uses Qt-produced glyph runs and leans each glyph around its own
baseline while preserving line breaks, advances, and cursor indices. A box
shear cannot provide those semantics. Outline glyphs use paths; pathless/color
glyphs use the bounded raster fallback.

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
position. After the warp, the controller queries the raw source caret rectangle,
maps it through the composite, and paints it over the completed destination
with `RasterOp_NotDestination`. Selection remains part of source capture.

### Resize

Every resize event is derived from one frozen drag-start coordinate frame.
`TextBlkShapeControl.beginResize()` stores the initial logical/absolute
rectangles, opposite scene anchor, source handle, and a frozen
scene-to-source mapper. Each pointer sample maps through that frozen transform,
updates a rectangle relative to the initial one, and restores the opposite
visual handle to its original scene position.

Only this reshape mapper uses branch-aware `extrapolate=True`. Ordinary text
hit testing retains bounded seam behavior. Mapping through geometry changed by
the previous event creates feedback and can collapse an outward curvature drag.

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
  ballontranslator/ui/text_transform.py \
  ballontranslator/ui/text_transform_variants.py \
  ballontranslator/ui/text_item_geometry.py \
  ballontranslator/ui/text_transform_editor.py

QT_API=pyqt6 QT_QPA_PLATFORM=offscreen \
  /opt/miniconda3/envs/common/bin/python -m unittest \
  discover -s tests -p 'test_text_transform_undo.py'

git diff --check
```

Painting changes also need a themed-app pass covering both writing modes,
neutral/matrix/nonlinear stacks, Glyph Slant with effects, typing and selection,
resize/rotation, zoomed interaction, and export.
