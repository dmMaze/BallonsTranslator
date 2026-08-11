# Composable text transforms

Read [Text engine](text_engine.md) first. This guide is an orientation to the
transform subsystem: where it fits, who owns each part, and which contracts are
easy to break. The code and focused tests are authoritative for individual
controls and algorithms.

## Mental model

```text
QTextDocument + SceneTextLayout
  -> Glyph Slant around each shaped glyph's visible-space anchor
  -> fill, stroke, shadow, gradient
  -> ordered global stack: Projective / Bend / Sine Wave / Grid
  -> QGraphicsItem position and rotation
```

Glyph Slant is a typography effect applied before the global stack. It is not
reorderable. The global stack transforms the completed text-and-effects box;
its order is significant and duplicate transform types are allowed.

Upright glyphs slant around their mapped baseline. A quarter-turned glyph has
no horizontal visible baseline, so its original ink-to-baseline distance is
re-established on the visible y axis. This keeps the intended slant direction
without translating rotated punctuation away from the Roman-glyph column.
The final ink box is centered on that anchor so mirrored outlines do not drift
in opposite directions.

`compile_text_transform_stack()` is the transform-stack compiler. It receives
the committed stack plus the current logical bounds, padded source
bounds, and writing mode. Registered stage factories turn each active entry
into a matrix or nonlinear mapper, and the compiler returns one
`CompiledTextTransform` for `TextItemGeometryController` to store and install.

| Compiled output | Purpose |
| --- | --- |
| `native_matrix` | Qt item transform for a matrix-only stack; identity on the nonlinear path |
| `surface_mapper` | Complete nonlinear mapping used for painting and visual/input geometry |
| `stages` | Per-entry input context and optional mapper used to position and edit selected-stage controls inside the full stack |

That one result drives painting, bounds, hit testing, cursor/IME mapping,
resize, and selected-stage overlays. It has two execution outcomes:

- With no nonlinear stage, matrix stages collapse into one native
  `QTransform`.
- With any nonlinear stage, all active stages enter one
  `CompositeTextTransformMapper`, and the completed source surface is warped
  once.

Never install a matrix stage natively as well as inside the composite mapper.
Item position and built-in rotation remain outside the stack.

Projective currently follows the matrix path; Bend, Sine Wave, and Grid follow
the nonlinear path. Glyph Slant remains the separate pre-stack effect shown
above.

## Owners

| Concern | Owner |
| --- | --- |
| Model values, stack, and persistence | [`utils/fontformat.py`](../../ballontranslator/utils/fontformat.py) |
| Stage math and composite mapping | [`ui/text_engine/transforms/`](../../ballontranslator/ui/text_engine/transforms/) |
| Variant registration and compilation policy | [`ui/text_engine/transforms/registry.py`](../../ballontranslator/ui/text_engine/transforms/registry.py) |
| Item geometry, installed mapping, and render lifecycle | [`ui/text_engine/geometry.py`](../../ballontranslator/ui/text_engine/geometry.py) |
| Selection-scoped preview and commit | [`ui/text_engine/transforms/edit_session.py`](../../ballontranslator/ui/text_engine/transforms/edit_session.py) |
| Panel and variant controls | [`ui/text_engine/transforms/panel.py`](../../ballontranslator/ui/text_engine/transforms/panel.py), [`ui/text_engine/transforms/controls.py`](../../ballontranslator/ui/text_engine/transforms/controls.py) |
| Canvas undo and paired-editor coordination | [`ui/text_engine/editing/commands.py`](../../ballontranslator/ui/text_engine/editing/commands.py), [`ui/text_engine/editing/manager.py`](../../ballontranslator/ui/text_engine/editing/manager.py) |
| Shared rubber-band and modal-event routing | [`ui/canvas.py`](../../ballontranslator/ui/canvas.py) |
| Final surface warp and Glyph Slant painting | [`ui/text_engine/rendering/`](../../ballontranslator/ui/text_engine/rendering/) |
| Selected-stage canvas overlays | [`ui/text_engine/transforms/grid_control.py`](../../ballontranslator/ui/text_engine/transforms/grid_control.py), [`ui/text_engine/transforms/projective_control.py`](../../ballontranslator/ui/text_engine/transforms/projective_control.py) |

New variants extend the model, registry, and stage factory. Do not add
variant-specific branches to `TextBlkItem` or `TextItemGeometryController`.

## State, persistence, and undo

`TextTransform` subclasses are immutable model values with a stable
`transform_type`, an exact neutral state, and runtime-only `is_nonlinear`
capability metadata. UI controls constrain edits to their supported ranges and
canonical precision before producing model values. The model does not clamp or
range-validate persisted parameters. `TextTransformStack` is an immutable value
containing the ordered operation tuple and Glyph Slant angle; neutral entries
remain in model and UI state but are skipped by the compiler.

`TextTransformStack` combines the ordered global operations with the fixed
pre-stack `glyph_slant_angle`; undo snapshots that one immutable value so it
restores the complete visible transform state.
Project JSON stores only this committed model data. Preview values,
matrices, mappers, bounds, and caches are derived state and must not be
serialized.

Persisted transform entries use a registered type and known fields; parameter
values are assumed valid, and omitted fields may use the variant's defaults.
Passive loading still ignores structurally unknown transform entries.

`TextTransformEditSession` owns transient UI state. Typed edits commit at their
normal editing boundary; drags preview and then create one command or cancel.
Before a structural edit, save, undo/redo, page change, or scene replacement,
resolve pending values and previews so stack indices cannot move underneath
active controls.

```text
panel or canvas control
  -> TextTransformEditSession transient preview
     -> cancel: restore prior state
     -> commit: create one SetTextTransformCommand
        -> committed state on each selected item
        -> geometry compilation, painting, and overlay refresh
```

One user action creates one `SetTextTransformCommand` containing complete
before/after transform states for its targets. This command owns transform
state and overlay refresh only. It must not consume `QTextDocument` history or
modify paired-editor text; text-edit commands and `SceneTextManager` own those
paths.

For multiple selected items, indexed controls are meaningful only when all
targets have the same sequence of transform types; matching indices may still
show mixed values. Do not reinterpret existing indices when stack shapes
differ, although appending a new stage remains safe.

## Compilation, geometry, and rendering

Each stage is built against the bounds produced by the preceding stages, so
reordering changes the result. Compilation depends on the immutable stack,
writing mode, logical rectangle, and padded source rectangle. Rich-text edits
can expose intermediate sizes; defer compilation until the edit settles, then
publish the installed geometry once.

The compiler rejects non-finite or non-invertible matrices and projective
transforms that cross their source horizon, and it folds adjacent matrix
stages. Projective scale, slant, rotation, and perspective form one homography
during compilation; painting must not reconstruct those components.

```text
matrix-only stack                    stack with a nonlinear stage
-----------------                    ----------------------------
active matrix stages                 all active stages
  -> one combined QTransform           -> matrix adapters + nonlinear mappers
  -> native item transform              -> one composite mapper
  -> no surface warp                    -> identity native stack transform
                                         -> one final surface warp
```

Qt applies its built-in item rotation and base transform in an order different
from the stack's semantics. The native path therefore installs a compensated
matrix so the item-local stack still precedes built-in rotation. Preserve exact
identity and zero-rotation fast paths; floating residue can incorrectly enable
custom geometry and caches.

After deferred compilation, emit `visual_geometry_changed` only after
installing the settled geometry so shape and transform overlays never observe
stale bounds.

A nonlinear mapper must provide the complete interaction and raster contract:

```python
forward_point(source)
forward_arrays(source_x, source_y)
inverse_point(visual, previous_source=None, *, extrapolate=False)
inverse_arrays(visual_x, visual_y, *, return_valid=False)
visual_bounds(source_rect=None)
map_rect_path(source_rect)
geometry_key
```

`previous_source` preserves branch continuity near folds. Extrapolation is only
for reshape beyond the visible mapped surface; ordinary hit testing stays
bounded. Array inversion supplies a validity mask for raster and dense overlay
work. Bounds must include interior extrema, and `geometry_key` must contain
every input that changes the mapping.

Forward stages run in order and inverse stages in reverse. Painting, outlines,
handles, hit testing, cursor/IME geometry, and resize must all cross the same
mapping boundary.

The effective render paths are:

| State | Layout paint | Global geometry | Final warp |
| --- | --- | --- | --- |
| Neutral | Native | Identity | No |
| Matrix-only stack | Native | Combined matrix | No |
| Glyph Slant, no nonlinear stage | Custom glyph layout | Identity or matrix | No |
| Any active nonlinear stage | Native or custom glyph layout | Composite mapper | Once |

For nonlinear output, capture fill, gradient, stroke, and shadow into one
padded source surface, inverse-map it once, and draw editing UI over the mapped
destination where necessary. Matrix-only stacks stay on Qt's native path.
Effect padding changes the source rectangle, not the persistent logical text
rectangle.

Stroke and shadow use the same bounded, device-scale-aware effect raster
policy for neutral, Glyph Slant, matrix, and nonlinear paths. Effects and
delegated Glyph Slant painting disable Qt's redundant outer item cache; the
effect renderer remains the sole raster-cache owner and can therefore rebuild
at the current view or export scale.

### Cache boundaries

The subsystem caches work at four boundaries because geometry and pixels
change independently:

| Cached work | Reuse and invalidation boundary |
| --- | --- |
| Compiled transform | Reuse while stack, writing mode, logical rectangle, and source rectangle match. Reapply cached output because page or layout lifecycle code may have detached the mapper or changed Qt's matrix. |
| Glyph geometry | Keep committed and preview geometry separate and bounded. Layout generation and Glyph Slant render state determine reuse. |
| Nonlinear inverse-remap coordinates | Reuse across text, selection, and IME changes while mapper geometry, source/destination rectangles, and render scale match. |
| Final nonlinear surface pixels | Key by layout, glyph/effect state, document content, and selection. Do not retain this cache during parameter or resize previews. |

A caret-only repaint may reuse final surface pixels, but it must run a
transparent source-layout probe so Qt updates native blink visibility.
Selection belongs in the surface key, while every IME event explicitly
invalidates transient preedit pixels.

All transform cache namespaces and item-owned render caches must be releasable
on item/page removal and return to neutral. Grid's optional compiled inverse
kernels are runtime acceleration, not transform-result caches: warm them
outside the Qt thread and keep the working fallback until they are ready.

## Interaction invariants

Qt's text control remains authoritative for shaping, cursor, selection, IME,
and document history. Visual input is inverse-mapped into source-layout
coordinates; output rectangles and overlays are mapped forward. 

The following details are easy to regress during otherwise local changes:

- A nonlinear warp can move editing pixels outside Qt's source-local dirty
  rectangles, so cursor, selection, and IME changes must repaint the whole
  `TextBlkItem`. Keep editing UI on the same source-to-visual mapping boundary
  as the text.
- Resize maps every pointer sample through geometry frozen at drag start.
  Reusing geometry changed by an earlier sample creates feedback and can
  reverse or collapse an outward drag.
- The panel and canvas overlays share one selected stack index. A selected Grid
  or Projective stage for exactly one text block owns the transform overlay and
  hides the normal shape overlay. These overlays are mutually exclusive,
  follow selection and editing lifecycle, and never appear in export.
- Grid-owned right-button selection is consumed without opening the Canvas
  context menu.
- The Canvas owns the only scene-space rubber-band gesture and visual. An
  active Grid reuses it for handle selection; do not add a second rectangle or
  mouse lifecycle.
- Grid control points are normalized. Text, font, writing-mode, or box changes
  rebuild their stage against settled bounds rather than retaining stale pixel
  coordinates.
- Glyph Slant remains outside `QTextDocument` layout so Qt keeps ownership of
  shaping, wrapping, cursor indices, and selection. Fill, effects, and bounds
  must reuse the same slanted glyph geometry.
- Activation and return to neutral are symmetric lifecycle transitions: native
  effects and geometry must be restored, gradient geometry refreshed, and
  transformed-only state released.

## Adding a variant

1. Add an immutable `TextTransform` subclass with a stable
   `transform_type`, exact neutral state, and correct `is_nonlinear` capability.
2. Register its model type, localized controls, and stage factory under the
   same stable key.
3. Return either a validated matrix or a mapper satisfying the complete
   geometry contract; use the incoming stage bounds.
4. Preserve tolerant loading of invalid optional project data and avoid new
   item/controller type branches.
5. Cover persistence, neutral/active lifecycle, stack order, preview and
   cancel/commit, undo isolation, both writing modes, effects, interaction, and
   export as applicable.

A nonlinear variant without stable point inversion and vectorized array
inversion does not fit this architecture.

## Focused verification

[`tests/test_text_transform_undo.py`](../../tests/test_text_transform_undo.py)
is the main focused regression suite. Run its relevant tests first, then broaden
according to the ownership boundaries changed:

```bash
QT_API=pyqt6 QT_QPA_PLATFORM=offscreen \
  /opt/miniconda3/envs/common/bin/python -m unittest \
  discover -s tests -p 'test_text_transform_undo.py'

git diff --check
```

Rendering or interaction changes still need a themed-app pass covering the
affected writing modes and neutral, matrix, and nonlinear states.
