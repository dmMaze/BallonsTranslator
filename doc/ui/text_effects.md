# Text effects

Read [Text engine](text_engine.md) first. This guide documents the contracts
that make the effect stack safe to extend. The code and focused tests remain
authoritative for individual controls and raster algorithms.

## Mental model

```text
QTextDocument + settled SceneTextLayout
  -> canonical glyph source, including Glyph Slant
  -> canonical foreground or the Gradient/Texture foreground group
  -> movable Image / Stroke / Shadow / Glow / Filter cards in panel order
  -> TextBlock alpha mask (Eraser)
  -> overall Opacity
  -> global Text Transform stack
  -> item position and rotation
  -> selection, caret, and IME feedback
```

The panel is an application-order projection, not an arbitrary layer graph.
Its first movable card runs first and a newly appended card runs last. The
persisted tuple remains topmost-first for compatibility, so renderer traversal
is reversed. Gradient and Texture form a separate structural foreground group;
Hollow, Eraser, and Opacity are fixed controls rather than movable cards.

Effects compose inside the isolated text-item surface, never against the page
backdrop. Filters transform all pixels accumulated before their card. Generated
effects keep explicit geometry sources: changing order does not turn arbitrary
filtered or Image pixels into a glyph outline.

## Owners

| Concern | Owner |
| --- | --- |
| Immutable values, stack order, validation, and tolerant loading | [`utils/text_effects.py`](../../ballontranslator/utils/text_effects.py) |
| Style persistence and legacy migration | [`utils/fontformat.py`](../../ballontranslator/utils/fontformat.py) |
| Generic raster references, project import, and decode cache | [`utils/raster_assets.py`](../../ballontranslator/utils/raster_assets.py), [`utils/proj_imgtrans.py`](../../ballontranslator/utils/proj_imgtrans.py) |
| Block-owned Eraser state | [`utils/text_alpha_mask.py`](../../ballontranslator/utils/text_alpha_mask.py), [`utils/textblock.py`](../../ballontranslator/utils/textblock.py) |
| Composition, padding, preview namespaces, and raster caches | [`ui/text_engine/effects/renderer.py`](../../ballontranslator/ui/text_engine/effects/renderer.py) |
| Filter discovery and implementations | [`ui/text_engine/effects/filters/`](../../ballontranslator/ui/text_engine/effects/filters/), [Text filters](text_filters.md) |
| Selection-scoped preview and commit | [`ui/text_engine/effects/edit_session.py`](../../ballontranslator/ui/text_engine/effects/edit_session.py) |
| Panel projection and cards | [`ui/text_engine/effects/panel.py`](../../ballontranslator/ui/text_engine/effects/panel.py), [`ui/text_engine/effects/cards.py`](../../ballontranslator/ui/text_engine/effects/cards.py) |
| Eraser brush input and mask undo | [`ui/text_engine/effects/alpha_mask_edit_session.py`](../../ballontranslator/ui/text_engine/effects/alpha_mask_edit_session.py) |
| Image-generation crop, request, and worker | [`ui/text_engine/effects/image_generation.py`](../../ballontranslator/ui/text_engine/effects/image_generation.py), [`modules/llm_image.py`](../../ballontranslator/modules/llm_image.py) |
| Source/visual bounds and global mapping | [`ui/text_engine/geometry.py`](../../ballontranslator/ui/text_engine/geometry.py), [`ui/text_engine/transforms/`](../../ballontranslator/ui/text_engine/transforms/) |

The panel only projects values and emits edit requests. `TextBlkItem` is the Qt
boundary, `TextEffectRenderer` owns the completed effect surface, and the
geometry controller owns its final mapping. Do not introduce a parallel effect
model, renderer, or edit session.

## Values, order, and sources

`FontFormat.text_effects` is one immutable `TextEffectStack`. Stroke, Shadow,
Glow, Gradient, Texture, Image, and Filter are repeatable; Hollow is unique.
`overall_opacity` applies after the completed stack. Neutral or disabled values
remain persisted and visible but must not allocate surfaces when they cannot
change output.

| Effect | Source and composition contract |
| --- | --- |
| Gradient / Texture | Repeatable paints composed in their own visible order, clipped once by canonical glyph coverage, and used in place of rich foreground. If none can render, rich foreground remains. An enabled transparent Gradient can intentionally erase the face; an Empty or missing Texture is neutral interactively. |
| Stroke | Canonical glyph alpha. Width retains the historical native-outline meaning; Center splits it across the edge, while Outside and Inside clip that same width to one side. New and legacy-flat strokes default to Outside. |
| Drop / Long Shadow | Canonical glyph alpha plus enabled Stroke cards already applied. Output is clipped outside the canonical face. |
| Inner Shadow | Canonical glyph alpha only; suppressed by Hollow. |
| Outer Glow | Canonical glyph alpha plus enabled Stroke cards already applied. |
| Inner Glow | Canonical glyph alpha only; suppressed by Hollow. |
| Hollow | Suppresses the canonical face, foreground-paint group, and interior effects while retaining Stroke and exterior output. |
| Image | Repeatable project raster. In Front uses source-over; Behind uses destination-over. It does not become a generated-effect source. Empty is neutral. |
| Filter | Transforms the accumulated RGBA result at its position. Alpha is non-expanding unless the plug-in declares and bounds expansion. |
| Eraser | Multiplies the completed stack by the item-owned alpha mask. It is not reusable `FontFormat` data. |

Stroke, Shadow, Glow, Gradient, and Texture accept Normal and the Darken and
Lighten blend families. Persistence stores the flat leaf mode; submenu families
are presentation only. Paint alpha, stop alpha, and effect Opacity each apply
once. Custom blend leaves may require a straight-RGBA bridge, but must preserve
the same output and order in full, tiled, preview, and export paths.

Gradient coordinates and Texture mapping are anchored to the unpadded logical
rectangle, not individual glyphs or tiles. Project raster references point only
inside the managed `assets/` directory. Import snapshots, validates, hashes,
and atomically installs a bounded RGBA8 asset. Texture and Image share the
project decode cache and premultiplied-alpha sampling path.

## Rendering and geometry

The renderer captures canonical glyph pixels once and compiles the effective
stack:

1. Build canonical rich foreground or the Gradient/Texture group unless Hollow
   suppresses it.
2. Walk movable cards in displayed application order. Painter-compatible layers
   and consecutive Filters are batched without changing semantic boundaries.
3. Generate Stroke, Shadow, and Glow from their declared canonical or
   Stroke-inclusive alpha source, not from accumulated arbitrary pixels.
4. Apply the TextBlock alpha mask and overall Opacity.
5. Hand one padded source surface to the global transform path described in
   [Composable text transforms](text_transforms.md).

The persistent text rectangle never includes effect padding or transformed
overflow. Padding is derived from visible effect reach; it is paint geometry,
not interaction geometry. Hit testing, resizing, and ordinary selection stay on
the logical box plus layout-owned ink overhang. Linear gradients, textures, and
Eraser points remain anchored to the logical origin even in bounded tiles.

Qt remains authoritative for shaping, cursor, selection, and IME. Selection and
the deferred caret paint after the completed effect surface, so Hollow, Filters,
and Eraser cannot alter editing feedback. Image layers are intentionally omitted
during native horizontal and vertical text editing; ordinary Filters remain
active. Editing visibility participates in cache identity so settled Image
pixels cannot leak into the editing surface or vice versa.

Interactive rendering may bypass an active missing optional raster, invalid
Filter, or bounded allocation failure with a warning and compatible fallback.
Strict export must fail rather than silently omit requested output. Empty raster
fields are valid neutral state. A nonempty missing reference stays persisted so
restoring the managed file can recover the effect.

## Persistence and compatibility

Live constructors and explicit writes are strict. Passive project, config, and
preset loading is permissive: warn, discard or default only the invalid portion,
and preserve valid siblings and unknown optional data. In particular:

- `text_effects` is authoritative when present, including an empty or malformed
  typed payload; do not revive legacy fields behind it.
- Legacy flat opacity and Stroke migrate to overall Opacity and one solid
  Outside Stroke. Legacy Shadow and Gradient settings are ignored.
- Older typed Stroke without `position` loads as Center to preserve its saved
  appearance. Older Shadow color and Gradient/Text Fill payloads use their
  existing targeted coercions; removed Gradient effect-level opacity is ignored.
- Invalid known blend modes fall back to Normal without dropping the effect.
- Unknown Filter IDs, schemas, and scalar params remain round-trippable without
  importing plug-in code. See [Text filters](text_filters.md).
- Valid-but-missing raster references remain saved. Malformed Image fields
  recover independently; a malformed Texture discards only that paint.
- Global formatting and reusable presets strip project-only Texture and Image
  values because they have no project asset registry. Concrete TextBlocks keep
  them.
- The removed singleton `TextBlock.rendered_image` is ignored, not revived as a
  compatibility owner.
- Serialization keeps the typed stack authoritative and writes only the
  compatibility views still required by older readers.

Run formatting preserves the same ownership. A Run Stroke override inserts or
updates the primary Stroke without replacing sibling effects or unrelated
Stroke parameters.

## Preview, commit, and selection

```text
control edit
  -> replace complete effective stack in preview state
  -> cancel: restore committed state, no command
  -> commit: one SetTextEffectStackCommand for all targets

Eraser stroke
  -> complete-mask preview owned by the canvas session
  -> release: one SetTextAlphaMaskCommand for one item
```

Preview never mutates committed `FontFormat`, project JSON, `QTextDocument`
history, or the paired editor. Structural changes settle or cancel incompatible
previews before indices change. Save, undo/redo, page replacement, selection
change, and teardown must resolve pending work at its owner.

The panel displays the most recently selected primary item's exact values. For
multi-selection it derives an occurrence map by effect identity and occurrence
number, intersecting the available count across all targets; it never builds a
merged stack or Mixed values. Matched cards receive the selection border and
preview/commit fan out through one command. Unmatched and Image cards edit only
the primary item. A newly added non-Image effect is inserted after the common
occurrences in every target so it is immediately matched. Reorder fans out only
when the relevant structural sequences align.

The Eraser remains single-item and stores immutable stroke history on
`TextBlock`. Activation inserts an empty enabled mask as its own undo step; a
completed Erase or Restore stroke is one later command. The canvas freezes the
source mapping for the stroke and discards incomplete input on Escape,
selection/page changes, or teardown.

## Image generation

Model, Context, and Prompt are card-local, non-undoable draft fields. They do
not rerender or invalidate the effect stack. Generate is available for exactly
one concrete item; the committed Image remains unchanged until a request
succeeds and its managed asset plus recipe are committed together.

Source and Inpainted use the exact finite in-bounds pre-transform logical crop.
Lettered draws only that item's untransformed, effect-free text over the
Inpainted crop. None sends no image context. Failure, Stop, stale selection, or
teardown discards the result and preserves the existing asset. Stop is
cooperative: an in-flight provider call may finish, but its output is ignored.
The request boundary owns generic image-edit transport so a future local backend
does not need to change card, crop, worker, or undo ownership.

## Caches and responsive preview

Caches are bounded derived state and never serialized:

| Cached work | Boundary |
| --- | --- |
| Completed surface or tiles | Separate committed, preview, and export namespaces keyed by stack, mask, layout, geometry, writing mode, scale, and editing visibility |
| Pre-mask and below-filter prefixes | Reuse mask-only and Filter-only previews without replaying unchanged lower work |
| Canonical glyph pixels and alpha | Reuse paint, Opacity, Blend, and generated-layer changes without rerasterizing text |
| Positioned Stroke coverage | Keyed by canonical source, width, and position; independent of paint and Opacity |
| Project raster decode | Shared by Texture and Image at the project asset boundary; positive entries only |

Requested-quality previews render on the next scene paint and may promote on
commit. The opt-in Faster Preview preference uses a non-promotable 0.5x scratch
surface; commit and export always use requested quality. Reshape temporarily
omits effects and rebuilds once geometry settles. Overall Opacity is native
group state and does not rebuild effect pixels.

Gradient acceleration is optional and byte-identical to the NumPy path. Numba
warms after main-window construction outside the Qt thread; painting must never
import or compile it synchronously. Keep antialiasing enabled and use the
existing full/tile policy instead of creating unbounded surfaces.

## Extending the stack

1. Add or extend one immutable typed value with a stable ID, strict live
   validation, tolerant passive loading, serialization, and an explicit neutral
   state.
2. Choose its canonical source, stack phase, blend boundary, external reach,
   Hollow behavior, and whether alpha may expand.
3. Extend the existing renderer, panel/card projection, and complete-stack
   preview/commit path. Keep visible strings translatable.
4. Audit full/tiled output, both writing modes, logical coordinates, padding,
   cache identity, reshape, strict export, nonlinear transforms, and neutral
   activation/deactivation.
5. Cover malformed optional data, migration when required, one-command undo,
   paired-editor isolation, and resource release.

Do not add a generic effect graph, new raster owner, or generated-layer cache
without a measured need. A blend mode that requires the page backdrop is a new
cross-layer and export contract, not another submenu leaf.

## Focused verification

Start with the affected ownership suites under both supported bindings when Qt
behavior is involved:

```bash
QT_API=pyqt6 QT_QPA_PLATFORM=offscreen \
  /opt/miniconda3/envs/common/bin/python -m pytest -q \
  tests/test_text_effect_domain.py tests/test_text_effect_persistence.py \
  tests/test_text_effect_preview.py tests/test_typed_text_effect_renderer.py

git diff --check
```

Add the Image, Filter, alpha-mask, transform, rich-text, and Run-pipeline suites
when their boundaries change. Rendering or interaction work still needs a
themed-app pass covering horizontal and vertical text, Hollow with
selection/caret, Eraser, preview/cancel/commit, undo/redo, nonlinear transforms,
zoom, and strict export.
