# PR #1238 clean redesign

This document separates observed facts from the design decisions and invariants
of the clean implementation. It is intentionally based on `upstream/dev`, not
on code from the rejected feature or its later worklog.

## Observed facts

- Repository: `dmMaze/BallonsTranslator`
- Current `upstream/dev`: `6155f9b303033b24f57a2c025d2edbfed3eb847f`
- PR #1238 base: `6bff00ee017706eb54637dce828cb0149632ecca`
- PR #1238 head: `57e9f1c604fc9ccbc79dc9fbf7ad91d77592cf04`
- Worklog head: `47b47ca37e30ee2a94ab1926e9c454e89a0ecf96`
- The rejected PR has three feature commits. The later worklog is 39 commits
  ahead of the current upstream snapshot and changes 23 files by roughly
  13,800 insertions and 950 deletions.
- Neither rejected head is an ancestor of this feature branch. The merge base
  of this branch and `upstream/dev` is the exact upstream SHA above.
- The original approach mixed visual scale with document point size and font
  stretch. Later repairs added multiple compensating coordinate/state layers,
  per-line transforms, cloned render documents/layouts, and opaque future-state
  handling. This made wrapping, vertical layout, effects, editing, persistence,
  and undo disagree about which geometry was logical.

The rejected branches were inspected only for requirements, observable legacy
field names, migration signatures, and regression cases. No feature commit,
patch, file, helper, state object, or cache structure was applied or copied.

## Design decisions

### Canonical state

`TextBlock.fontformat` is the only persistent owner of:

```text
horizontal_scale = 1.0
vertical_scale   = 1.0
slant_angle      = 0.0
```

The normalized ranges are `[0.1, 4.0]`, `[0.1, 4.0]`, and
`[-45.0, 45.0]`. Values use six-decimal canonical precision and normalize
negative zero to zero. The Advanced Text Format boundary alone converts scale
factors to and from percentages.

The rich-text document remains the owner of logical text and native character
formatting, including native italic shaping. No transform factor changes point
size, character stretch, HTML, wrapping, or line breaks.

### Geometry

There is one post-layout base transform on `TextBlkItem`. Its pivot is the
center of the unpadded logical block rectangle. For a local point `(x, y)` and
pivot `(px, py)`:

```text
k  = -tan(radians(slant_angle))
x' = px + horizontal_scale * (x - px) + k * vertical_scale * (y - py)
y' = py + vertical_scale * (y - py)
```

The single `QTransform` is applied with `combine=False`. Existing block
rotation remains `QGraphicsItem` rotation about the same pivot and is therefore
applied after the base transform. Horizontal and vertical layouts use the same
matrix; changing writing direction never swaps scale axes.

Logical persistence/layout callers continue to use the untransformed rectangle.
Visual callers use the four mapped corners. The shape control maps those exact
corners from scene space into its parent, so shear is never collapsed to an
axis-aligned substitute. Resize maps the pointer through `item.mapFromScene()`
and compensates item position to keep the opposite scene anchor fixed.

### Rendering and editing

All passes use the live document and its attached layout. The intended order is
shadow, stroke, normal fill/gradient, then Qt's selection/cursor/IME overlay.
Effects are measured in logical object space and inherit the item transform.
Effect padding is recomputed from zero so removing an effect can shrink it.

Mouse hit testing remains local-layout hit testing. Qt item/scene mapping is the
only inverse boundary for transformed cursor and IME geometry. A transform-only
preview or commit does not mutate the document, cursor position/anchor, logical
rectangle, item position, or layout. Direction changes replace only the layout
and restore the exact cursor position and anchor without writing corrective
spacing into rich text.

### UI and undo

Each transform control has three states: idle, pending text, and drag preview.
Typing changes no model. Enter, focus loss, and selection transition commit once;
Escape reverts. Drag move changes only transient item transforms, and release
creates one command. Mixed drag applies the same display-unit delta to each
item's own starting value; mixed numeric input applies one absolute value.

One undo command owns all selected items plus per-item before/after canonical
tuples. A normalized no-op creates no command and triggers no item update. The
command stores no HTML, pixmap, scene snapshot, panel state, or derived matrix.

### Persistence

The canonical project root schema is version 1. Each block writes the three
canonical fields inside `fontformat`. Loading is transactional:

```text
preflight raw versions
-> migrate a deep copy
-> construct and validate an isolated candidate project
-> replace live project state
```

| Input | Result |
| --- | --- |
| Upstream project with no transform fields | Neutral defaults; HTML unchanged |
| Canonical schema 1 | Validate/normalize; HTML unchanged |
| Known failed logical representation | Canonicalize known aliases; HTML unchanged |
| Neutral unversioned failed representation | Canonicalize; HTML unchanged |
| Exact reversible effective representation | Restore logical font geometry; record warning |
| Ambiguous effective representation | Reject the whole project |
| Future root or block representation | Reject before live-state mutation |
| Nonnumeric or non-finite value | Reject |
| Finite out-of-range value | Clamp and record a migration warning |

Canonical output removes legacy aliases and markers. A second load/save performs
no migration. Duplication continues to use the existing deep-copy contract.

## Required invariants

- The feature branch starts at the exact upstream SHA recorded above.
- Transform state has one persistent owner and one item-local affine matrix.
- Visual transforms never enter logical HTML, font geometry, wrapping, masks,
  or persisted block rectangles.
- Preview never mutates canonical state; commit never derives state from a
  matrix.
- Direction, edit mode, canvas rendering, and scene export share geometry and
  rendering paths.
- Stroke, shadow, and gradient use the current document/layout and cannot create
  alternate transform formulas.
- Transform-only undo is atomic across the selection and preserves HTML,
  document revision, logical geometry, and cursor selection.
- Unsupported or ambiguous payloads cannot partially replace an open project.
- No-op operations leave undo count, repaint notifications, document/layout
  generations, and item matrices unchanged.

Verification commands, baseline results, manual artifacts, and the independent
review report are stored with the implementation artifacts rather than asserted
as facts here before they have run.
