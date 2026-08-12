# Text layout

Read [Text engine](text_engine.md) first. This guide records the durable layout
contract: shaping, wrapping, vertical flow, placement records, alignment, and
their interaction with painting and editing. The code and focused tests remain
authoritative for individual glyph classes and algorithms.

## Mental model

```text
QTextDocument rich text (Qt UTF-16 positions)
  -> SceneTextLayout fragment metrics
  -> horizontal lines or vertical columns
  -> shared placement records
  -> fill, effects, annotations, cursor, selection, and hit testing
  -> TextItemGeometryController bounds and visual mapping
```

Qt remains the text model and shaper. The custom layouts place Qt
`QTextLine`s; they do not create a second editable text representation.

## State and ownership

| State | Contract |
| --- | --- |
| `FontFormat` writing mode, alignment, line spacing, and Roman mode | Item-wide persistent input |
| `QTextDocument` character formats | Range-bound font, character spacing, emphasis, and tate-chu-yoko input |
| `block_charfmt_lst`, `_map_charidx2frag` | Per-fragment metrics and Qt-position lookup rebuilt before layout |
| `x_offset_lst`, `y_offset_lst`, `line_spaces_lst` | Settled column, cell, whitespace, caret, and hit-test geometry |
| `per_char_records`, `text_combine_ranges` | Derived line width and tate-chu-yoko geometry |
| `_draw_offset` | Final visible-ink placement shared by fill and effects |
| `layout_generation`, annotation ink bounds | Cache invalidation and paint overflow |

All records below the first two rows are derived and must be rebuilt as one
settled layout. Never persist them or update only the record needed by one
painter.

`TextBlock.text_layout_version` versions item-wide layout semantics. A missing
or zero version migrates a vertical block to right alignment, matching the
placement used before vertical alignment existed. Inline HTML extensions are
versionless and follow the compatibility rules in [Text engine](text_engine.md).

## Writing modes

### Horizontal

`HorizontalTextDocumentLayout` keeps Qt shaping, cursor positions, glyph runs,
and `WrapAtWordBoundaryOrAnywhere`. Character spacing is passed to Qt as a
range font property. Qt's native wrap-boundary treatment of leading and
trailing spaces is therefore also the current horizontal behavior; changing
that policy requires an explicit feature, not reuse of the vertical workaround.

### Vertical orientation

`VerticalTextDocumentLayout` uses `WrapAnywhere`, normally creates one
`QTextLine` per vertical cell, then places columns from right to left. The
punctuation sets near the top of `layout.py` are semantic orientation and
alignment classes; extend those classes rather than adding draw-site glyph
exceptions.

`FontFormat.standard_vertical_roman_alignment` defaults to `True`. In that
mode proportional Roman glyphs remain upright and centered, while punctuation
uses the standard vertical classes. When disabled, proportional Roman glyphs
rotate clockwise and CLREQ-oriented punctuation uses the Chinese mixed-layout
path, including upper-right pause/stop placement. Horizontal layout is
unaffected.

Tate-chu-yoko is one ordinary horizontal Qt line occupying one vertical cell.
It keeps its natural horizontal advance centered on the column; excess width
is paint overflow and must not widen the column or move neighboring columns.
Whitespace inside the group remains part of the horizontal run. Emphasis adds
layout margins or ink overflow around the same established line placement.

## Vertical flow and spacing

Available content size is the logical box minus effect padding on both sides.
Effect padding is derived layout state and never document content.

Vertical character spacing is a trailing cell advance derived from the leading
fragment's semantic multiplier. Positive trailing advance does not by itself
force the final glyph into a new column. Qt may shape repeated dashes, ellipses,
and similar marks as one cluster; they retain one signed run-level advance and
keep their derived cell bounds monotonic when compressed.

The first column uses identity line spacing to anchor content at the logical
right edge. Every later column uses the configured line spacing, including a
terminal glyph that overflows and is settled during the same layout iteration.
Paragraph boundaries do not restart that first-column rule.

Vertical whitespace is deliberately explicit. `line_spaces_lst` records
leading/trailing space counts, their cell boundaries, and the Qt line position
so spaces consume height and remain hittable. Removing this machinery as
redundant will change wrapping, caret movement, and hit testing.

When character spacing changes on a squeezed, single-column item,
`spacing_change_height_growth()` may grow the logical height before applying
the range format. This preserves the existing column instead of turning a
spacing edit into a horizontal reflow. Multi-column items retain normal
fixed-area reflow. The automatic path grows but does not silently shrink a box.

Treat column settlement as one transaction. First-column anchoring, configured
column advances, terminal overflow, whitespace cells, UTF-16 positions,
fragment metrics, and draw/hit records are coupled; changing one without the
others will break otherwise unrelated punctuation, annotations, effects, or
editing geometry.

## Alignment and resize

Vertical alignment is a post-layout horizontal translation of all settled
columns:

| Alignment | Fixed growth anchor | Added-width movement |
| --- | --- | --- |
| Left | Top-left | Columns stay fixed and grow rightward |
| Center | Top-center | Columns move by half and grow evenly |
| Right | Top-right | Columns move with the right edge and grow leftward |

The translation must move every `QTextLine`, `x_offset_lst`, `layout_left`, and
annotation ink bound together. It changes `layout_generation`, but not text
flow, logical size, or document content.

`reLayoutForResize()` may use this translation-only path when width is the only
changed input and the content still fits. Height, effect padding, or an
insufficient width requires full reflow. The geometry controller preserves the
matching scene-space anchor during resize; layout and scene movement must not
both compensate for the same delta.

## Indices, placement, and rendering

Qt text positions and fragment lengths are UTF-16 code units. Use the shared
UTF-16 and grapheme helpers whenever Python strings meet `QTextLine` positions,
especially for emoji, tate-chu-yoko, cursor placement, and hit testing. Never
return a caret inside a surrogate pair or combined run.

`vertical_line_placement()` is the common placement boundary for rotated
glyphs, tate-chu-yoko, emphasis, Glyph Slant, and effect rendering. Cursor,
selection, and `hitTest()` must consume the same cells and transforms. Visible
tate-chu-yoko and annotation overhang belongs in source-paint and hit geometry,
not in persistent logical bounds.

The background effect document reuses the foreground layout's settled draw
offsets. `updateDrawOffsets()` deliberately preserves them during the stroke
pass; recomputing offsets independently will move stroke and shadow away from
the fill. Likewise, Glyph Slant and transformed-surface caches are namespaced by
`layout_generation` and must not retain records from an earlier layout.

Actual-ink measurement caches must describe glyph-run identities and positions,
not only text, font, and total line size. Fragmented spacing can produce equal
line widths with different glyph placement. Avoid layout-time pixel probes when
font metrics or tight bounds suffice: probing can warm Qt's process-global
glyph raster at the wrong scale and degrade later effect rendering.

Replacing a document layout is also a QObject lifetime boundary.
`QTextDocument.setDocumentLayout()` may delete the previous native layout
synchronously, so detach any glyph/layout renderer before the call and attach a
new renderer afterward. No geometry query may retain the deleted layout.

## Invalidation order

The normal content/format path is:

```text
documentChanged
  -> reLayoutEverything: rebuild fragment metrics and position maps
  -> reLayout: settle lines and columns
  -> updateDrawOffsets and annotation ink bounds
  -> documentSizeChanged / guarded size enlargement
  -> geometry and effect refresh
```

Alignment-only and valid width-only resize use the smaller translation path.
Publish size changes only after placement records are complete; effects,
geometry, and cursor painting may run synchronously from Qt signals.

## Verification

Prefer relationship assertions over font-dependent screenshots. Cover the
affected combinations of:

- horizontal and vertical writing, both Roman modes, and all alignments;
- first, middle, terminal, exact-fit, overflow, newline, and whitespace cases;
- non-identity line spacing and fragmented character spacing;
- punctuation, joined marks, tate-chu-yoko, emphasis, effects, and Glyph Slant;
- UTF-16 text, selection, cursor, insertion, hit testing, resize, and mode
  switching.

Focused coverage lives in `tests/test_vertical_alignment.py`,
`tests/test_vertical_roman_alignment.py`, and
`tests/test_rich_text_annotations.py`. Run both PyQt5 and PyQt6 when layout
lifetime, glyph runs, cursor geometry, or painting behavior changes.
