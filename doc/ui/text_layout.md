# Text layout

Read [Text engine](text_engine.md) first. This guide records the behavior and
ownership shared by shaping, wrapping, vertical flow, painting, and editing.
Implementation-specific algorithms belong in code comments and focused tests.

## Mental model

```text
QTextDocument rich text (Qt UTF-16 positions)
  -> SceneTextLayout fragment metrics
  -> horizontal lines or vertical columns
  -> settled placement
  -> fill, effects, annotations, cursor, selection, and hit testing
  -> TextItemGeometryController bounds and visual mapping
```

Qt remains the editable text model and shaper. The custom layouts place Qt
`QTextLine`s; they do not create a second text representation.

## Core contract

- `FontFormat` supplies item-wide writing mode, alignment, and compatibility
  defaults. `QTextDocument` formats own range-bound typography and
  paragraph-bound line spacing.
- Placement records, ink bounds, and caches are derived. Rebuild them together
  for one settled layout generation and never persist them.
- Fill, effects, annotations, cursor, selection, hit testing, and visual bounds
  must consume the same settled cells and transforms.
- Qt positions are UTF-16 code units. Use the shared UTF-16 and grapheme helpers
  wherever Python strings meet Qt positions; never expose a caret inside a
  surrogate pair or combined run.
- Effect padding and visible ink overflow belong to source geometry, not the
  persistent logical rectangle.

`TextBlock.text_layout_version` versions item-wide layout semantics. Missing or
version-zero vertical blocks migrate to right alignment, matching their earlier
effective placement. Inline HTML extensions remain versionless and follow the
compatibility rules in [Text engine](text_engine.md).

## Writing modes

### Horizontal

`HorizontalTextDocumentLayout` keeps Qt shaping, glyph runs, cursor behavior,
and word-boundary wrapping. It adds only the geometry Qt does not expose in the
form the editor needs. In particular, overflowing trailing U+0020 spaces stay
in the document but receive derived continuation-row cells so wrapping, box
growth, cursor, selection, and hit testing agree. Other Unicode separators keep
Qt behavior.

Common ligatures work under both bindings. Discretionary/contextual ligatures
and oldstyle figures require Qt 6.11; Qt 5 preserves their CSS without applying
them. Character spacing and font features are applied per range. Identity spacing is
left unset when common ligatures should remain available because an explicit Qt
spacing property may suppress optional ligatures. Version-specific feature-tag
handling stays inside the layout/annotation boundary.

Representative ink bounds define row and cell metrics when available from the
selected font. Missing or empty reference ink falls back to that font's line
metrics without moving valid text above the logical box. Vertical glyph
placement retains its shaped ink as the authoritative geometry.

### Vertical

`VerticalTextDocumentLayout` normally creates one cell per grapheme and places
columns from right to left. Punctuation orientation and alignment are semantic
classes near the top of `vertical_layout.py`; extend those classes instead of
adding paint-time glyph exceptions.

Hangul placement preserves font side bearings by centering the advance rather
than each syllable's ink bounds. Within each block, Hangul in a column
shares a top correction without changing flow advances. Any remaining ink
overflow contributes to painting and interaction bounds.

Standard Roman mode keeps proportional Roman glyphs upright and centered. The
alternate mode rotates them clockwise and uses the Chinese mixed-layout
punctuation path. Compact punctuation shortens eligible punctuation cells
without clipping their ink. Repeated dashes, bars, leaders, and ellipses form
indivisible runs, with character spacing applied after the run.

Japanese punctuation classes follow [JLREQ](https://www.w3.org/TR/jlreq/)
and [Unicode vertical orientation](https://www.unicode.org/reports/tr50/).
Fullwidth stops `、。，．` anchor to the upper-right of the base character
frame independently of Roman orientation. ASCII punctuation retains its
mode-specific behavior.
Standard mode rotates fullwidth colons and semicolons; alternate mode retains
the existing CLREQ placement. Compact opening brackets must not lose more
leading space than their actual font bearing permits.
Character membership belongs in the semantic sets in
`vertical_layout.py` and their focused tests.

The engine rotates and translates Qt horizontal glyphs; it does not implement
full vertical shaping with `vert`/`vrt2` substitutions. Rotated curly quotes
preserve authored input as a fallback, not a recommendation for Japanese
vertical quotation style. No characters are automatically replaced, and these
rotation rules do not apply inside tate-chu-yoko. Line-start/end prohibitions,
contextual punctuation spacing, hanging punctuation, and vertical shaping of
Ruby readings require separate support.

Tate-chu-yoko is a horizontal Qt run occupying one vertical flow cell. Multi-character
runs shape explicit full-width forms as their narrow equivalents, preserving the
original document text and UTF-16 positions. Single-character runs and unrelated
compatibility characters stay unchanged. Normalized runs use the same placement
for paint, effects, selection, and hit testing; their temporary shaping layouts
contain only the run's text. `TateChuYokoRun` translates run-local glyph and cursor
indices to block-local UTF-16 offsets at the Qt boundary. These runs are rebuilt
with the owning vertical layout. Native IME composition remains authoritative
until commit.

TCY ignores authored letter spacing and uses the font's half-width punctuation
plus matching half-, third-, or quarter-width feature when available.
Standard Roman mode keeps that shaped run's natural horizontal
width; the alternate mode horizontally scales any remaining excess to one em.
The resulting visible ink is centered without changing the stored text. Glyph
ink may overhang the column, but that overhang affects only painting and
interaction bounds, never neighboring columns.

Qt may include trailing spaces beyond the authored tate-chu-yoko range in the
same line. Those spaces remain separate vertical cells so flow advance, caret,
selection, and hit testing agree at the group boundary. Spaces authored inside
the group remain horizontal.

Ruby/furigana is attached layout content, not a detached overlay. Group Ruby is
indivisible; mono Ruby may wrap only between base/reading pairs. Each unit uses
the larger of its base and annotation advances, and the shorter run is spaced
within that cell. Horizontal Ruby appears above or below; vertical Ruby remains
upright on the right or left. The same cells own wrapping, paint, selection,
cursor, hit testing, effects, and visible bounds. Ruby and tate-chu-yoko cannot
overlap, and automatic Ruby overhang is not supported.

## Flow and spacing

Character spacing is range-bound; line spacing is paragraph-bound. A caret
formats its paragraph, a selection formats intersected paragraphs, and Enter
inherits the block format. `FontFormat` supplies defaults for old or empty rich text.

Whitespace remains document content and must consume explicit editable cells.
Horizontal and vertical layouts may represent those cells differently, but
neither may move whitespace into a second text model or drop it from cursor and
hit geometry. Vertical whitespace contributes flow advance, not the ink bounds
used to center the neighboring glyph in its column.

Character spacing is a trailing advance for the affected glyph or joined run;
W3C tate-chu-yoko composition ignores it. On a squeezed single-column vertical
item, increasing it may grow the logical height to preserve that column;
multi-column items keep normal fixed-area reflow, and automatic growth never
silently shrinks the box.

Line spacing is owned by the destination row or column. The first visual row or
column stays anchored without leading spacing; each later one uses its
paragraph's spacing value and mode. Paragraph boundaries do not restart this
visual leading-edge rule.

Settle a layout as one transaction. Wrapping, whitespace, annotations,
fragment metrics, UTF-16 positions, ink bounds, and interaction geometry are
coupled and must be published only when complete.

## Alignment and resize

Vertical alignment translates settled columns horizontally:

| Alignment | Fixed growth anchor | Added-width movement |
| --- | --- | --- |
| Left | Top-left | Columns stay fixed and grow rightward |
| Center | Top-center | Columns move by half and grow evenly |
| Right | Top-right | Columns move with the right edge and grow leftward |

Alignment changes every placement and ink-bound record together but does not
reshape text or change document content. A width-only resize may reuse that
translation when the settled content still fits; height, padding, or flow
changes require full layout. The geometry controller preserves the matching
scene-space anchor, so layout and scene movement must not both compensate for
the same resize.

## Painting and interaction

`vertical_line_placement()` is the shared boundary for rotated glyphs,
tate-chu-yoko, emphasis, Glyph Slant, and effects. Cursor, selection, and hit
testing must use the same placement. Ligatures and joined glyphs may change
shaping, but they do not change the logical UTF-16 editing range.

Document backgrounds paint below selection, and glyph ink paints above it.
Foreground and effect layouts must reuse the same settled offsets. Caches tied
to placement must be invalidated with the layout generation and must not retain
records from a replaced document layout.

## Invalidation and verification

The normal path is:

```text
document or format change
  -> rebuild fragment metrics and position maps
  -> settle lines or columns
  -> update draw offsets and ink bounds
  -> publish size and refresh geometry/effects
```

Test relationships rather than exact font-dependent pixels. Cover the affected
writing modes, alignments, spacing, annotations, effects, UTF-16 text, editing,
resize, and mode switches. Focused coverage lives in:

- `tests/test_horizontal_whitespace.py`
- `tests/test_vertical_alignment.py`
- `tests/test_vertical_interaction.py`
- `tests/test_vertical_roman_alignment.py`
- `tests/test_rich_text_annotations.py`
- `tests/test_ruby_furigana.py`

Follow the shared [text-engine verification](text_engine.md#invalidation-and-verification)
for binding and visual checks.
