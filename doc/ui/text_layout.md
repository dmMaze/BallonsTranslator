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
  surrogate pair or grapheme cluster.
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

Common ligatures work under both bindings; discretionary/contextual ligatures
and oldstyle figures require Qt 6.11. Qt 5 preserves unsupported CSS without
applying it. Font-metric fallbacks and binding differences belong at the
layout/annotation boundary.

### Vertical

`VerticalTextDocumentLayout` normally creates one cell per grapheme and places
columns from right to left. Punctuation orientation and alignment are semantic
classes near the top of `vertical_layout.py`; extend those classes instead of
adding paint-time glyph exceptions.

Standard Roman mode keeps proportional Roman glyphs upright and centered. The
alternate mode rotates them clockwise and uses the Chinese mixed-layout
punctuation path. Compact punctuation shortens eligible punctuation cells
without clipping their ink. Repeated dashes, bars, leaders, and ellipses form
indivisible runs, with character spacing applied after the run.

Tate-chu-yoko (TCY) is an authored horizontal run occupying one vertical cell.
Only its selected range participates in shaping; surrounding whitespace keeps
vertical cells. Multi-character runs shape full-width forms as narrow equivalents
without changing document text or UTF-16 positions. Single-character runs and
unrelated compatibility characters stay unchanged. `TateChuYokoRun` owns index
mapping between the isolated run and source block. Native IME composition remains
authoritative until commit.

TCY ignores authored letter spacing. Standard Roman mode keeps its natural
shaped width; alternate mode shrinks excess width to one em. Visible ink is
centered, and overhang affects paint and interaction bounds without widening
neighboring columns.

Ruby/furigana is attached layout content, not a detached overlay. Group Ruby is
indivisible; mono Ruby may wrap only between base/reading pairs. Each unit uses
the larger of its base and annotation advances, and the shorter run is spaced
within that cell. Horizontal Ruby appears above or below; vertical Ruby remains
upright on the right or left. Ruby and TCY cannot overlap, and automatic Ruby
overhang is not supported.

## Flow and spacing

Koharu blocks (`TextBlock.det_model == 'koharu'`) use measured automatic fitting
when Auto layout is enabled, including Run -> Rendering. Pipeline Font Formatting
controls whether size is fitted or kept at the configured value. Other detectors
retain the image-based automatic layout during translation; render-only runs
preserve their existing wrapping. Explicit current frames take precedence;
new detections use their associated bubble, or the rotated source frame if none
was found. Automatic breaks remain derived Qt placements, preserving paragraphs,
rich-text ranges and the paired translation editor.

[`BubbleShape`](../../ballontranslator/utils/bubble_shape.py) stores a closed cubic
path as normalized `(anchor x/y, incoming handle x/y, outgoing handle x/y)` nodes
and an initial page frame. Tracing approximates the usable body, removing narrow
tails and spikes before preferring four-node rectangles or ellipses. Clipped and
compound bodies use short cubic chains with redundant anchors merged; bounded
fit error preserves significant concavities. Intricate contours retain simplified
corners when a short curve cannot represent them safely. Coverage of each block's
native text segmentation, clipped to its detected text box, is a separate adjustment
after fitting: keep a covering fit unchanged; otherwise tune its anchors and tangent
handles, splitting at most four additional anchors for difficult concavities. Full
retained foreground cells, including disconnected strokes, guide coverage; empty
bounding-box corners do not. Masks remain temporary detector data. The adjustment
never selects another fit.
Detection works on native mask crops, selects the component containing the source
text, and partitions shared bubbles at physical necks, with source bisectors as a fallback.
Coincident source anchors receive separate strips.
`TextBlock.bubble_layout` records whether that path is active for wrapping. Koharu
enables it when a bubble was detected; missing fields use ordinary defaults.
Resizing scales the local path without retracing. Automatic fitting honors this
switch, including user-created constraints on blocks from other detectors.

The anchor pen on the font panel's second row toggles the constraint. Enabled
blocks show the actual wrapping path instead of rectangular resize handles;
disabling retains the path, and enabling a block without one creates a rectangle.
[`bubble_control.py`](../../ballontranslator/ui/text_engine/bubble_control.py)
owns anchor selection and editing: drag anchors/handles, Shift to select multiple
anchors or constrain movement, Alt to pull/break handles, double-click the curve
to insert an anchor or an anchor to switch smooth/corner, and Delete to remove
selected anchors (at least three remain). A context menu provides the same point
operations. Arrows nudge selected anchors; Escape cancels the gesture.

Each drag previews only live layout geometry and commits one canvas undo command
with the shape and logical frame. Reframing preserves the untransformed contour
while updating its logical frame. Save, history, selection and tool transitions
cancel unfinished drags. The outline and raster share `bubble_path()`; guides and
handles are excluded from export. Text, font size and document history are not
modified by anchor editing.

[`bubble_layout.py`](../../ballontranslator/ui/text_engine/bubble_layout.py) owns a
bounded raster profile retained by each live layout, reused across font probes and
resizes and replaced when the path changes. Both writing modes take the widest
continuous contour spans and derive a shared alignment axis from them. Vertical
columns follow the saved Top/Center/Bottom setting; whitespace/caret cells move
with the glyphs. Insets reserve one ink band with a four-pixel floor. Balanced line
breaks reuse Qt advances and legal boundaries and preserve mandatory paragraphs;
Ruby, bidi text and unusually long documents retain their native wrapping.
Automatic sizing probes detached Qt items before applying the result; fixed sizes
are laid out directly. Fitting reports overflow and sizes below 9 page pixels.
The minimum is a readability floor, not a guarantee that every translation
fits. Manual auto-layout groups native document edits and frame/size changes in
one canvas command, preserving earlier typing history in both editors.
Save/export rebuild the same placements from the saved document.

For paragraph spacing, a caret formats its paragraph, a selection formats
intersected paragraphs, and Enter inherits the block format.

Whitespace remains document content and must consume explicit editable cells.
Horizontal and vertical layouts may represent those cells differently, but
neither may move whitespace into a second text model or drop it from cursor and
hit geometry. Vertical whitespace contributes flow advance, not the ink bounds
used to center the neighboring glyph in its column.

Character spacing is a trailing advance for the affected glyph or joined run.
On a squeezed single-column vertical item, increasing it may grow the logical
height to preserve that column; multi-column items keep normal fixed-area
reflow, and automatic growth never silently shrinks the box.

Line spacing is owned by the destination row or column. The first visual row or
column stays anchored without leading spacing; each later one uses its
paragraph's spacing value and mode. Paragraph boundaries do not restart this
visual leading-edge rule.

## Alignment and resize

The formatting panel shows Left/Center/Right in horizontal mode and
Top/Center/Bottom in vertical mode. `FontFormat.vertical_alignment` aligns each
vertical column along its height, including bubble-guided flow. It defaults to
Top when absent in older projects/styles. The two settings are independent and
survive writing-mode changes; Pipeline Font Formatting's alignment override
copies both from the global format.

The existing `FontFormat.alignment` also controls horizontal placement of vertical
columns and their width-growth anchor:

| Alignment | Fixed growth anchor | Added-width movement |
| --- | --- | --- |
| Left | Top-left | Columns stay fixed and grow rightward |
| Center | Top-center | Columns move by half and grow evenly |
| Right | Top-right | Columns move with the right edge and grow leftward |

Horizontal column placement changes every placement and ink-bound record together
without reshaping text or changing document content. A width-only resize may reuse that
translation when the settled content still fits; height, padding, or flow
changes require full layout. The geometry controller preserves the matching
scene-space anchor, so layout and scene movement must not both compensate for
the same resize.

## Painting and interaction

`vertical_line_placement()` is the shared boundary for rotated glyphs, TCY,
emphasis, Glyph Slant, effects, and editing geometry. Ligatures and joined glyphs
may change shaping, but not the logical UTF-16 editing range.

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

Test relationships rather than exact font-dependent pixels. Cover affected
writing modes, fonts, spacing, annotations, effects, UTF-16 editing, resize, and
mode switches. Start with `tests/test_horizontal_whitespace.py`,
`tests/test_vertical_*.py`, `tests/test_rich_text_annotations.py`, and
`tests/test_ruby_furigana.py`; add focused font/effect suites when those owners change.

Follow the shared [text-engine verification](text_engine.md#invalidation-and-verification)
for binding and visual checks.
