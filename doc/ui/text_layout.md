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

Character spacing and font features are applied per range. Identity spacing is
left unset when common ligatures should remain available because an explicit Qt
spacing property may suppress optional ligatures. Version-specific feature-tag
handling stays inside the layout/annotation boundary.

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

일본어 부호의 방향과 칸 안 배치는 [JLREQ §3.1 및 부록 A](https://www.w3.org/TR/jlreq/)
와 [Unicode 세로쓰기 방향](https://www.unicode.org/reports/tr50/)을 기준으로 삼는다.
현재 엔진은 Qt 가로쓰기 글리프의 회전·이동으로 처리하며, 폰트의 `vert`/`vrt2`
대체 글리프를 적용하는 완전한 세로쓰기 셰이핑 엔진은 아니다.

| 부호 분류 | 현재 세로쓰기 처리 |
| --- | --- |
| 괄호·인용 부호(cl-01/02) | 전각 괄호, 겹괄호, 기유메, `〝〞〟`를 시계 방향으로 회전한다. 낫표와 따옴표의 좌우 기준을 유지한다. |
| 하이픈류(cl-03), 장음 | `‐–〜゠` 및 `ー`를 회전한다. 진행량은 폰트의 실제 폭을 사용한다. |
| 느낌표·물음표(cl-04) | 표준 영문 모드에서는 바로 세워 중앙에 배치한다. 대체 모드의 기존 CLREQ 배치를 유지한다. |
| 중점류(cl-05) | `・`는 중앙에 둔다. 전각 `：；`는 표준 모드에서 회전하고, 대체 모드에서는 기존 배치를 유지한다. |
| 구점·독점(cl-06/07) | `、。`는 영문 모드와 무관하게 칸의 오른쪽 위에 배치한다. 가로쓰기용 `,.，．`는 자동 치환하지 않는다. |
| 대시·리더(cl-08) | `—―…‥`를 회전하고, 동일 부호의 연속 구간은 기존 분리 금지 처리를 유지한다. 이미 세로 방향인 `〳〴〵`는 회전하지 않는다. |

곡선 따옴표 `“”‘’`는 회전 폴백을 유지하며 `〝〟` 등으로 자동 치환하지 않는다.
압축한 여는 괄호의 선행 여백은 실제 글리프 여백보다 더 제거하지 않는다.
배치 결과는 효과·선택·커서·히트 테스트가 공유한다. 세로 중 가로쓰기 묶음 내부에는
이 회전 규칙을 적용하지 않는다. 금칙, 문맥별 약물 간격, 매달기 조판, 루비 읽기의
세로쓰기 셰이핑은 이 부호 배치 규칙만으로 구현되지 않는다.

Tate-chu-yoko is a horizontal Qt run occupying one vertical flow cell. Its
layout ignores authored letter spacing and uses the font's half-width
punctuation plus matching half-, third-, or quarter-width feature when
available. Standard Roman mode keeps that shaped run's natural horizontal
width; the alternate mode horizontally scales any remaining excess to one em.
The resulting visible ink is centered without changing the stored text. Glyph
ink may overhang the column, but that overhang affects only painting and
interaction bounds, never neighboring columns.

Ruby/furigana is attached layout content, not a detached overlay. Group Ruby is
indivisible; mono Ruby may wrap only between base/reading pairs. Each unit uses
the larger of its base and annotation advances, and the shorter run is spaced
within that cell. Horizontal Ruby appears above or below; vertical Ruby remains
upright on the right or left. The same cells own wrapping, paint, selection,
cursor, hit testing, effects, and visible bounds. Ruby and tate-chu-yoko cannot
overlap, and automatic Ruby overhang is not supported.

## Flow and spacing

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

Run both PyQt5 and PyQt6 when layout lifetime, shaping, cursor geometry, or
painting behavior changes.
