# PR #1238 manual validation record

Implementation/test artifact key: `f101479edbaf4b35606fe7503ae752846055d71d`

## GUI availability

The Windows Computer Use workflow found the running `BalloonsTranslator`
window, but its first observation failed with the exact error:

```text
SetIsBorderRequired failed: 해당 인터페이스를 지원하지 않습니다. (0x80004002)
```

Per the Computer Use safety/recovery guidance, no input was sent after the
failed observation and no unrelated Windows automation was used. Therefore no
human-equivalent mouse/keyboard item below is claimed as manually passed.

The five required screenshots were generated through the same live Qt widget,
`TextBlkItem`, `QGraphicsScene`, and actual `Canvas.render_result_img()` paths
with `QT_QPA_PLATFORM=offscreen`, then opened and visually inspected. This is
recorded as an **automated substitute**, not as manual GUI execution.

## Checklist

| Scenario | Manual GUI | Automated substitute and evidence |
|---|---|---|
| Three Advanced Text Format controls | Not performed | Pass: `controls.png`; exact 125.0%, 80.0%, 17.0° values visible; UI suite |
| Drag preview / release | Not performed | Pass: panel integration tests cover cumulative preview and one-command release |
| Numeric Enter / focus-out | Not performed | Pass: committed-control and panel integration tests |
| Pending selection change | Not performed | Pass: pending value commits to the old selection before target switch |
| Invalid input / Escape | Not performed | Pass: invalid input restoration and preview cancellation tests |
| CJK IME | Not performed | Pass: real `QInputMethodEvent` preedit/commit regression |
| Cursor | Not performed | Pass: exact scene/local round trip returns cursor position 4 |
| Partial selection | Not performed | Pass: selection paint and whole-format cursor/anchor restoration tests |
| Horizontal / vertical writing | Not performed | Pass: `horizontal-vertical.png`; common matrix and 16-direction tests |
| Multi-block | Not performed | Pass: multiline/trailing-empty paragraph and vertical layout tests |
| Undo / redo / new edit | Not performed | Pass: whole-format/text-edit/effects chain and redo-branch replacement tests |
| Save / restart / reload | Not performed | Pass: `save-reload.png`; `ProjImgTrans.save()` writes a portable `fixture-project/project.json`, then two distinct fresh Python processes independently read it from disk against a real `page.png` |
| Repeated load | Not performed | Pass: both independent process attestations preserve the canonical transform exactly; no repeated-save sequence is claimed |
| Final export | Not performed | Pass: right pane of `editing-canvas-export.png` is actual `Canvas.render_result_img()` after the evidence-only dashed item is removed from the scene |
| Mixed CJK / Latin | Not performed | Pass: H/V, editing/export, save/reload evidence and cross-binding render tests |
| Native italic + slant | Not performed | Pass: native font italic remains document formatting; slant stays item-local |
| Stroke / shadow / gradient | Not performed | Pass: all five PNGs plus live-layout effect/cache regressions |
| H/V extremes | Not performed | Pass: `extreme-effects.png`; canonical 0.1..4.0 clamp tests |
| ±45° slant | Not performed | Pass: `extreme-effects.png`; outward-clamp strict no-op tests |
| Rotation | Not performed | Pass: extreme/effect evidence and exact polygon/rotation tests |
| Clipping | Not performed | Pass: transparent cache border and rasterized transformed-polygon containment tests |
| Transformed shape control | Not performed | Pass: blue quadrilateral evidence; exact corners/eight handles/anchor-fixed resize tests |
| Copy / paste / duplicate | Not performed | Pass: actual manager copy/paste methods and `PasteBlkItemsCommand` undo/redo test |

## Screenshot inspection

- `controls.png`: readable Advanced panel, all three controls, shadow and gradient groups.
- `horizontal-vertical.png`: horizontal and vertical mixed-script items with exact dashed bounds.
- `editing-canvas-export.png`: selection and dashed evidence overlay appear only on the editing side; the actual export retains effects without evidence markup.
- `extreme-effects.png`: all four labels and transformed bounds are inside the padded source viewport; clamp extremes, rotation, shear, stroke, shadow and gradient are visible.
- `save-reload.png`: before/two-independent-reload geometry matches; both captions show `(1.234568, 0.625, -14.5)`.

Machine-readable assertions, fresh-process PIDs, JUnit hashes, environment data,
and Git provenance are in `test-results/generation-report.json` and the adjacent
JSON/text logs. Migration fixtures are materialized from the exact feature Git
objects and verified in `migration-fixtures/manifest.json`.

No unchecked manual action is represented as passed.
