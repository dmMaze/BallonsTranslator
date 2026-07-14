# Independent reviewer report

Implementation/test artifact key: `f101479edbaf4b35606fe7503ae752846055d71d`

## Inputs

- Repository: `F:\ballon\BallonsTranslator`
- Branch: `codex/pr1238-clean-redesign-20260714`
- Exact upstream: `6155f9b303033b24f57a2c025d2edbfed3eb847f`
- Rejected PR head: `57e9f1c604fc9ccbc79dc9fbf7ad91d77592cf04`
- Rejected worklog head: `47b47ca37e30ee2a94ab1926e9c454e89a0ecf96`

The reviewer used a fresh context, inspected history/diffs/runtime behavior and
artifacts directly, and independently reran the focused binding matrix and both
full-suite comparison trees.

## Final verdict

| Severity | File/location | Issue | Evidence | Required fix |
|---|---|---|---|---|
| None | Corrected implementation and artifact bundle | No remaining finding | Code, history, tests, raw logs, screenshots, JSON/XML, provenance, and stable inventory passed independent checks | None |

```text
blocking = 0
high = 0
medium = 0
low = 0
approval = APPROVED
```

Human-equivalent GUI interaction remains an explicitly disclosed limitation,
not a claimed pass. The first Computer Use observation failed, so the manual
checklist records every mouse/keyboard scenario as `Not performed` and labels
the offscreen Qt evidence as an automated substitute.

## Review and correction loop

1. The first artifact audit found that the evidence used in-process reloads,
   left an evidence-only dashed polygon in the export scene, and framed cropped
   extreme cases too weakly. It also requested stronger environment and Git
   attestation.
2. The generator was corrected to save a portable project with a real page,
   load it from disk in two distinct fresh Python processes, remove the dashed
   polygon before `Canvas.render_result_img()`, contain all extreme bounds in a
   padded viewport, materialize migration fixtures from exact Git blobs, and
   emit environment/provenance/hash records. The reviewer reopened every image
   and revalidated the JSON/XML.
3. A second audit requested raw evidence for static checks, package-aware
   doctests, and the exact-upstream feature-test overlay, plus a refreshed
   porcelain snapshot.
4. Four raw logs were added with exact cwd, environment, argv, expected/actual
   exit code, stdout, and stderr. The generator asserts their key totals and was
   rerun after final artifact cleanup.
5. The final narrow re-audit compared the recorded line counts and results to
   the current bytes, matched the 37 provenance entries to the 36 artifacts plus
   architecture document, and reduced all severity counts to zero.

## Independent tests

| Run | Result |
|---|---:|
| Focused PyQt5 | 100 passed, 186 subtests passed |
| Focused PyQt6 | 100 passed, 186 subtests passed |
| Focused PySide6 | 100 passed, 186 subtests passed |
| Feature full PyQt6 | 7 failed, 259 passed, 1 skipped, 193 subtests passed |
| Exact-upstream full PyQt6 | 7 failed, 159 passed, 1 skipped, 7 subtests passed |

The two full runs have the same seven pre-existing failure identities. The
reviewer also verified `py_compile` for all 11 changed production modules,
`git diff --check`, zero forbidden identifiers, one production tangent formula,
doctests `fontformat` 4/4 and `text_transform` 5/5, overlay collection with nine
expected errors, and the collectable overlay pair with 20 expected failures and
one pass.

## Provenance audit

- The first feature commit has the exact upstream SHA as its sole parent; the
  implementation is a linear 13-commit chain with no merge commit.
- Neither rejected head is an ancestor or descendant of the feature head.
- Stable patch IDs: feature 13, rejected PR 3, worklog 38; both intersections
  are zero. `git cherry` marks all feature commits unique, and targeted
  `range-diff` maps none to either rejected range.
- Reflog contains branch creation from the exact upstream followed only by the
  13 ordinary commits; no merge, rebase, reset, or cherry-pick entry appears.
- No feature-changed blob is identical to a worklog-changed blob. Normalized
  added-line windows of 5, 10, and 20 lines have zero matches; the longest
  common run is four generic `QImage`/`QPainter` setup lines.
- No unrelated production, dependency, build, or generated-binary change was
  found. Tracked production and tests match the implementation SHA exactly.

## Artifact integrity

- All five evidence PNGs and the fixture page were opened at original detail.
- The clean export has no selection or evidence polygon; all extreme-effect
  bounds are inside the recorded padded source rectangle.
- Two distinct process IDs loaded the real portable project from disk and
  preserved `(1.234568, 0.625, -14.5)` exactly.
- Every JSON/XML parses. Screenshot/JUnit hashes match, all eight migration
  files match the exact feature Git blobs, and the feature/upstream JUnit
  failure identity sets are equal.
- The final raw-log hashes independently recorded by the reviewer are:
  `4b4780345508ce02a21fe56ce22bcf806e46002ee366a1d9e56213640ee4f828`,
  `bd53b0be540e055b1dd50db1c4a1aa882367874532f1cb5cb60505a5cf9b1431`,
  `7b5b7db73f072654d5f236888a46f04a14a32fe045c7e256a5016dd8523a5fda`,
  and `5cc5294e34f56684d01f6b03bb258f325ac96937b21648238e9b7813aada53e7`.
- No `__pycache__`, `.pyc`, `.pyo`, or `.tmp` file is present in the bundle.

The provenance JSON is intentionally a generation-time snapshot. This report
is written after approval and therefore cannot attest to itself without making
the evidence cycle self-referential; the final Git commit and post-commit status
provide the enclosing integrity boundary.
