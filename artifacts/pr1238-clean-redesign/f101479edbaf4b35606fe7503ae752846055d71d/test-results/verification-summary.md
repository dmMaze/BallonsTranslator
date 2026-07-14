# Verification summary

## Environment

```text
Python 3.10.6 (tags/v3.10.6:9c7b4bd, Aug 1 2022, 21:53:49) [MSC v.1932 64 bit (AMD64)]
PyQt5 5.15.11 / Qt 5.15.2
PyQt6 6.6.1 / Qt 6.6.1
PySide6 6.8.2.1 / Qt 6.8.2
QT_QPA_PLATFORM=offscreen
Python executable=F:\python\python.exe
dependency overlay=F:\ballon\.pr1238-test-deps
```

Every binding run used a fresh Python process. The bundle now includes
`python-vv.txt`, `pip-freeze.txt`, and `evidence-environment.json` from the same
runtime/overlay used to regenerate the evidence. These are a regeneration-time
environment snapshot; they are not represented as files captured at the earlier
JUnit start time. The JUnit XML retains each actual run timestamp and hostname.

## Focused matrix

Command selected every `tests/test_text_transform_*.py` plus
`tests/test_vertical_text_stroke.py`.

| Binding | Result | JUnit |
|---|---:|---|
| PyQt5 | 100 passed, 186 subtests passed, 1 existing warning | `focused-pyqt5.xml` |
| PyQt6 | 100 passed, 186 subtests passed, 1 existing warning | `focused-pyqt6.xml` |
| PySide6 | 100 passed, 186 subtests passed, 1 existing warning | `focused-pyside6.xml` |

The warning is the unchanged NumPy `np.bool8` deprecation in
`ballontranslator/utils/io_utils.py:22`.

## Full-suite exact-upstream comparison

Both runs used the same executable, dependency overlay, environment, and
`python -m pytest -q` command.

| Tree | SHA | Failed | Passed | Skipped | Subtests passed | Classification |
|---|---|---:|---:|---:|---:|---|
| Exact upstream worktree | `6155f9b303033b24f57a2c025d2edbfed3eb847f` | 7 | 159 | 1 | 7 | Baseline |
| Feature | `f101479edbaf4b35606fe7503ae752846055d71d` | 7 | 259 | 1 | 193 | No new failures; +100 passed, +186 subtests |

The same seven failures occur in both trees:

- two lazy-metadata/data-directory expectations;
- four Windows `NamedTemporaryFile` reopen `PermissionError` failures;
- one pre-existing Gemini builtin `vision_model` mismatch subtest.

JUnit files: `full-upstream-pyqt6.xml`, `full-feature-pyqt6.xml`.

`generation-report.json` parses and hashes the JUnit files. It asserts both the
seven-failure totals and the exact failed `classname::name` set are identical;
it does not claim byte-identical failure messages.

## Feature-test overlay on upstream

The latest feature tests and fixtures were copied to
`F:\ballon\pr1238-baseline-overlay-6155f9b` and imported only against the exact
upstream source tree.

- 9 files stop at collection because the upstream intentionally lacks the new
  transform module, command, migration exceptions, gradient marker, UI control,
  UTF-16/grapheme helper, or related API.
- The two collectable UI/shape files produce 20 expected feature-absence
  failures and 1 pass.
- These are expected red results demonstrating that the new regressions are not
  vacuous against upstream.

Raw commands, exact working directories/environment overrides, exit codes, and
stdout/stderr are preserved in `upstream-overlay-collection.log` (expected exit
2, 9 collection errors) and `upstream-overlay-collectable.log` (expected exit 1,
20 failed and 1 passed for `test_text_transform_panel_integration.py` plus
`test_text_transform_shape_control.py`).

## Static and doctest

- changed production Python files: `py_compile` passed;
- `git diff --check`: passed;
- forbidden reviewer identifiers: 0;
- actual `math.tan`: exactly one occurrence in `ui/text_transform.py`;
- percentage conversion: only the Advanced control boundary helpers;
- `QFont.setStretch`, legacy marker and `italic_angle`: migration oracle only;
- cloned `QTextDocument`, `drawContents`, or `toPixmap` in effect paint path: 0;
- package-aware doctest: `fontformat` 4/4 and `text_transform` 5/5 passed.

The complete static commands and grep output are in `static-checks.log`.
Verbose package-aware doctest output is in `package-aware-doctest.log`. Each log
records its exact cwd, environment overrides, argv, exit code, stdout, and
stderr. Expected `rg` zero matches are explicitly recorded with exit code 1;
they are not presented as command failures.

The literal file-mode doctest command cannot import `fontformat.py`'s relative
package imports. The package-aware equivalent was therefore used and passed.

## Evidence generation and provenance

- `git-provenance.json` records the branch, exact `HEAD`, porcelain status, and
  independent clean comparisons of tracked `ballontranslator/` and `tests/`
  paths to feature SHA `f101479edbaf4b35606fe7503ae752846055d71d`.
- `fixture-project/project.json` was written by `ProjImgTrans.save()` with a
  portable `directory: "."` and a real sibling `page.png`. Two simultaneously
  live, distinct Python child processes independently loaded that file from
  disk; their PID, project/page hash, transform, environment and Git attestation
  are recorded in `reload-process-1.json` and `reload-process-2.json`.
- This proves two independent restart-style disk reads after one production
  save. It does **not** claim sequential repeated saves between reloads.
- Before the actual `Canvas.render_result_img()` call, the generator asserts
  that the evidence-only dashed polygon has been removed from the scene.
- The extreme/effects source rectangle is computed from `itemsBoundingRect()`
  plus explicit padding, and the containment assertion is recorded.
- Migration fixtures are emitted from `FEATURE_SHA:path` Git blobs rather than
  copied from mutable working-tree files; `migration-fixtures/manifest.json`
  records and verifies each destination SHA-256.
