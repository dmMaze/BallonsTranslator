# LLMTranslator

This is an orientation guide for maintainers and coding agents. The code and
tests are authoritative; use this document to find the right owners and
preserve behavior that spans several files.

## Mental model

- One request sends the current non-empty text blocks as a numbered JSON batch.
- `page` means no prior-page examples. It does not force a whole-page caller.
- `+history` adds completed earlier pages as chronological user/assistant pairs.
- `ProjImgTrans` is authoritative. The in-memory history window is a disposable
  optimization for sequential requests and provider prefix caching.
- Glossary selection is independent of history and works in either mode.

## Owners

| Concern | Owner |
| --- | --- |
| Prompt assembly, provider request, parsing, runtime history integration | [`trans_llm.py`](../../ballontranslator/modules/translators/trans_llm.py) |
| Text-block preprocessing, postprocessing, and history-commit decision | [`base.py`](../../ballontranslator/modules/translators/base.py) |
| History selection, rebuild, eviction, and overflow recovery | [`context/history.py`](../../ballontranslator/modules/context/history.py) |
| Glossary parsing and matching | [`context/glossary.py`](../../ballontranslator/modules/context/glossary.py) |
| Token estimates and provider usage fields | [`context/token_usage.py`](../../ballontranslator/modules/context/token_usage.py) |
| Context settings and LLM profiles | [`config.py`](../../ballontranslator/utils/config.py), [`llm_profiles.py`](../../ballontranslator/utils/llm_profiles.py) |
| Full-page and selected-block worker boundaries | [`module_manager.py`](../../ballontranslator/ui/module_manager.py) |
| Page order, completion, target metadata, and load identity | [`proj_imgtrans.py`](../../ballontranslator/utils/proj_imgtrans.py) |

## Request flow

GUI and headless runs use the same worker path:

```text
worker
  -> BaseTranslator.translate_textblk_lst(...)
     -> omit empty sources and apply pre-translation substitutions
     -> decide commit_history_window
     -> LLMTranslator.translate(...)
        -> resolve profile and freeze RequestContext
        -> assemble messages and call chat.completions.create(...)
        -> parse JSON and validate the exact ID set
        -> commit reusable window state after a valid parse
     -> finalize results and assign TextBlock.translation
  -> mark the page translated when the caller's completion rule passes
```

`LLMTranslator.concate_text` is `False`: every non-empty source becomes one
one-based JSON item. The canonical response is:

```json
{"translations":[{"id":1,"translation":"..."}]}
```

The parser tolerates the compatibility shapes implemented in
`_parse_response()`, but always requires exactly IDs `1..N`. Full-page results
then run normalization, result substitutions, and optional uppercase before
the page can become history. Selected-block translation retains its narrower
postprocessing behavior.

## Prompt layout and context modes

The message order is deliberately stable:

```text
system: translation contract + profile instructions
system: complete glossary                         # All mode only
user / assistant: completed history page pairs   # +history only
user: current JSON + matching glossary            # Matching suffix only
```

The system contract fixes target language, IDs, item count, and JSON shape;
profile instructions may affect wording and style only.

`pcfg.module.translate_context` is the generic `textblock`/`page` grouping
setting used by other translators. LLM context is the separate
`pcfg.module.llm_translate_context` setting shown as `page`/`+history`.

### `page`

No earlier pages are sent. With no glossary, the request is just `system` plus
the current `user` message. A request in this mode clears `_history_window`; a
later switch back to `+history` rebuilds safely from project data.

The current request may still be a selected subset. “Page” only means that
prior-page history is disabled.

### `+history`

Earlier pages are rendered as glossary-free source/translation examples. A
page is eligible only when:

- it precedes the current page in `project.pages` order;
- `FIN_TRANSLATE` is set;
- every source-bearing block has a non-empty stored translation;
- at least one source-bearing block exists;
- `translation_target` matches the active target language.

Missing target metadata remains accepted for old projects. Explicit metadata
for another target makes the page ineligible. Snapshots contain immutable
strings, use the current pre-translation source substitutions, and use the
fully finalized translations stored in the project.

Calls without both `project` and `page_key` still translate, but cannot read or
advance reusable project history.

## Runtime history window

`RequestContext` freezes history and glossary for one request. Ordinary retries
reuse the same messages. `_history_window` records only the most recent
successful sequential state; it does not persist project data.

The window can grow only when the same project load and prompt-shaping settings
are active, retained snapshots still match, and the requested page immediately
follows the last successful request page. Otherwise history is rebuilt from a
recent eligible project suffix.

The key covers project load identity, source language, model, rendered system
prompt, and history budget. The system prompt already captures target language
and profile instructions. Glossary settings are excluded because stored
history pairs are glossary-free.

Diagnostics name the transition: `empty`/`rebuild` selects a recent eligible
suffix, `grow` appends the previous page, `reuse` keeps the prefix when that
page is too large, `evict` removes oldest pages before appending, and
`context-recovery` removes more history after a provider overflow.

Project reloads, page jumps, prompt/model/language/budget changes, edited
snapshots, or an incomplete previous page force a rebuild.

### Commit semantics

`commit_history_window` authorizes committing the selected prior-page window
and remembering the current `page_key`; it does not mark the current page
translated or insert it into history immediately.

`BaseTranslator.translate_textblk_lst()` enables it for a full-page call or a
selected call containing every source-bearing block on the page. A partial
selection may read history but must not advance the reusable window.

The LLM commits only after valid JSON parsing. The worker marks the current
page complete only after postprocessing and assignment succeed. Full-page
workers clear stale completion before starting. Thus a failed, partial, or
wrong-target page cannot leak into later history.

## Prefix-cache rationale

There is no application-managed provider cache. Adjacent prompts are arranged
so each usually extends the previous one:

```text
page 1: S | U1
page 2: S | U1 | A1 | U2
page 3: S | U1 | A1 | U2 | A2 | U3
```

Providers that cache leading tokens can reuse the unchanged prefix. This is an
optimization, never a correctness requirement.

Rebuild and eviction target `HISTORY_LOW_WATER_RATIO = 0.60` rather than
filling the budget. Removing an oldest page changes an early prefix; bulk
eviction pays that cache break once and leaves room for several append-only
requests.

The history budget counts rendered history pairs only. System instructions,
the current batch, glossary, and output are outside it. Known models use
`tiktoken`; unknown models use the deterministic fallback estimator. Pages are
indivisible. On a recognized provider overflow, recovery removes oldest whole
pages without consuming the ordinary retry budget and never truncates the
current input or glossary.

For useful cache behavior, process contiguous pages in order, keep stable
material first, avoid rewriting old pairs, and keep one translator instance's
`+history` queue sequential. Compare context actions with provider `cache_hit`,
`cache_miss`, and `cache_write` fields before tuning the low-water ratio.

`Matching` glossary prompts are intentionally volatile suffixes. When a live
page contains matches, its later glossary-free history form is not byte-for-byte
identical, so the exact prefix ends before that page; older clean pairs remain
reusable. `All` is more stable when the entire glossary is unchanged, but sends
every term on every request.

## Glossary

Supported UTF-8 formats are:

```json
[{"src":"勇者","dst":"Hero","info":"title"}]
```

```text
勇者<TAB>Hero<TAB>title
勇者 -> Hero # title
```

`.tsv` uses two or three fields. `.txt` accepts tabs or arrows. Blank lines and
lines beginning with `#`, `//`, or `\\` are ignored.

- `Matching` performs case-insensitive literal matching against the current
  request and appends selected entries to the current user message.
- `All` adds every entry as a system message before history.

Entries preserve file order. Exact duplicates are removed; conflicting targets
for the same case-insensitive source are rejected. An empty path disables the
feature. A configured missing or malformed glossary fails explicitly rather
than silently translating without it.

## Reliability and diagnostics

- History changes remain speculative until parsing succeeds; a final failed
  attempt preserves the prior committed window.
- Context overflow evicts history separately from ordinary retries. When no
  history remains, the error is re-raised.
- `_history_window` is cleared on unload and is safe to rebuild from the
  project after restart.
- `ProjImgTrans.load_identity` invalidates a window even when the same path is
  reopened.
- The synchronous provider call cannot be interrupted in flight; the stop event
  prevents later attempts and interrupts waits.

Context logs normally expose aggregate page, action, page-count, token-budget,
attempt, and provider cache fields. A healthy contiguous run is usually
`empty/rebuild -> grow ... -> evict -> grow`. Missing cache fields mean “not
reported.” ID-count failures log the current prompt, so treat those logs as
potentially containing project and glossary text.

## Change checklist

Before changing this subsystem, preserve these contracts:

- project completion is authoritative; the runtime window is disposable;
- history contains immutable whole-page pairs, not `TextBlock` references;
- partial selections do not advance history;
- retries use one frozen request snapshot;
- stable prompt content precedes volatile content;
- current input and glossary are not sacrificed to history recovery;
- page completion happens after result finalization;
- the translation queue remains sequential when `+history` is active.

Focused executable specifications are
[`test_llm_translator.py`](../../tests/test_llm_translator.py),
[`test_llm_translation_context.py`](../../tests/test_llm_translation_context.py),
[`test_proj_imgtrans_translation_context.py`](../../tests/test_proj_imgtrans_translation_context.py),
and [`test_run_pipeline_dialog.py`](../../tests/test_run_pipeline_dialog.py).

```bash
QT_QPA_PLATFORM=offscreen /opt/miniconda3/envs/common/bin/python \
  -m unittest discover -s tests -p 'test_llm*.py'
```
