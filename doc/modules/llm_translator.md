# LLMTranslator

This is an orientation guide for maintainers and coding agents. The code and
tests are authoritative; use this document to find the right owners and
preserve behavior that spans several files.

## Mental model

- One request sends the current non-empty text blocks as a numbered JSON batch.
- `page` means no prior-page examples. It does not force a whole-page caller.
- `+history` adds completed earlier pages as chronological user/assistant pairs.
- `Vision` attaches only the current page image. `Summary` extends the same
  translation response with reusable page memory, with or without Vision.
- `Memory` applies editable project-wide context to every request. When history
  is active, summarized pages retired at a budget boundary can update it.
- `ProjImgTrans` is authoritative. The in-memory history window is a disposable
  optimization for sequential requests and provider prefix caching.
- Glossary selection is independent of history and works in either mode.

## Owners

| Concern | Owner |
| --- | --- |
| Translation prompt, message, schema, history-rendering, and response contract | [`llm_translation_contract.py`](../../ballontranslator/modules/translators/llm_translation_contract.py) |
| Runtime request, retry, context-snapshot, compaction, and persistence orchestration | [`trans_llm.py`](../../ballontranslator/modules/translators/trans_llm.py) |
| Chat client lifecycle, throttling, provider argument quirks, status normalization | [`llm_chat.py`](../../ballontranslator/modules/llm_chat.py) |
| Text-block preprocessing, postprocessing, and history-commit decision | [`base.py`](../../ballontranslator/modules/translators/base.py) |
| Bilingual history selection, rebuild, and whole-page eviction | [`context/history.py`](../../ballontranslator/modules/context/history.py) |
| Immutable request context, saved-summary fitting, compact-memory rendering, and overflow recovery | [`context/translation_context.py`](../../ballontranslator/modules/context/translation_context.py) |
| Glossary parsing and matching | [`context/glossary.py`](../../ballontranslator/modules/context/glossary.py) |
| Token estimates and provider usage fields | [`context/token_usage.py`](../../ballontranslator/modules/context/token_usage.py) |
| Context settings and LLM profiles | [`config.py`](../../ballontranslator/utils/config.py), [`llm_profiles.py`](../../ballontranslator/utils/llm_profiles.py) |
| Full-page and selected-block worker boundaries | [`module_manager.py`](../../ballontranslator/ui/module_manager.py) |
| Page order, completion, target metadata, saved summaries/memory, and load identity | [`proj_imgtrans.py`](../../ballontranslator/utils/proj_imgtrans.py) |
| User-owned summary/memory editing and main-window save integration | [`llm_context_editor.py`](../../ballontranslator/ui/llm_context_editor.py), [`mainwindow.py`](../../ballontranslator/ui/mainwindow.py) |

## Request flow

GUI and headless runs use the same worker path:

```text
worker
  -> BaseTranslator.translate_textblk_lst(...)
     -> omit empty sources and apply pre-translation substitutions
     -> decide commit_history_window
     -> LLMTranslator.translate(...)
        -> resolve profile and freeze RequestContext
        -> freeze the current image when Vision is enabled
        -> assemble messages and translation-specific API arguments
        -> LLMChatRequester calls chat.completions.create(...)
        -> parse translations and the optional page summary
        -> commit reusable window state after a valid parse
     -> finalize results and assign TextBlock.translation
  -> mark the page translated when the caller's completion rule passes
  -> persist matching pending summary and compact-memory updates
```

`LLMTranslator.concate_text` is `False`: every non-empty source becomes one
one-based JSON item. The canonical response is:

```json
{"1":"..."}
```

The parser also tolerates the compatibility shapes implemented by
`parse_translation_response()`, but always requires exactly IDs `1..N`.
Full-page results then run normalization, result substitutions, and optional
uppercase before the page can become history. Selected-block translation
retains its narrower postprocessing behavior.

With Summary enabled, the same text or multimodal response instead uses:

```json
{"translations":{"1":"..."},"page_summary":"..."}
```

A missing or malformed summary does not discard a complete translation map and
does not erase existing user-owned summary text.

## Prompt layout and context modes

The message order is deliberately stable:

```text
system: translation contract + profile instructions
system: complete glossary                         # All mode only
system: compacted older-page memory               # Memory only
user / assistant: completed history page pairs   # +history only
user: saved page context + current JSON + matching glossary + image
                                                    # image is Vision only
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
advance reusable project history or attach an image. Project memory can operate
without history whenever a project is supplied.

## Vision, summaries, and memory

Vision uses the selected profile's vision model. The encoded, bounded JPEG is
the last part of the current user message, so stable instructions, full
glossary, compacted memory, and recent history remain ahead of the volatile
page suffix. Retries reuse the frozen data URL and never replay older images.

Summary asks the already-selected text or vision translation request for concise
English narrative memory alongside translations, then fills an empty
`llm_visual_summary` record in that page's existing `image_info`. Provenance is
diagnostic only. A saved summary remains usable across source, image, language,
profile, model, and page-order changes until the user edits or clears it.

Saved summaries have their own immutable request snapshot and do not inherit
bilingual-history eligibility. With Summary enabled, current and preceding
saved summaries that are not already represented by selected history are added
to the final user message as read-only context. Thus an incomplete page cannot
become a translation example, but its user-owned summary can still guide later
translation. The current summary is retained; additional summaries are selected
newest-first as whole entries within the existing context budget. Because this
block is in the volatile final message, summary edits do not disturb the stable
system/memory/history prefix.

The translator keeps a generated summary pending until the worker has
postprocessed and assigned translations and marked the page complete. Partial
selections and failed pages therefore do not add one. Generated results never
replace an existing summary, including one the user clears while a request is
running; the context editor is the direct editing boundary.

Compact memory is an optional project-level record. When Memory is enabled, its
text is frozen into every request before recent history, irrespective of Vision,
Summary, history mode, model, language, current page, or recorded coverage.
Coverage metadata only prevents redundant automatic compaction; generated text
also starts with an explicit `Coverage:` line.

Automatic compaction runs when history selection leaves summarized older pages
outside the recent suffix, either at ordinary eviction or while rebuilding a
window. It reads saved summaries directly, so translation completion controls
only bilingual examples and cannot hide an edited summary from compaction. Each
request also checks omitted prior pages, allowing a summary added later to an
incomplete page to participate without forcing a history rebuild. One text-model
request merges the previous memory with newly omitted page summaries. Failure
or an oversized result keeps the previous record and ordinary whole-page
eviction. A successful result is staged until page finalization, then stored in
the project unless the user edited memory while the request was running.
The accepted checkpoint is capped by the budget left after mandatory summary
context and selected exact history, so compaction cannot silently displace a
page whose summary was absent from that compaction input.

Page summaries and compact memory remain independently user-owned. Editing a
summary already named by memory coverage does not silently regenerate,
invalidate, or overwrite the memory; the memory editor remains authoritative.

## Runtime history window

`RequestContext` freezes history, saved summary context, glossary, and optional
compacted memory for one request. Ordinary retries reuse the same messages.
`_history_window` records only the most recent successful sequential state; it
does not persist project data.

The window can grow only when the same project load and prompt-shaping settings
are active, retained snapshots still match, and the requested page immediately
follows the last successful request page. Otherwise history is rebuilt from a
recent eligible project suffix.

The key covers project load identity, source language, model, rendered system
prompt, history budget, memory enablement, and a digest of the editable memory.
The system prompt already captures target language and profile instructions.
Glossary settings are excluded because stored history pairs are glossary-free.

Diagnostics name the transition: `empty`/`rebuild` selects a recent eligible
suffix, `grow` appends the previous page, `reuse` keeps the prefix when that
page is too large, `evict` removes oldest pages before appending, and
`context-recovery` first removes optional prior summaries, then more history,
after a provider overflow. Current-page summary context is retained.

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

The history budget counts rendered history pairs, compacted memory, and saved
summary context. Current-page summary text and memory are not silently dropped
when their user-owned text exceeds that budget; recent history simply receives
no remaining space. Other
system instructions, the current batch, glossary, image, and output are outside
it. Known models use
`tiktoken`; unknown models use the deterministic fallback estimator. Pages and
summaries are indivisible. On a recognized provider overflow, recovery removes
optional prior summaries and then oldest whole pages without consuming the
ordinary retry budget; it never truncates the current input, current-page
summary, or glossary.

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
- Compaction or a user edit changes the early prefix once; later requests reuse
  that memory block until the next memory epoch.
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
- saved page summaries and project memory are user-owned optional data;
- stable prompt content precedes volatile content;
- current input and glossary are not sacrificed to history recovery;
- page completion happens after result finalization;
- the translation queue remains sequential when `+history` is active.

Focused executable specifications are
[`test_llm_chat.py`](../../tests/test_llm_chat.py),
[`test_llm_translator.py`](../../tests/test_llm_translator.py),
[`test_llm_translation_context.py`](../../tests/test_llm_translation_context.py),
[`test_proj_imgtrans_translation_context.py`](../../tests/test_proj_imgtrans_translation_context.py),
[`test_llm_context_editor.py`](../../tests/test_llm_context_editor.py),
and [`test_run_pipeline_dialog.py`](../../tests/test_run_pipeline_dialog.py).

```bash
QT_QPA_PLATFORM=offscreen /opt/miniconda3/envs/common/bin/python \
  -m unittest discover -s tests -p 'test_llm*.py'
```
