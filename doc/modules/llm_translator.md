# LLMTranslator

This guide describes the stable contracts and ownership boundaries of LLM
translation. The code and focused tests remain authoritative.

## Architecture

| Concern | Owner |
| --- | --- |
| Translation prompt, message order, JSON schema, and response parsing | [`llm_translation_contract.py`](../../ballontranslator/modules/translators/llm_translation_contract.py) |
| Request snapshots, retries, history orchestration, summaries, and compaction | [`trans_llm.py`](../../ballontranslator/modules/translators/trans_llm.py) |
| Provider clients, throttling, endpoint quirks, and completion normalization | [`llm_chat.py`](../../ballontranslator/modules/llm_chat.py) |
| Image encoding shared by LLM modules | [`llm_vision.py`](../../ballontranslator/modules/llm_vision.py) |
| History, saved-context packing, glossary parsing, and token estimates | [`context/`](../../ballontranslator/modules/context) |
| Text-block preprocessing, finalization, and page-coverage decisions | [`base.py`](../../ballontranslator/modules/translators/base.py) |
| Full-page and selected-block worker lifecycle | [`module_manager.py`](../../ballontranslator/ui/module_manager.py) |
| Project completion, saved summaries, memory, and load identity | [`proj_imgtrans.py`](../../ballontranslator/utils/proj_imgtrans.py) |
| Settings and user-owned context editing | [`config.py`](../../ballontranslator/utils/config.py), [`run_pipeline_dialog.py`](../../ballontranslator/ui/run_pipeline_dialog.py), [`llm_context_editor.py`](../../ballontranslator/ui/llm_context_editor.py) |

GUI and headless translation share this path:

```text
worker
  -> BaseTranslator.translate_textblk_lst(...)
     -> preprocess non-empty sources and decide page coverage
     -> LLMTranslator.translate(...)
        -> freeze profile, project context, and optional page image
        -> assemble messages and call LLMChatRequester
        -> validate translations and optional page summary
        -> commit the reusable history window after a valid parse
     -> finalize and assign TextBlock.translation
  -> mark a successfully finalized page translated
  -> let LLMTranslator persist pending summary/memory updates
```

`ProjImgTrans` is authoritative. `_history_window` and every `RequestContext`
are disposable runtime snapshots; neither replaces project state.

## Request contract

`LLMTranslator.concate_text` is `False`. Each non-empty source block becomes a
one-based item in the current JSON array. The ordinary response is:

```json
{"1":"Translated text"}
```

When the request asks for a page summary, the response is:

```json
{"translations":{"1":"Translated text"},"page_summary":"Concise page memory"}
```

`parse_translation_response()` owns compatibility response shapes, but every
accepted response must contain exactly IDs `1..N`. A missing or malformed
summary never discards an otherwise complete translation map.

Messages are assembled in cache-friendly prefix order:

```text
system: translation contract + profile instructions
system: complete glossary                         # All mode
system: compact memory                            # if saved and enabled
user / assistant: completed page examples        # +history
user: saved page summaries + current input + matching glossary + image
```

The system contract fixes the target language, ID set, item count, and response
shape. Profile instructions may affect style and wording only. Stable material
precedes page-specific material; an image is the last part of the final user
message.

Vision always uses the model selected for Translator. It attaches only the
current page image and asks the model to infer natural comic reading order for
interpretation while returning every translation under its original input ID.
Translation does not reorder project blocks; full-page LLM OCR owns its own
optional reorder operation.

Ordinary retries reuse the same rendered messages, saved-context snapshot, and
encoded image. Provider-facing input cannot change midway through one request.

## History

`pcfg.module.llm_translate_context` controls only prior-page examples:

- `page` sends no bilingual history. It does not require a whole-page caller.
- `+history` adds completed earlier pages as chronological, glossary-free
  user/assistant pairs.

A prior page is eligible when it precedes the current page, has
`FIN_TRANSLATE`, contains at least one source-bearing block, and every such
block has a stored translation. Explicit `translation_target` metadata must
match the active target; missing metadata remains accepted for older projects.
Snapshots contain immutable strings after configured source preprocessing and
use the finalized translations stored in the project.

`BaseTranslator.translate_textblk_lst()` authorizes a reusable-window commit
for a full-page call or a selected call containing every source-bearing block
on that page. Partial selections may read history but do not advance the
window. The LLM commits only after valid response parsing; workers mark page
completion only after postprocessing and assignment succeed.

Full-page workers clear prior completion before starting and restore it only
after successful finalization. Selected-block calls do not clear completion up
front; they mark the page complete only after all source-bearing blocks have a
translation.

The window grows only for the page immediately following the last successful
request with the same project load and prompt-shaping settings. Project reload,
page jump, model/language/prompt/budget change, compact-memory edit, changed
page snapshot, or incomplete previous page causes a rebuild from a recent
eligible suffix.

History pages are indivisible. Rebuild and eviction use
`HISTORY_LOW_WATER_RATIO = 0.60` to create room for several adjacent appends.
An oversized previous page is skipped rather than split.

## Page summaries and compact memory

`Summary` is independent of `+history`. When enabled, saved summaries
through the current page can guide translation even when their pages are
incomplete or history is disabled.

Unless overwrite is enabled, the current page summary is retained as required
input. Older summaries already represented by selected bilingual history are
not repeated; remaining summaries form the newest chronological suffix that
fits the shared context budget.

The translation request asks for a new target-language summary only when the
current page has none. `Overwrite Existing Summary` instead omits the raw
current summary, requests a replacement, and stores it only if usable summary
text is returned. A generated summary remains pending until a page completes.
The request-start record is compared before saving, so an edit or clear made
while the request is running always wins.

Compact memory is one optional project-level record rendered as a stable system
message before history. Its `covered_pages` metadata prevents redundant
automatic compaction; coverage is not shown in the editable body and is never
sent with translation requests. Memory applies regardless of history mode,
Vision, model changes, target changes, or recorded coverage.

Automatic compaction is a separate text-only request using the selected
Translator model:

- before translation, older uncovered summaries are compacted when they no
  longer fit the shared budget;
- after the last project page finalizes, remaining uncovered summaries are
  compacted even if they still fit.

The request merges previous memory with an oldest summary batch and returns the
complete memory body in the active target language. Compaction has the selected
profile's provider/output limits but is not capped by the translation context
budget. A successful pre-translation compaction is saved before assembling the
translation request, allowing a retried page to reuse the new memory prefix.
If memory or an input summary changed in flight, the generated result is not
written. An exhausted compaction failure stops the run rather than being
retried again by every later page.

Page summaries and memory are independently user-owned. Editing or clearing one
does not silently invalidate or regenerate the other; the context editor is the
explicit review and correction boundary.

## Budget and provider prefix caching

The context token budget covers:

- compact memory;
- current and prior saved summaries;
- bilingual history pairs.

Memory and the current-page summary are retained even when they consume the
budget; optional older summaries and history receive only the remaining space.
The current translation batch, system contract, glossary, image, and output are
outside this budget. Known models use `tiktoken`; unknown models use the
deterministic fallback estimator.

There is no application-managed provider cache. Adjacent `+history` prompts are
arranged so each normally extends the previous prefix:

```text
page 1: S | U1
page 2: S | U1 | A1 | U2
page 3: S | U1 | A1 | U2 | A2 | U3
```

Bulk low-water eviction changes an early prefix once, then leaves room for more
append-only requests. A memory change starts a new cache epoch. Requests that
ask for `page_summary` use a different system/response contract from requests
that do not, so those contracts intentionally do not share the full prefix.

`All` glossary mode is cache-friendly while the complete glossary is unchanged.
`Matching` glossary entries belong to the volatile current-page suffix; later
history remains glossary-free.

## Glossary

Supported UTF-8 files are JSON, TSV, and TXT:

```json
[{"src":"勇者","dst":"Hero","info":"title"}]
```

```text
勇者<TAB>Hero<TAB>title
勇者 -> Hero # title
```

`Matching` selects case-insensitive literal matches for the current input.
`All` sends every entry in a stable system message. Entries preserve file
order; exact duplicates are removed and conflicting targets for the same
case-insensitive source are rejected. A configured missing or malformed file
fails explicitly.

## Failure behavior and diagnostics

- Recognized context-length errors first remove optional prior summaries, then
  oldest whole history pages, without consuming the ordinary retry budget.
  Current input, current summary, memory, glossary, and image are retained.
- User-action errors such as a provider output-limit failure bypass retries and
  fallbacks, stop the run, and reach the UI boundary immediately.
- Invalid JSON, wrong IDs, unusable summaries, context actions, compaction
  decisions, and provider token/cache fields are logged at their owning layer.
  Debug response content can contain project or glossary text.
- `_history_window` is cleared on unload and can always rebuild from project
  state after restart.
- An in-flight synchronous provider call cannot be interrupted; the stop event
  prevents subsequent attempts and interrupts waits.

A healthy contiguous `+history` run usually reports
`empty/rebuild -> grow ... -> evict -> grow`. Missing provider cache fields mean
the provider did not report them, not that the application inferred a miss.

## Change checklist

Preserve these contracts when changing the subsystem:

- project state is authoritative and request snapshots are immutable;
- exact IDs remain mapped to their original blocks;
- partial selections do not advance page history or save generated summaries;
- user-owned summary and memory edits win over in-flight generation;
- stable content precedes volatile current-page content;
- context recovery never sacrifices the current input or glossary;
- page completion follows translation finalization;
- `+history` requests remain sequential for deterministic window state.

Focused specifications live in `tests/test_llm_translation_*.py`,
`tests/test_llm_translator.py`, `tests/test_llm_chat.py`, and
`tests/test_proj_imgtrans_translation_context.py`.

```bash
QT_QPA_PLATFORM=offscreen /opt/miniconda3/envs/common/bin/python \
  -m pytest -q tests/test_llm_translation_*.py tests/test_llm_translator.py \
  tests/test_llm_chat.py tests/test_proj_imgtrans_translation_context.py
```
