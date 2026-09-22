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
  -> LLMTranslator.translate_textblk_lst(...)
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

## Codex backend

**Modules → Codex** signs in with ChatGPT using browser OAuth and PKCE.
Its HTTP client (`httpx[socks,brotli]`) and credential-storage libraries (`keyring`
and `cryptography`) are core dependencies, installed and checked by the normal
startup dependency flow. No Codex SDK, native process, or LiteLLM dependency is required.
The direct subscription protocol follows the Codex backend and may require
updates when that service changes.

[`codex.py`](../../ballontranslator/modules/codex.py) owns OAuth, credential
rotation, and the direct ChatGPT Responses and image transports. Credentials are
encrypted with Fernet in `config/codex/http-auth.json`; a small random encryption
key is held by the native system credential store through `keyring`. When secure
storage is unavailable during a save, the existing portable obfuscation is used
instead. Obfuscation is reversible and is not encryption; its use is logged.
Windows Credential Manager, macOS Keychain, and Linux
Secret Service (including KDE's bridge) are supported; file-based keyring plugins
are not treated as secure storage. File writes are atomic, with private POSIX permissions.
An unreadable encrypted file is preserved and reported; reads never replace a
missing key or downgrade protection. Legacy plaintext files are ignored and show
a signed-out state; a new sign-in replaces them. Existing SDK/CLI credentials are neither imported
nor changed. Config and profile exports contain no tokens.
Token rotation is serialized; a failed save retains the rotated token in memory
and requires successful persistence before reuse.

Translator and OCR requests are stateless: only the supplied instructions, history, current input,
and images are sent, with tools disabled. The existing pipeline job boundary owns
cancellation. Each requester assigns a cache/session key per job, profile, model,
and account generation; pages, ordinary retries, context recovery, and compaction
retain that key. A new job or account change gets a new key. Refreshing an access
token does not change identity. There is no hidden conversational history or
agent mode. Cancellation interrupts network waits and rejects late results.

[`codex_account.py`](../../ballontranslator/ui/codex_account.py) owns asynchronous
GUI account operations; [`CodexSettingsPanel`](../../ballontranslator/ui/codex_settings.py)
owns the dedicated account, model, and request settings UI. One canonical
`LLMProfile` with ID `codex` retains the existing requester and shared-selector
contract. It is excluded from the API profile editor and clipboard imports.
Config loading accepts only that canonical Codex identity; malformed entries are
discarded without converting profiles or remapping their IDs. Restoring API
profiles preserves Codex settings.
Account state follows committed credential changes independently of model refresh
success, so refresh failures do not incorrectly switch the account button.
`module.codex_models` is the single saved public catalog and supplies text/vision
model options. Without a catalog, Codex's own built-in offline model lists seed both
selectors, with saved custom selections retained. These defaults do not read API
profiles; a nonempty authenticated catalog replaces them according to its modalities.
Text, vision, and image capabilities remain available in the UI independently of
sign-in. Sign-in and explicit refresh retrieve the authenticated account's model
catalog; a successful empty result replaces stale metadata. Failed refreshes and
sign-out retain the cache and selected models. Viewing or selecting settings
performs no network or credential IO. After GUI startup,
the account worker restores saved sign-in and refreshes models, renewing an expired
access token when needed. Headless requests restore credentials on demand.
Requests require the
app's ChatGPT account and cannot fall back to API billing.

`CodexSignInRequiredError` distinguishes missing sign-in from rejected authentication
and stops retries and the current Run, including remaining headless directories.
HTTP 401 receives one token-renewal attempt before becoming an invalid sign-in;
permission, quota, and model-access failures remain separate. Invalid authentication
is cached in memory until successful sign-in, without deleting the saved credentials.
The GUI account controller owns one recovery dialog shared by refresh and module
failures. Signing in updates that dialog but never replays interrupted work.
Startup without saved credentials remains quiet.

Translator and OCR use the existing retry, token-reporting, and project-persistence
paths. Their feature owners define prompts and response contracts. Explicit
model/reasoning choices are validated. Requests honor image detail and the owning
module's HTTP proxy; account operations use
standard environment proxy settings. Unsupported sampling and output-token controls
remain hidden. Only a completed final response reaches the parser; truncation,
authentication, and quota failures require user action, while context overflow uses
the existing optional-history recovery.

Inpainting uses `LLMInpaint` and the shared `LLMImageRequester`, with Codex's fixed
`gpt-image-2` model offered independently of the text/vision catalog and sign-in state.
The image transport follows the [native Codex image API](https://github.com/openai/codex/blob/main/codex-rs/codex-api/src/endpoint/images.rs):
PNG references go to `/images/edits`, while existing Image-card requests without
context use `/images/generations`. Only inline image results are accepted.
The module's proxy, timeout, shared throttle, retries, and cancellation apply.

The subscription endpoint has no native mask parameter. Inpainting sends the
source crop and a labelled black/white mask reference; downscaling preserves thin
marked regions. Crops beyond the model's 3:1 aspect limit are padded before the
request and unpadded afterward to preserve alignment. The existing crop pipeline
owns context margins and alpha handling.
Masked results are resized to the input crop and composited only into the original
mask, so model changes outside it are discarded. Empty masks require no request.
Image calls are independent of translation history and consume the ChatGPT
subscription allowance; there is no API-billing fallback. Generative reconstruction
inside the mask can vary, especially on fine screentones or line art.

Brush and rectangle inpainting share a selection stored in `drawpanel`, independent
of Run's module, profile, and model. Their menus reuse `ModuleSelectionMenu` with
local selection values. A nonempty stripped prompt override replaces the selected
profile's inpainting prompt for drawing requests; blank overrides use that prompt.
Rectangle edits use the selected segmentation method when `Use mask` is checked.
Unchecking it skips segmentation, omits LLM mask conditioning, and applies the full
returned crop; local inpainters instead receive a full-rectangle mask. The whole
rectangle is recorded as edited for undo and erasing. Drawing LLM edits bypass
the native flat-background fill shortcut.
`ModuleManager.canvas_inpaint()` snapshots the draw module and copied profile when
submitted, so queued requests retain their model, prompt, and mask mode without
changing saved profiles. Canvas and Run prepare and use the same inpainter only after its worker
is idle; a page change discards queued canvas work and obsolete results.
Draw request logs report the backend, model, rectangle, mask mode, and elapsed
worker time without logging prompts or image data; local background fills are
logged in the drawing panel.

## Request contract

`LLMTranslator.concate_text` is `False`. Each non-empty source block becomes a
one-based item in the current JSON array. OpenAI-compatible profiles retain the
numeric-map response:

```json
{"1":"Translated text"}
```

Codex uses a fixed strict schema with integer IDs and string translations:

```json
{"translations":[{"id":1,"translation":"Translated text"}]}
```

Its schema is independent of the current block count, avoiding a changing schema
before the reusable message prefix. Prompt instructions and history examples use
the same shape. Cache hits still depend on the backend; stable requests do not
guarantee reuse, and the subscription endpoint does not support the API's
`prompt_cache_options` diagnostics.

With Summary enabled, `page_summary` precedes `translations`.
The latter keeps the provider's map or array shape; for example:

```json
{"page_summary":"Short factual page summary","translations":{"1":"Translated text"}}
```

`parse_translation_response()` owns compatibility response shapes. For text
pages, accepted responses must contain exactly IDs `1..N`, once each. Codex array
items reject coerced IDs and non-string translations. A missing or malformed
summary never discards an otherwise complete set of translations.
Parsing accepts either field order for compatibility; the prompt, schema, and
history examples put `page_summary` before `translations`.

With both Vision and Summary enabled, full-page calls also request summaries
for pages without source text. They use the same prompt, context, and image
suffix as normal pages, with an empty input array. Only a usable `page_summary`
is required; translation payload formatting and IDs are ignored. Missing or
blank summaries are retried. Existing-summary and overwrite rules still apply.

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
`FIN_TRANSLATE`, and every source-bearing block has a stored translation.
Pages without source text are eligible when the summary response contract is
active and they have a saved summary; their history pairs use an empty input
array and an empty translation map or array for the selected contract. Explicit
`translation_target` metadata must match the active target; missing metadata
remains accepted for older projects.
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

History pages are indivisible. A budget overflow removes oldest retained pages
before appending the just-completed page; eviction planning reserves space for
that append. A page larger than the available full budget is skipped.

## Page summaries and compact memory

`Summary` is independent of `+history`. When enabled, saved summaries
through the current page can guide translation even when their pages are
incomplete or history is disabled.

When a new summary is needed, the prompt first requests a factual summary of the
current page's key events or new information useful for understanding the current and later dialogue,
grounded in its text or image and using established character names. It then
instructs the model to use that newly written summary, saved summaries, compact
memory, and any attached image as context for translating each line. Both
fields are generated in one response. The summary's 500-word ceiling is prompt
guidance; accepted summary text is not truncated.

Unless overwrite is enabled, the current page summary is retained as required
input. Older summaries already represented by selected bilingual history are
not repeated. Summaries covered by compact memory are omitted from the raw
summary suffix; their saved project records remain intact. Remaining summaries
form the newest chronological suffix that fits the shared context budget.

The translation request asks for a new target-language summary only when the
current page has none. Otherwise, the current-page instructions request an empty
`page_summary`; any unwanted generated replacement is ignored.
`Overwrite Existing Summary` instead omits the raw
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

- before translation, overflowing the combined history and raw-summary budget
  compacts uncovered summaries immediately before bulk eviction; this also
  applies when rebuilding history or when only saved summaries are available;
- after the last project page finalizes, remaining uncovered summaries are
  compacted even if they still fit.

The request merges previous memory with an oldest summary batch and returns the
complete memory body as plain text in the active target language. Empty output
is retried. Its prompt retains supported identities, relationships, and essential
ongoing plot context, pruning duplicate, incidental, and superseded material.
It requests at most 600 words; this is a prompt instruction, not a validated
length limit. Compaction has the
selected profile's provider/output limits but is not capped by the translation
context budget. A successful pre-translation compaction is saved before
assembling the translation request, allowing a retried page to reuse the new
memory prefix.
If memory or an input summary changed in flight, the generated result is not
written. An exhausted compaction failure stops the run rather than being
retried again by every later page.

Page summaries and memory are independently user-owned. Editing or clearing one
does not silently invalidate or regenerate the other; the context editor is the
explicit review and correction boundary.

## Budget and provider prefix caching

The context token budget covers:

- current and prior saved summaries;
- bilingual history pairs.

The current-page summary is retained even when it consumes the budget; optional
older summaries and history receive only the remaining space. Compact memory
is retained separately and never reduces this allowance or evicts history.
The current translation batch, system contract, glossary, image, and output are
also outside this budget. The provider's actual context limit still applies to
the full request. Known models use `tiktoken`; unknown models use the
deterministic fallback estimator. Context diagnostics report history and saved
summaries as `tokens=used/budget`, with `memory_tokens` reported separately.
Eviction reports `action=evict` and `summaries_evicted` for the retired summary batch.

Budget-driven eviction targets `HISTORY_LOW_WATER_RATIO = 0.50` of the budget
for the next request's combined history and raw summaries, including the newly
appended history page. Rebuilds also select history toward this target. The
current-page summary remains required, and one indivisible history page may
exceed the soft target if it fits the available full budget.

There is no application-managed provider cache. Adjacent `+history` prompts are
arranged so each normally extends the previous prefix:

```text
page 1: S | U1
page 2: S | U1 | A1 | U2
page 3: S | U1 | A1 | U2 | A2 | U3
```

Bulk low-water eviction changes an early prefix once, then leaves room for more
append-only requests. A memory change starts a new cache epoch. Turning Summary
on or off changes the system/response contract. While enabled, pages with and
without saved summaries share that contract and the same history response shape;
the per-page summary-generation decision belongs only to the current user suffix.

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
- An in-flight OpenAI-compatible synchronous call cannot be interrupted; the
  stop event prevents subsequent attempts and interrupts waits. Codex also
  cancels in-flight HTTP and sign-in waits.

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
`tests/test_llm_translator.py`, `tests/test_llm_chat.py`, `tests/test_codex*.py`, and
`tests/test_proj_imgtrans_translation_context.py`. Image transport, mask geometry,
and canvas cancellation are covered by `tests/test_llm_inpaint.py` and
`tests/test_canvas_inpaint_lifecycle.py`; selector integration lives in
`tests/test_module_selection_menu.py` and `tests/test_drawing_inpainter.py`.

```bash
QT_QPA_PLATFORM=offscreen /opt/miniconda3/envs/common/bin/python \
  -m pytest -q tests/test_llm_translation_*.py tests/test_llm_translator.py \
  tests/test_llm_chat.py tests/test_codex*.py tests/test_proj_imgtrans_translation_context.py
```
