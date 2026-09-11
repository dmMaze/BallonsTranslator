# LLMTranslator

This guide describes the stable contracts and ownership boundaries of LLM
translation. The code and focused tests remain authoritative.

## Architecture

| Concern | Owner |
| --- | --- |
| Translation prompt, message order, JSON schema, and response parsing | [`llm_translation_contract.py`](../../ballontranslator/modules/translators/llm_translation_contract.py) |
| Request snapshots, retries, history orchestration, summaries, and compaction | [`trans_llm.py`](../../ballontranslator/modules/translators/trans_llm.py) |
| Provider clients, throttling, endpoint quirks, and completion normalization | [`llm_chat.py`](../../ballontranslator/modules/llm_chat.py) |
| Official Codex App Server stdio transport and owned process lifecycle | [`llm_codex.py`](../../ballontranslator/modules/llm_codex.py) |
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

## Codex subscription backend

Install the official Codex CLI and run `codex login` with ChatGPT. In an LLM
profile, select **Translation / OCR Backend → Codex App Server**, or select
**Codex** directly from the translator or OCR menu. The built-in profile is
included in fresh configurations and added once when loading older configurations.
The migration preserves existing profiles and selections; its saved marker keeps
later user deletions from being undone at startup.
Set the text and vision model to a model available to the signed-in account.
Both menus include `gpt-6-astra`, `gpt-5.6-sol`, `gpt-5.6-terra`, and
`gpt-5.6-luna`, alongside `gpt-5.5`. Astra requires a recent Codex client;
an upgrade-required error means updating the CLI or pointing **Codex Executable**
to a newer installed official client. Use the explicit Sol
name: the `gpt-5.6` alias is not accepted by the tested ChatGPT-backed App Server.
Loading a saved built-in profile merges new choices without changing its
selected models or custom choices.
The text and vision defaults are both `gpt-5.6-sol`; account/client availability follows the
[official Codex model catalog](https://learn.chatgpt.com/docs/models).

If `codex` is not on the application's PATH, set **Codex Executable** to its full
path. Run `codex login status` to check authentication, or `codex login` to open
ChatGPT sign-in in a browser. Use the same executable, OS user, and `CODEX_HOME`
as the application. This backend requires ChatGPT login, not API-key login.
Codex reads its existing credentials from `CODEX_HOME/auth.json` (by default
`~/.codex/auth.json`, or `%USERPROFILE%\.codex\auth.json` on Windows) or the OS
credential store, according to `cli_auth_credentials_store`. Authentication
and token refresh remain owned by Codex; BallonsTranslator never reads or
stores ChatGPT tokens, and needs no API key in its Codex profile. See
[official authentication guidance](https://learn.chatgpt.com/docs/auth).

Codex thinking choices use an offline snapshot of `model/list`, supplemented
by live-verified `none` support. The profile editor, shortcut menus, and request
validation use the same capability table in `llm_profiles.py`. Sol, Terra,
Luna, and GPT-5.5 accept `none`; Astra rejects it. The model catalog does not
advertise `none`, although the App Server accepts it and the
[Sol model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
lists it. Astra, Sol, and Terra offer reasoning through `ultra`; Luna stops at
`max`, and GPT-5.5 at `xhigh`. `Auto`, `Disabled`, and `minimal` are not offered.

Codex profiles save text `thinking_level` and OCR `vision_thinking_level`
independently, both defaulting to `none`. Each selector follows its own model;
the request carries the corresponding effort even when both models have the
same name. Translation (including its optional image context and summaries)
uses the text setting. Old profiles without a vision effort copy the previous
shared setting once. Unsupported settings reset with a warning to `none`, or
`medium` for Astra. Unknown models have no selectable efforts and require a
supported model before a request can run. HTTP profiles retain their existing
shared thinking behavior and legacy `None`/`Auto` interpretation.

Both `LLMTranslator` (including Vision, history, and summaries) and `LLMOCR`
(crop and full-page OCR) use this backend. Existing profiles default to the
OpenAI-compatible HTTP backend. Image generation/inpainting remains a separate
HTTP integration and still requires its image endpoint and credentials.

Each request launches a hidden local `codex app-server --listen stdio://` process
in a temporary directory and creates a fresh thread, ephemeral by default. Prior
messages retain their roles through `thread/inject_items`; the current image is passed directly
as an image input. JSON Schema and reasoning effort are forwarded. Codex chooses
its output limit and sampling settings; the HTTP Max Tokens, temperature,
top-p, penalties, image detail, and proxy settings do not control Codex calls.
Its existing CLI/network environment applies. No listening HTTP port or adapter
service is required. Tool integrations and workspace access are disabled for
these translation/OCR requests.

The profile editor hides unsupported HTTP controls for Codex; the translation
and OCR module parameter dialogs also omit Proxy without erasing its saved HTTP
value. Vision, Summary, Overwrite Summary, history, and the prior-context token
budget remain available: they work with Codex. Vision adds image input, history
adds prior-page text, and summaries add output and may require compaction calls.
They can improve quality and consistency, so they are not disabled automatically.
OCR page batching, masking, and reading order remain available as well.

**Save Codex Sessions (token monitors)** opts into official session persistence
(`thread/start.ephemeral=false`). Codex writes its standard JSONL under
`$CODEX_HOME/sessions` (default `~/.codex/sessions`), readable by tools such as
[token-monitor](https://github.com/Javis603/token-monitor) through Tokscale.
Use the same Codex home in both applications. This saves conversation content,
including prompts and images, as well as token usage. The switch defaults off;
disabling it only affects future requests and does not delete saved sessions.
Requests still start fresh, without resuming a saved thread. Past ephemeral
requests cannot be recovered by enabling this option.

Completed translation, summary-compaction, and OCR responses log token usage
at INFO level in the application's `logs/*.log`: `LLM token usage` for
translation/summaries and `LLM OCR token usage` for OCR. Available fields include
`prompt`, `completion`, `reasoning`, `total`, `cache_hit`, and `cache_write`;
reasoning and cache details are already included in the totals. OCR also logs
its module-lifetime `cumulative_total`. Missing reports say `usage=unavailable`.
Codex uses the latest thread-total snapshot for its fresh request,
so repeated updates are not added together and intermediate model calls are
included. These response logs are not account-wide billing or quota records.

Full-page batches (including headless runs) and selected-block tasks log each
enabled LLM stage separately as `LLM OCR run usage` and `LLM translation run usage`,
followed by `LLM run usage` for the combined total, once after their LLM workers
finish. Each line includes tokens and estimated cost. Counters reset per task;
summary compaction belongs to translation, and completed retry/output-limit
responses belong to their requesting stage. `total_tokens` includes only reported
usage; `missing_usage_requests`
identifies requests without a usable report, including failed or cancelled calls.
Session persistence is not required for these in-memory counters.

`estimated_cost_usd` is the Standard OpenAI API equivalent, not subscription
billing. Rates in `context/token_usage.py` were verified on 2026-09-11 against
[OpenAI pricing](https://developers.openai.com/api/docs/pricing) and the
[GPT-5.5 model page](https://developers.openai.com/api/docs/models/gpt-5.5).
The five built-in Codex models are covered. Estimates account for cached input,
reported cache writes (Astra/5.6), and the >272K input long-context tier per
response; reasoning is already included in output. They exclude Fast mode,
regional surcharges, and third-party provider pricing. Codex thread totals can
combine internal model calls, so their long-context estimate is approximate.
Unknown prices or incomplete usage produce `estimated_cost_usd=unavailable`
with a `priced_subtotal_usd` and `unpriced_requests`, not a guessed zero bill.

**Codex Timeout** bounds the entire request (default 180 seconds). Cancellation
requests `turn/interrupt` when the turn ID is known; cleanup always closes the
owned server. Missing CLI/login, quota exhaustion, protocol failures, and
uncertain completion stop the run without application-level resubmission.
Explicit context-window rejection uses the existing context recovery path;
completed but invalid model output follows existing parsing/retry rules.
Completed pages and project saves keep their existing ownership and resume rules.

Use a current CLI with the App Server `thread/inject_items` and `environments`
fields; experimental protocol access is negotiated during initialization.
See the [official App Server documentation](https://learn.chatgpt.com/docs/app-server).
The transport tests run without login. An opt-in test exercises real crop/full-page
OCR, two-page vision translation with history, and project save/reload:

```powershell
$env:BALLONTRANSLATOR_CODEX_LIVE = '1'
python -m pytest -q -s tests/test_llm_codex.py -k CodexLiveTest
```

This test uses subscription quota and only generated test images. Set
`BALLONTRANSLATOR_CODEX_EXECUTABLE` if the CLI is outside PATH.

## Request contract

`LLMTranslator.concate_text` is `False`. Each non-empty source block becomes a
one-based item in the current JSON array. The ordinary response is:

```json
{"1":"Translated text"}
```

When the request asks for a page summary, it requests that field first:

```json
{"page_summary":"Short factual page summary","translations":{"1":"Translated text"}}
```

`parse_translation_response()` owns compatibility response shapes. For text
pages, accepted responses must contain exactly IDs `1..N`. A missing or malformed
summary never discards an otherwise complete translation map.
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
array and translation map. Explicit `translation_target` metadata must
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

History pages are indivisible. A budget overflow removes oldest retained pages
before appending the just-completed page; eviction planning reserves space for
that append. A page larger than the available full budget is skipped.

## Page summaries and compact memory

`Summary` is independent of `+history`. When enabled, saved summaries
through the current page can guide translation even when their pages are
incomplete or history is disabled.

The prompt first requests a factual summary of the current page's key events
or new information useful for understanding the current and later dialogue,
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
- An in-flight synchronous HTTP provider call cannot be interrupted; the stop
  event prevents subsequent attempts and interrupts waits. Codex requests also
  interrupt the active turn and close their owned server.

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
