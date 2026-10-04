# LLMTranslator

## Ownership

| Concern | Owner |
| --- | --- |
| Prompts, message order, response schema, and parsing | [`llm_translation_contract.py`](../../ballontranslator/modules/translators/llm_translation_contract.py) |
| Request snapshots, retries, history, and compaction | [`trans_llm.py`](../../ballontranslator/modules/translators/trans_llm.py) |
| API clients, throttling, and provider compatibility | [`llm_chat.py`](../../ballontranslator/modules/llm_chat.py) |
| Profiles and defaults | [`llm_profiles.py`](../../ballontranslator/utils/llm_profiles.py), [`config.py`](../../ballontranslator/utils/config.py) |
| Codex account and subscription transport | [`codex.py`](../../ballontranslator/modules/codex.py) |
| API Responses and shared WebSocket transport | [`openai_responses.py`](../../ballontranslator/modules/openai_responses.py), [`responses_ws.py`](../../ballontranslator/modules/responses_ws.py) |
| Image dispatch and generation | [`llm_image.py`](../../ballontranslator/modules/llm_image.py), [`image_generation.py`](../../ballontranslator/modules/image_generation.py) |
| History selection, glossary, and context budgeting | [`context/`](../../ballontranslator/modules/context) |
| Workers and request validity | [`module_manager.py`](../../ballontranslator/ui/module_manager.py) |
| Project completion and saved context | [`proj_imgtrans.py`](../../ballontranslator/utils/proj_imgtrans.py) |
| Context editing | [`llm_context_editor.py`](../../ballontranslator/ui/llm_context_editor.py) |

GUI and headless translation share one lifecycle: preprocess blocks, freeze request
inputs, validate and finalize translations, then mark completion and save pending
context. `ProjImgTrans` owns saved state; request snapshots and history windows are
disposable.

## Profiles and backends

`backend="openai"` uses API keys and profile URLs, including compatible gateways.
`backend="codex"` uses the app's ChatGPT subscription account. Neither backend falls
back to the other's billing. Feature owners define prompts and response contracts;
transports own authentication and service compatibility.

Codex and supported direct OpenAI requests automatically use Responses WebSockets;
other API requests use HTTP. Connections and continuation state belong to the
active run, profile, model, credentials, and proxy. Reuse must respect changed
context and must never restore history removed by the feature owner.

Codex has one canonical profile, `codex`, separate from editable API profiles and
clipboard operations. Offline defaults and saved model choices remain usable
without a catalog. Profile selection and settings construction perform no network
or credential IO. [`codex_account.py`](../../ballontranslator/ui/codex_account.py)
owns asynchronous GUI account operations; headless requests restore credentials
on demand. Account changes and catalog refresh are independent, and signing in
does not replay interrupted work.

Codex credentials stay outside profile exports. Storage failures must preserve
existing secrets, and rotated credentials must be saved before reuse.

## Image editing

[`LLMInpaint`](../../ballontranslator/modules/inpaint/inpaint_llm.py) owns crop
preparation and compositing; `LLMImageRequester` owns dispatch, throttling, and
retries. Direct image models use their provider route; reasoning/image model pairs
use Responses image generation. Requests use the selected image profile,
independently of translation and OCR settings.

Source and mask references must remain aligned through resizing and padding.
Compositing honors the requested mask; an empty mask needs no request. Canvas
request ownership and validity are covered by the [draw panel guide](../ui/draw_panel.md);
text-effect integration belongs to the [text effects guide](../ui/text_effects.md).

## Request contract

Non-empty source blocks receive IDs `1..N`. Non-GPT API models return a numeric
translation map; recognized GPT API models and Codex return an array of integer
IDs and string translations. Prompts, schemas, history examples, and parsing must
agree. Accepted translations cover every ID exactly once without coercion;
compatibility shapes belong in `parse_translation_response()`.

An optional `page_summary` is generated before translations, but parsing accepts
either field order. A malformed summary does not discard valid translations.
Full-page Vision/Summary requests for textless pages require only a usable summary.

Messages keep stable context before current-page material:

```text
system: translation contract + profile instructions
system: complete glossary, then compact memory       # when enabled
user/assistant: completed-page example pairs          # +history
user: saved summaries + current input + matching glossary + image
```

The contract owns language, IDs, and response shape; profile instructions control
style and wording. Vision uses the Translator model and current page image while
preserving block IDs. Ordinary retries reuse frozen inputs.

## History and saved context

`page` mode sends no prior-page examples. `+history` adds chronological,
glossary-free pairs with page-local IDs. Eligible pages precede the current page,
are marked translated, and have translations for every source block. Explicit
target-language metadata must match; older projects without it remain compatible.
Project filenames are not sent.

History advances sequentially after successful full-page translation, including
selections covering every source block. Partial selections may read history but
cannot advance it or save generated summaries. Completion follows postprocessing
and assignment; full-page retries clear prior completion first. Page jumps,
project reloads, or relevant source, summary, and setting changes rebuild history.

Pages completed in the active run contribute their original response translations;
postprocessing and translation edits do not rewrite those examples. New runs use
saved translations. Page eligibility, source, language, and summaries remain
subject to validation.

Summary is independent of history mode. Saved summaries through the current page,
including incomplete or textless pages, can guide translation. Existing summaries
are retained unless overwrite is enabled; generated replacements remain pending
until page completion. Context already represented by history or compact memory
is omitted from requests without deleting project records.

Compact memory is a separate project record rendered before history. Compaction
uses the Translator model to merge uncovered summaries before budget eviction and
after the final page. Coverage metadata prevents repeated compaction. Successful
pre-translation compaction is saved before the next request; exhausted failure
stops the run.

In-flight summary and memory writes must preserve concurrent user edits or clears.
These records are independently owned: editing one does not regenerate or
invalidate the other.

## Budget, caching, and glossary

The context budget covers saved summaries and bilingual history. The current
summary is reserved; older summaries and whole history pages share the remainder.
Memory, current input, glossary, image, instructions, and output are outside this
budget, but still subject to the provider's complete-request context limit.

Provider caching is an optimization, never a correctness condition. Stable
prefixes and connection reuse do not guarantee cache hits. Provider output and
encrypted reasoning used for replay stay in runtime state, never project files,
and may increase actual input beyond the text-based history budget.

[`glossary.py`](../../ballontranslator/modules/context/glossary.py) owns UTF-8
JSON/TSV/TXT parsing. `Matching` uses case-insensitive literal source matches;
`All` supplies a stable system message. An empty path disables the glossary;
unreadable or malformed files and conflicting targets fail explicitly. See the
[parser tests](../../tests/test_translator_glossary.py) for formats and ordering.

## Failures and extension points

Context-limit recovery drops optional older summaries, then oldest whole history
pages, without consuming ordinary retries. Required current context is preserved.
Errors requiring user action stop the run. Missing credentials, rejected
authentication, permissions, quota, and output limits must remain distinguishable.

WebSocket setup may fall back to the same backend's HTTP path. Once delivery is
uncertain, the transport must not silently resend; feature owners control retries.
Only completed final answers reach parsing. Cancelled or replaced runs cannot
publish late results or errors, even when synchronous HTTP work cannot be
interrupted. Never log credentials; debug responses may contain project text.

Update contract builders and parsers together. Keep provider quirks in transports,
context policy in `context/` and the translator, and persistence in the project.
Verify the affected boundaries:

| Boundary | Focused tests |
| --- | --- |
| Profiles and persistence | `test_llm_profiles.py`, `test_proj_imgtrans_translation_context.py` |
| Contracts, context, and retries | `test_llm_translation_*.py`, `test_llm_translator.py`, `test_llm_chat.py` |
| Glossary | `test_translator_glossary.py` |
| Responses and authentication | `test_codex*.py`, `test_openai_responses.py` |
| Images, masks, and canvas results | `test_llm_inpaint.py`, `test_canvas_inpaint_lifecycle.py` |
| Profile and drawing selection | `test_llm_profile_widgets.py`, `test_module_selection_menu.py`, `test_drawing_inpainter.py` |

Run relevant suites with `python -m pytest`; use `QT_QPA_PLATFORM=offscreen` for Qt
checks and verify both PyQt5 and PyQt6 when changing UI or QObject lifetimes.
