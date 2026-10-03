# LLMTranslator

Maintainer guide to LLM translation and the shared profile/request boundaries.
Implementation details and edge cases belong in the owning code and focused tests.

## Architecture

| Concern | Owner |
| --- | --- |
| Translation prompts, message order, response schema, and parsing | [`llm_translation_contract.py`](../../ballontranslator/modules/translators/llm_translation_contract.py) |
| Request snapshots, retries, history orchestration, and compaction | [`trans_llm.py`](../../ballontranslator/modules/translators/trans_llm.py) |
| API clients, throttling, provider compatibility, and completion normalization | [`llm_chat.py`](../../ballontranslator/modules/llm_chat.py) |
| Profile loading, defaults, and model choices | [`llm_profiles.py`](../../ballontranslator/utils/llm_profiles.py), [`config.py`](../../ballontranslator/utils/config.py) |
| Codex authentication, credential storage, and subscription transport | [`codex.py`](../../ballontranslator/modules/codex.py) |
| Image dispatch and shared generation protocol | [`llm_image.py`](../../ballontranslator/modules/llm_image.py), [`image_generation.py`](../../ballontranslator/modules/image_generation.py) |
| Image encoding | [`llm_vision.py`](../../ballontranslator/modules/llm_vision.py) |
| History selection, saved-context packing, glossary parsing, and token estimates | [`context/`](../../ballontranslator/modules/context) |
| Text-block preprocessing, finalization, and page coverage | [`base.py`](../../ballontranslator/modules/translators/base.py) |
| Worker lifecycle and canvas request validity | [`module_manager.py`](../../ballontranslator/ui/module_manager.py) |
| Project completion, summaries, memory, and load identity | [`proj_imgtrans.py`](../../ballontranslator/utils/proj_imgtrans.py) |
| User-owned context editing | [`llm_context_editor.py`](../../ballontranslator/ui/llm_context_editor.py) |

GUI and headless translation use the same lifecycle: preprocess source blocks,
freeze request inputs, assemble and validate the response, finalize translations,
then mark page completion and persist pending context updates. `ProjImgTrans` is
authoritative; request snapshots and the reusable history window are disposable.

## Profiles and backends

`backend="openai"` selects the API-key transport, including compatible gateways;
the profile URLs determine the service. `backend="codex"` selects the app's
ChatGPT subscription account. Neither transport falls back to the other's billing.
Feature owners define prompts and response contracts; transports handle service
compatibility and authentication.

Codex has one canonical profile with ID `codex`, separate from editable API
profiles and clipboard operations. Its saved public model catalog is
`module.codex_models`; offline defaults and saved custom choices remain usable
without a catalog. Profile selection and settings construction perform no network
or credential IO. [`codex_account.py`](../../ballontranslator/ui/codex_account.py)
owns asynchronous GUI account operations, while headless requests restore
credentials on demand. [`codex_settings.py`](../../ballontranslator/ui/codex_settings.py)
owns the dedicated settings panel.

Catalog and generation requests share a Codex client identity resolved from
OpenAI's latest stable GitHub release. The first request and subsequent requests
after one hour refresh it in the calling worker, including headless operation;
no CLI installation is needed. Release lookup uses the request's proxy, sends no
account credentials, and times out after five seconds. Failures retain the last
known version (initially `0.159.0`) and retry after the same one-hour interval.
The catalog query, `version` header, and versioned user agent stay aligned across
HTTP and WebSocket connections. This follows stable client compatibility gates;
it does not change the selected model or guarantee future protocol compatibility.

Credentials remain outside config/profile exports. Codex encrypts its credential
file using a key in the native credential store; unavailable secure storage uses
logged, reversible obfuscation on save. Reads preserve unreadable encrypted data
without replacing keys or downgrading protection. Token rotation is serialized,
and a failed save must succeed before the rotated credentials are reused.

Codex requests are stateless, with no hidden conversation or agent loop. Missing
sign-in and invalid authentication stop the run; authentication rejection permits
one renewal/replay, while permission and quota failures remain separate. Signing
in does not replay interrupted work. Account state and model-catalog refresh are
independent, so catalog failures cannot undo a committed account change.

## Image editing

[`LLMInpaint`](../../ballontranslator/modules/inpaint/inpaint_llm.py) owns crop
preparation and compositing; `LLMImageRequester` owns dispatch, throttling, and
retries. Direct image models use their provider route. A reasoning/image model
pair uses Responses with one forced image-generation call. Profile helpers derive
these choices without network access; only the selected pair is saved, while
image option lists retain base image IDs. Requests use the selected image service,
credentials, proxy, and model rather than translation/OCR settings.

Source and mask references must stay aligned through resizing and padding. Local
compositing enforces the requested mask boundary; empty masks need no request.
Provider errors must retain their classification: missing models, permission
failures, and invalid endpoints must not trigger API-key recovery.

Drawing selections are independent of Run. Canvas requests snapshot the selected
module, profile, prompt, and mask mode when submitted, and share the inpainter
only while its worker is idle. Page reloads or edits to the request crop invalidate
queued work and late results; unrelated edits outside that crop do not. Unmasked
rectangle editing intentionally applies the full returned crop. Text-effect
integration is owned by the [text effects guide](../ui/text_effects.md).

## Request contract

Each non-empty source block receives a one-based ID. Non-GPT API models return a
numeric translation map; recognized GPT API models and all Codex models return an
array of integer IDs and string translations. Contract selection happens before
the request, and prompts, schemas, history examples, and parsing must agree.
Strict schemas stay independent of the current block count. Accepted translations
must cover exactly `1..N` once each; array IDs and translations are not coerced.
Compatibility response shapes belong only in `parse_translation_response()`.

When requested, `page_summary` precedes translations in the generated contract;
parsing accepts either field order. A malformed summary does not discard valid
translations. Full-page Vision/Summary calls can summarize pages without source
text, in which case only a usable summary is required.

Messages keep stable context before current-page material:

```text
system: translation contract + profile instructions
system: complete glossary, then compact memory       # when enabled
user/assistant: completed-page example pairs           # +history
user: saved summaries + current input + matching glossary + image
```

The contract owns language, IDs, count, and response shape; profile instructions
control style and wording. Vision uses the Translator model and only the current
page image. It must preserve ID-to-block mapping rather than reorder project
blocks. Ordinary retries reuse the frozen messages, context, and encoded image.

## History and project context

`page` mode sends no prior-page examples. `+history` adds chronological,
glossary-free user/assistant pairs with IDs local to each page. A prior page must
precede the current page, be marked translated, and have translations for every
source-bearing block. Explicit target-language metadata must match; missing
metadata remains compatible with older projects. Textless pages can contribute
saved summaries when Summary is enabled. Project filenames are not sent.

History pairs are immutable and indivisible. The window grows only across
contiguous successful requests with unchanged project identity and prompt-shaping
settings; page jumps, reloads, source or summary edits, or setting changes rebuild
the window. Full-page calls and selections covering every source-bearing block may
advance the window after valid parsing. Partial selections may read history but
cannot advance it or save generated summaries. Page completion follows successful
postprocessing and assignment; full-page retries clear prior completion first.
`+history` requests remain sequential.

For every LLM backend, pages completed in the active run supply their original
response translations to history rendering and validation. Postprocessing,
layout line breaks, and translation edits do not rewrite those examples during
that run. The existing pipeline stop event identifies the run; original responses
remain only in the bounded history window. Other pages and new runs use saved
translations. Project eligibility, source text, target language, and saved
summaries are still validated.

Summary is independent of history mode. Saved summaries through the current page
can guide translation even for incomplete pages. Existing current summaries are
retained unless overwrite is enabled; a generated replacement stays pending until
page completion. Summaries represented by history or compact memory are omitted
from the request without deleting their project records.

Compact memory is a separate project record rendered before history. Automatic
compaction uses the Translator model to merge existing memory with uncovered
summaries before budget eviction and after the final project page. Coverage
metadata prevents repeated compaction and is not sent to the model. Successful
pre-translation compaction is persisted before assembling the translation request;
exhausted failure stops the run.

Summary and memory writes compare their request-start records before committing.
User edits or clears made in flight win. The two records are independently
user-owned: editing one does not silently regenerate or invalidate the other.

## Context budget and caching

The configured budget covers saved page summaries and bilingual history. The
current summary is required; older summaries and whole history pages use the
remaining allowance. Eviction leaves room for subsequent pages rather than
shifting the prefix on every request. Compact memory, current input, glossary,
image, system instructions, and output are outside this budget; the provider's
context limit still applies to the complete request.

Provider cache reuse is an optimization, never a correctness condition. Stable
messages precede volatile input, and retained history pairs keep their rendering.
Eviction, memory changes, or response-contract changes can break prefix reuse.
API cache policy belongs to the contract/requester; Codex uses a stable job cache
identity that survives retries and token renewal but changes with a new job or
account. Codex Responses sends that identity in `prompt_cache_key`, `session-id`,
`thread-id`, and `x-client-request-id`, matching the
[official Codex client](https://github.com/openai/codex/blob/rust-v0.155.1/codex-rs/codex-api/src/endpoint/responses.rs).
This subscription convention differs from the public API's per-request ID.
Codex chat requests use WebSocket by default. The translator and OCR module
parameter `codex websocket` can disable it in favor of HTTP/SSE; existing saved
choices remain authoritative. Successful continuation does not establish better
prompt-cache reuse.
For a process-wide test, `BALLOONTRANS_CODEX_WEBSOCKET=1` selects WebSocket for
Codex chat requests (translation, OCR, and memory compaction), overriding the
module checkboxes without rewriting saved settings. `0` selects SSE; when unset,
the existing module parameters apply. Image endpoints remain HTTP. Setup failures
can still fall back to SSE; debug logs identify the transport actually used.
The requester owns the connection, isolated by profile/model/account generation
and proxy. Transport switches preserve the job identity and close the old
connection. A session-owned thread keeps its event loop running between page
workers so idle sockets can answer server heartbeat pings;
new runs and obsolete account generations close their previous sessions. Idle
connections expire after five minutes, and connections older than 55 minutes
reconnect without changing the job's cache identity.

The job session retains only the official Codex infrastructure-cookie allowlist,
including the `__oailb` routing cookie. HTTP responses and WebSocket handshakes
update the same in-memory jar, so later SSE calls, reconnects, and transport
fallbacks retain routing cookies even when their HTTP client is replaced.
Cookie domain, path, expiry, deletion, and Secure rules use HTTPX's standard jar.
Account/session cookies are excluded; nothing is written to disk. New jobs,
account changes, proxy changes, and closed sessions discard the jar. Separate
catalog, OAuth, and release-discovery requests do not receive these cookies.
SSE diagnostics report only whether a routing-cookie header was sent. Cookie
retention supports routing affinity but does not establish cache-hit causality.
Responses requests also send the official `x-codex-routing-hint: model=...`
header on SSE calls and WebSocket handshakes, including assisted image requests.
Catalog, OAuth, and direct-image endpoints do not receive that Responses hint.
WebSocket error envelopes preserve `status`/`status_code` so HTTP-equivalent
authentication rejection gets one token renewal and permissions remain actionable.

Routing state has a shorter lifetime than the job or socket. Each translation,
OCR request, and memory compaction owns one `CodexTurnState` across its retries.
The first `x-codex-turn-state` from response/handshake headers or a
`response.metadata` event is replayed unchanged in SSE request headers or
WebSocket `client_metadata` (and reconnect headers). A new page/request starts
fresh even when reusing the socket; job/account changes clear the token too.
This follows the [official turn-state contract](https://github.com/openai/codex/blob/5fa5aaf0fffd6593ca60ed974af3e87e29f3be28/codex-rs/core/src/client.rs#L279).
Routing metadata is excluded from continuation comparison and history budgets.
Header lookup tolerates unrelated repeated headers. Debug logs record routing
fingerprints and exception types without exposing routing tokens or transport
exception messages.

Following [Pi's continuation contract](https://github.com/earendil-works/pi/blob/6f1072cc081f06b86a673bd142f03720d17afe15/packages/ai/src/api/openai-codex-responses.ts#L1425),
the transport sends `previous_response_id` and only appended input when settings
match and the full input starts with the preceding input plus returned output.
Eviction, changed history, or changed settings send full input on the same socket.
Missing continuation state gets one reconnect with full input. Setup failures
fall back to SSE for the run; a started stream is never automatically replayed
through SSE. Server validation and authentication failures retain their existing
handling. WebSocket support requires Python 3.9+ and the core `websockets>=15`
dependency; older environments use SSE. SOCKS proxies without the optional
`python-socks` transport also use the existing HTTPX/SSE path.

WebSocket requests use `OpenAI-Beta: responses_websockets=2026-02-06`; SSE uses
`OpenAI-Beta: responses=experimental`. Completed Codex pages
retain assistant output metadata and encrypted reasoning only in the runtime
history window. Chat requests omit `reasoning.context`, leaving reasoning-history
behavior to the backend default. The response's effective `reasoning.context` is
logged for verification; the selected reasoning effort is still sent when explicit.
Replay requires the same job/account/model and matching eligible
page sources and summaries; the transport rechecks ownership when sending.
Account changes fall back to the run's text messages without rebuilding history.
History rebuilds can reuse eligible responses from the same run. The existing budget
counts canonical text history, excluding the extra response items.
Do not send API-only cache controls to the subscription transport.
Missing provider cache statistics do not establish a cache miss.

## Glossary

[`glossary.py`](../../ballontranslator/modules/context/glossary.py) owns UTF-8
JSON/TSV/TXT parsing and deterministic selection. `Matching` uses case-insensitive
literal matches against current sources; `All` supplies a stable system message.
Entries retain file order, exact duplicates are removed, and conflicting targets
are rejected. An empty path disables the glossary; a configured unreadable or
malformed file fails explicitly. See the
[parser tests](../../tests/test_translator_glossary.py) for format examples.

## Failure behavior and verification

Context-limit recovery removes optional older summaries, then oldest whole history
pages, without spending ordinary retries. It never sacrifices current input,
current summary, memory, glossary, or image. Errors requiring user action bypass
retries and stop the run. Only completed provider responses reach
translation parsing.

Cancellation prevents subsequent attempts and interrupts waits. Synchronous
OpenAI-compatible chat calls cannot be interrupted in flight; Codex HTTP calls
can. Workers must reject obsolete results regardless of transport cancellation.
Diagnostics belong to their owning layer; provider usage and request fingerprints
are evidence, not proof of cache availability. Never log credentials, and treat
debug response content as potentially containing project or glossary text.

When extending a contract, update its builder and parser together. Keep provider
quirks in transports, context policy in `context/` and the translator, and saved
state in the project. Verify the boundaries affected by a change:

| Boundary | Focused tests |
| --- | --- |
| Profiles, defaults, and persistence | `test_llm_profiles.py`, `test_proj_imgtrans_translation_context.py` |
| Response contracts, context, and retries | `test_llm_translation_*.py`, `test_llm_translator.py`, `test_llm_chat.py` |
| Glossary parsing and selection | `test_translator_glossary.py` |
| Authentication and subscription requests | `test_codex*.py` |
| Image transport, masks, and obsolete canvas results | `test_llm_inpaint.py`, `test_canvas_inpaint_lifecycle.py` |
| Profile and drawing selection | `test_llm_profile_widgets.py`, `test_module_selection_menu.py`, `test_drawing_inpainter.py` |

Run relevant suites with `python -m pytest`; use `QT_QPA_PLATFORM=offscreen` for Qt
checks and verify both PyQt5 and PyQt6 when changing UI or QObject lifetimes.
