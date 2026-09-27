[简体中文](加别的翻译器.md) | English | [Русский](add_translator_ru.md)

# Add a translator

Implement a registered `BaseTranslator` in `custom_modules/trans_<name>.py`
for a local extension, or `ballontranslator/modules/translators/trans_<name>.py`
for a built-in module. Restart after adding the file. Discovery reads metadata
without importing the module; do not add an eager import to `__init__.py`.

## Minimal module

Save this as `custom_modules/trans_example.py`. It copies source text so discovery,
language selection, and result mapping can be checked without an API or model.

```python
from typing import List

from ballontranslator.modules.translators.base import BaseTranslator, register_translator


@register_translator("example_copy")
class ExampleTranslator(BaseTranslator):
    concate_text = False
    params = {"description": "Copy source text without a translation service."}

    def _setup_translator(self) -> None:
        self.lang_map["日本語"] = "ja"
        self.lang_map["English"] = "en"

    def _translate(self, src_list: List[str]) -> List[str]:
        return list(src_list)
```

The registry key is persisted in config; choose a unique, stable name. Language
keys come from `LANGMAP_GLOBAL` in
[`base.py`](../ballontranslator/modules/translators/base.py); values are the
service's language codes. Runtime calls can read them through
`self.lang_map[self.lang_source]` and `self.lang_map[self.lang_target]`.

## Extension contracts

- `_translate` receives a list and must return the same number of strings in the
  same order. Keep `concate_text=False` for list-aware APIs and local models.
  `True` lets the base class join page input and split the result; use it only
  when the service preserves the separators. The public `translate()` handles
  strings, empty input, model loading, and optional project context.
- Define config fields in class-level `params`. A string supplies a text field;
  a selector uses `{"type": "selector", "options": [...], "value": ...}`.
  Read values with `get_param_value()`. Override `updateParam()` only for actual
  runtime updates, calling the base implementation first.
- Keep metadata statically readable by
  [`lazy_registry.py`](../ballontranslator/modules/lazy_registry.py): literal
  params and language assignments, or supported pure helpers. Asymmetric language
  support can use `supported_src_list`/`supported_tgt_list` properties returning
  literal lists. Settings discovery must not run constructors, setup, model
  loading, or network requests.
- Heavy model state belongs in `BaseModule`'s `_load_model_keys`/`_load_model`
  lifecycle. Preserve worker-owned execution and headless behavior; follow
  [repository dependency and data rules](../AGENTS.md#changes-and-data-safety).
- Prefer overriding `_translate` over the public pipeline methods. If an
  integration needs shared LLM context, use the ownership boundaries in the
  [LLM guide](modules/llm_translator.md).

## Verification

From the repository root, using the app's Python environment:

```bash
python -m py_compile custom_modules/trans_example.py
python -c 'from ballontranslator.modules import TRANSLATORS; t = TRANSLATORS.get("example_copy")("日本語", "English"); assert t.translate(["one", "two"]) == ["one", "two"]; assert t.translate("one") == "one"'
```

For a real translator, check language metadata and params before initialization,
string/list/empty input, output count/order, and service/model failure behavior.
Use mocked transports for automated tests. Follow
[repository verification](../AGENTS.md#verification), and confirm settings can be
opened without loading a model or contacting the service.
