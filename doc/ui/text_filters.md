# Text filters

Read [Text engine](text_engine.md) and [Text effects](text_effects.md) first.
Filters are a lazy plug-in boundary inside the text-effect renderer, not
application automation modules or a second rendering pipeline.

## Owners

| Contract | Owner |
| --- | --- |
| Persisted value and passive coercion | [`utils/text_effects.py`](../../ballontranslator/utils/text_effects.py) |
| AST discovery, metadata validation, and lazy import | [`effects/filters/registry.py`](../../ballontranslator/ui/text_engine/effects/filters/registry.py) |
| Built-in and custom implementations | `effects/filters/filter_*.py`, `custom_modules/filter_*.py` |
| Order, tile halos, strict failure, and caches | [`effects/renderer.py`](../../ballontranslator/ui/text_engine/effects/renderer.py) |
| Cards, preview, and undo | [`effects/panel.py`](../../ballontranslator/ui/text_engine/effects/panel.py), [`effects/edit_session.py`](../../ballontranslator/ui/text_engine/effects/edit_session.py) |

## Value and stack contract

`FilterEffect` is a frozen, hashable, repeatable stack value containing
`enabled`, a stable `filter_id`, positive `schema_version`, and flat JSON-scalar
params. Params are sorted internally for stable cache keys. Multi-selection
identity includes ID and schema.

A Filter receives the structural foreground and all movable layers accumulated
before its visible card. Cards below it run afterward. Consecutive Filters run
in panel order through one straight-RGBA bridge. Eraser, overall Opacity, global
transforms, and editing feedback remain downstream and are never plug-in input.
Ordinary Filters stay active during native horizontal and vertical editing.

Returned alpha may stay equal or shrink by default. A plug-in that declares
`expands_alpha: True` may grow it only within its validated `tile_halo`; that
reach also contributes to effect padding. Full and tiled rendering must be
byte-identical for the same absolute coordinates.

## Passive loading, removal, and schema changes

Project, config, and preset loading never imports filter code. Unknown IDs,
newer schemas, and opaque scalar params survive round trips unchanged. Active
resolution passes only declared params; a missing or invalid known value uses
the validated metadata default without rewriting the preserved payload.

Removing a `filter_*.py` file and restarting removes it from discovery. Its
saved value remains a **Missing Filter** card that can still be disabled,
reordered, or deleted. An enabled missing or incompatible Filter warns and is
bypassed interactively; strict export fails instead. A disabled missing Filter
is neutral. Restoring compatible code and restarting reactivates the saved card.

Removing a parameter within the same schema is safe: its saved scalar remains
opaque and is omitted at runtime, then disappears after an explicit edit writes
the current declared set. Rename or reinterpret a parameter only by incrementing
the schema and supplying `migrate_params`. Failed or absent migration preserves
saved data, bypasses interactively, and fails strict export.

## Discovery and trust

One plug-in lives in one `filter_*.py` file:

- built-ins: `ballontranslator/ui/text_engine/effects/filters/filter_*.py`
- local custom filters: `custom_modules/filter_*.py`

Each file exposes one literal `FILTER_META` mapping. Discovery reads it with
`ast.literal_eval` without importing the module, snapshots deterministic results
for the process, and requires restart for changes. Built-in metadata errors fail
loudly; malformed custom files are warned and isolated. Built-ins win duplicate
IDs. Symlinks, path/ID mismatches, and scan-to-import replacement are rejected.

Runtime import requires matching ID/schema and callable `apply` and `tile_halo`.
Optional `migrate_params(from_version, params)` runs only during active
resolution or explicit editing, never passive load. Custom filters are trusted
local Python, not sandboxed; they must avoid file/network IO, dependency or model
lifecycle work, global RNG, and unbounded allocation.

Built-in names and parameter choices use the extractable `TextEffectPanel`
translation context. Custom metadata is shown literally.

## Metadata and runtime API

Metadata defines `filter_id`, display `name`, positive `schema_version`,
deterministic `order`, ordered parameter specs, and optional `expands_alpha`.
Supported parameter kinds are `float`, `int`, `bool`, and `choice`. Numeric
specs include their default and range; choices contain flat JSON scalars.

```python
from typing import Mapping

import numpy as np

from ballontranslator.ui.text_engine.effects.filters.registry import FilterContext
from ballontranslator.utils.text_effects import FilterScalar

FILTER_META = {
    "filter_id": "custom:posterize",
    "name": "Posterize",
    "schema_version": 1,
    "order": 100,
    "params": ({
        "key": "levels", "label": "Levels", "kind": "int",
        "default": 4, "minimum": 2, "maximum": 16, "step": 1,
    },),
}


def tile_halo(params: Mapping[str, FilterScalar], render_scale: float) -> int:
    return 0


def apply(
    rgba: np.ndarray,
    params: Mapping[str, FilterScalar],
    context: FilterContext,
) -> np.ndarray:
    levels = int(params["levels"])
    step = 255.0 / (levels - 1)
    rgba[:, :, :3] = np.rint(rgba[:, :, :3] / step) * step
    return rgba
```

Save this as `custom_modules/filter_posterize.py`; the filename suffix must
match the final `filter_id` segment. Keep metadata entirely literal.

`apply` receives an owned contiguous straight RGBA8 array plus normalized
declared params and returns the same shape and dtype. Reusing the input array is
allowed. `FilterContext` supplies render scale, strict-export state, and the
absolute integer pixel origin relative to the unpadded logical origin. Derive
randomness only from seed and absolute coordinates. `tile_halo` returns the
smallest bounded nonnegative physical-pixel sampling reach at the active scale.

An exception, invalid output, missing implementation, incompatible schema,
invalid/excessive halo, or undeclared/out-of-halo alpha growth bypasses only that
Filter interactively and warns once. Strict export fails through the existing
effect error boundary.

## Rendering and caches

Each committed, preview, and export namespace keeps a bounded below-filter
prefix. A Filter-only preview reuses that prefix plus canonical glyph and
positioned-Stroke caches; it does not rerasterize text. Consecutive Filters cross
to straight RGBA once. An intervening Image or generated painter batch creates a
new bridge because the composition boundary is real, not duplicated work.

Tiled rendering adds cumulative retained halos to tile overlap. Planning and all
tiles share asset resolution, absolute origins, and the same filter order.
Reordering or toggling a Filter changes the prefix identity; parameter-only
edits retain reusable lower pixels. There are no per-Filter prefix caches,
workers, GPU paths, or second vector/text rasterizer.

## Built-ins

| Filter | Contract |
| --- | --- |
| Noise | Seeded coordinate-deterministic color or monochrome pigment noise; preserves alpha. |
| Grain | Seeded blurred pigment and inward-only alpha grain. |
| Gaussian Blur | Finite premultiplied RGBA blur with transparent borders and no edge fringe. |
| Bloom | Thresholded premultiplied highlight blur and additive bloom. |
| Glitch | Seeded row-block displacement and RGB split derived from absolute coordinates, adapted from the MIT-licensed [Godot shader](https://godotshaders.com/shader/glitch-effect-shader/). |

## Adding or changing a filter

1. Keep metadata literal, IDs/path suffixes stable, params scalar, and runtime
   code deterministic and bounded.
2. Declare the smallest halo covering every sampled neighbor; omit
   `expands_alpha` unless growth is essential and fully halo-bounded.
3. Verify contiguous straight RGBA8 output and full/tiled byte identity at
   nonzero and negative origins, including cumulative halos.
4. Test interactive bypass and strict-export failure for missing code,
   exceptions, incompatible schema, invalid output, and invalid halo.
5. Exercise card preview/cancel/one-undo, reorder/remove/eye, deferred deletion,
   and Eraser deactivation under PyQt5 and PyQt6 when binding-sensitive.
6. Run the focused domain, renderer, registry, and panel suites plus
   `py_compile` and `git diff --check`; finish UI changes with a themed-app pass.
