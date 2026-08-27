# Text filters

Read [Text engine](text_engine.md) and [Text effects](text_effects.md) first.
Filters are a small lazy plug-in boundary inside the existing text-effect
renderer, not application automation modules or a second pipeline.

## Owners

| Contract | Owner |
| --- | --- |
| persisted repeatable value and passive coercion | `utils/text_effects.py` |
| AST discovery, parameter specs, and lazy runtime import | `effects/filters/registry.py` |
| built-in and custom implementations | `effects/filters/filter_*.py`, `custom_modules/filter_*.py` |
| stack order, tile halos, strict failure, and caches | `effects/renderer.py` |
| cards, Add submenu, preview, and undo | `effects/panel.py`, `effects/edit_session.py` |

## Value and ordered-stack contract

`FilterEffect` is a frozen, hashable, repeatable `TextEffectStack` value. It
stores `enabled`, stable `filter_id`, positive `schema_version`, and flat JSON
scalar params. Params serialize as an object and are sorted internally for
stable cache keys. Mixed-selection identity includes ID and schema. Unknown
IDs, newer schemas, and opaque params survive passive project, config, and
preset round trips without importing plug-in code. At active resolution only
declared params are passed to the plug-in; a missing or invalid known value
uses its validated metadata default while the persisted value stays unchanged.

Stroke, Shadow, Glow, and Filter cards share one top-to-bottom application
order in the panel. A Filter transforms the fixed Text Fill/Image base
plus generated layers accumulated above its card; cards below it run afterward.
The persisted tuple remains topmost-first for compatibility, so the renderer
traverses that tuple in reverse. Consecutive Filters execute panel top-to-bottom
in one chain. Text Eraser, Overall
Opacity, the global transform, and selection/caret/IME remain downstream and
structurally fixed. Returned alpha may only stay equal or shrink unless static
metadata declares `expands_alpha: True`. A declared
expander may grow alpha only inside its validated tile halo; that cumulative
reach also expands the item's effect padding so output is not clipped.

A newly added movable card appears at the bottom of the panel and executes
last. Existing project and configuration tuples retain their exact stored and
rendered order; only their panel projection is reversed.

## Removal and parameter evolution

Filters have no central registration or direct imports. Removing a
`filter_*.py` file and restarting removes it from discovery; an implementation
already imported by the current process remains alive until restart. Project,
configuration, and preset loading never resolves filter code, so the saved ID,
schema, enabled state, and scalar params survive unchanged. The panel presents
an unavailable value as a **Missing Filter** card whose eye, reorder, and delete
actions remain usable.

An enabled missing filter is bypassed interactively with one warning. Strict
export fails instead of silently producing output without the requested effect;
a disabled missing filter is neutral. Restoring a compatible file and
restarting makes the preserved card active again.

Removing a parameter without changing the schema is backward compatible:
unknown scalar keys remain in passive saved data but are omitted from runtime
params. Invalid values for still-declared keys use metadata defaults and warn
once. An explicit parameter edit writes the current declared parameter set, so
obsolete keys naturally disappear. Rename or reinterpret a parameter only with
a schema-version increment and `migrate_params`; absent or failed migration
bypasses interactively and fails strict export without corrupting saved data.

Image is suppressed during native editing, but ordinary Filters stay
active in both writing modes. Qt feedback is painted afterward and is never
visible to plug-ins. Enabled requested filters remain strict-export eligible
even when the text is empty or the implementation cannot be resolved.

## Discovery and trust

One plug-in lives in one `filter_*.py` file:

- built-ins: `ballontranslator/ui/text_engine/effects/filters/filter_*.py`
- local custom filters: `custom_modules/filter_*.py`

Each file defines one literal `FILTER_META` mapping. The filter-only registry
reads that mapping with `ast.literal_eval`; discovery does not import or execute
the file. Metadata is snapshotted in deterministic order for the process.
Changes and additions require restart. Built-in metadata errors fail loudly;
malformed custom files are warned and isolated. Built-ins win duplicate IDs.
Custom symlinks, path/ID mismatches, and scan-to-import path replacement are
rejected. An active import must expose matching runtime ID/schema plus callable
`apply` and `tile_halo` functions. Optional `migrate_params(from_version,
params)` runs only while resolving an active older value or explicit edit,
never during passive loading.

Built-in names, parameter labels, and choice labels use the extractable
`TextEffectPanel` translation context. Trusted custom metadata is shown
literally so local plug-ins do not impersonate application catalog entries.

Custom plug-ins are trusted local Python, not sandboxed. Their runtime methods
must do no file or network IO, model/dependency lifecycle work, or unbounded
allocation/computation. Use only deterministic bounded array operations.

## Metadata and runtime API

Metadata contains `filter_id`, display `name`, positive `schema_version`,
deterministic `order`, ordered parameter mappings, and the optional boolean
`expands_alpha` capability (false by default). Supported parameter kinds are
`float`, `int`, `bool`, and `choice`. Numeric metadata supplies a default,
minimum, maximum, display factor, step, decimals, and optional suffix; choice
values are flat JSON scalars.

The runtime API is:

```python
from typing import Mapping

import numpy as np

from ballontranslator.ui.text_engine.effects.filters.registry import FilterContext
from ballontranslator.utils.text_effects import FilterScalar


FILTER_META = {
    'filter_id': 'custom:posterize',
    'name': 'Posterize',
    'schema_version': 1,
    'order': 100,
    'params': (
        {
            'key': 'levels', 'label': 'Levels', 'kind': 'int',
            'default': 4, 'minimum': 2, 'maximum': 16, 'step': 1,
        },
    ),
}


def tile_halo(
    params: Mapping[str, FilterScalar], render_scale: float,
) -> int:
    return 0


def apply(
    rgba: np.ndarray,
    params: Mapping[str, FilterScalar],
    context: FilterContext,
) -> np.ndarray:
    levels = int(params['levels'])
    step = 255.0 / (levels - 1)
    rgba[:, :, :3] = np.rint(rgba[:, :, :3] / step) * step
    return rgba
```

Save that exact shape as `custom_modules/filter_posterize.py`; the filename
suffix must match the final segment of `filter_id`. Keep `FILTER_META` entirely
literal so discovery remains AST-only. The registry supplies normalized
declared params, while the implementation owns bounded pixel work.

`apply` receives an owned contiguous straight RGBA8 array plus declared,
validated params. It returns the same shape and dtype; returning the same array
is allowed. `context` provides render scale, integer absolute pixel origin
relative to the unpadded logical origin, and strict-export state. Never derive
randomness from tile/surface dimensions or global RNG. `tile_halo` returns the
filter's nonnegative bounded physical-pixel sampling reach at the active scale.

An exception, invalid output, undeclared alpha expansion, expansion beyond the
declared filter's halo, missing implementation, schema incompatibility, or
invalid/excessive halo bypasses only that filter interactively and warns once.
Strict export fails through the existing effect raster error boundary. Disabled
missing filters are neutral. A cumulative halo that leaves no tile core bypasses
the affected filter interactively and fails strict export.

## Rendering and caches

The renderer caches the fixed base plus generated nodes below the bottom active
Filter in a two-entry `pre_filter_cache` in each existing committed, 0.5x
preview, and export namespace. Its key includes the bottom-Filter boundary and
canonical Stroke dependencies of cached exterior layers while excluding Filter
parameters. A filter-only preview reuses that prefix and canonical/positioned-
Stroke caches. The renderer alternates only the necessary contiguous generated
painter batches and Filter chains; consecutive Filters cross to straight RGBA
once, while a generated batch separating two Filter groups necessarily creates
two bridges. There are no per-filter prefix caches, workers, GPU paths, or
second vector/text rasterization.

Tiled rendering adds the sum of active filter halos to the existing effect
overlap. Absolute origins and bounded source overlap make full and tiled output
byte-identical across interleaved generated/Filter segments. Reordering or
toggling a Filter changes the below-filter prefix key; filter-only parameter
edits retain its reusable pixels.

## Built-ins

- **Noise** — Amount, Color/Monochrome, and Seed. Adds coordinate-deterministic
  pigment noise and preserves alpha. Halo is zero.
- **Grain** — Amount, Size, and Seed. Applies coordinate-deterministic blurred
  pigment and inward-only alpha grain without per-surface normalization.
- **Rough Edge** — Amount, Size, Hardness, and Seed. Reproduces the original
  coarse-plus-fine noisy threshold, grows a jagged silhouette within its bounded
  halo, and extends the nearest visible source color into newly covered pixels.
- **Gaussian Blur** — Radius in logical pixels. Blurs premultiplied float32 RGBA
  through an exact finite kernel with transparent borders, so translucent edges
  do not acquire dark or colored fringes. Radius zero is byte-for-byte neutral.
- **Bloom** — Threshold, Radius, and Intensity. Extracts visible highlights from
  the maximum sRGB channel, blurs their premultiplied color and coverage through
  the same finite kernel, then adds the result while keeping valid premultiplied
  alpha. At a 100% threshold only exact 255-channel highlights contribute;
  Intensity zero is byte-for-byte neutral, while Radius zero is a valid
  face-only bloom.
- **Glitch** — Shift, Block Size, Activity, RGB Split, and Seed. Applies a
  deterministic static horizontal displacement per absolute physical row block,
  then samples red and blue at opposing offsets from immutable straight RGBA8
  input and reconstructs premultiplied-safe output. Negative origins use floor
  division, so crops and bounded tiles match the full surface. It is a seeded
  adaptation of the time-driven block shift and channel split in the MIT-licensed
  [Godot glitch shader](https://godotshaders.com/shader/glitch-effect-shader/).

Deferred v1 scope includes screentone, a broader sharpen/blur gallery, further
distortions, per-filter masks/blend modes, generic mix/opacity, plug-in icons,
hot reload, sandboxing, workers, GPU paths, and external dependencies.

## Extension and verification checklist

- Keep metadata literal, IDs/path suffixes stable, params scalar, and runtime
  imports free of IO, downloads, models, and global RNG.
- Define the smallest nonnegative physical-pixel halo covering every sampled
  neighbor; derive randomness only from seed and absolute pixel coordinates.
- Verify the implementation returns contiguous straight RGBA8 of the same
  shape. Leave `expands_alpha` absent unless growth is essential; declared
  growth must fit entirely inside `tile_halo` at every render scale.
- Test full versus tiled output byte-for-byte at nonzero and negative origins,
  including cumulative halos when the new filter is chained.
- Test interactive bypass and strict-export failure for missing code,
  exceptions, invalid output, incompatible schema, and invalid halo.
- Exercise card preview/cancel/one-undo, reorder/remove/eye, deferred deletion,
  and Eraser-session deactivation under both PyQt5 and PyQt6 offscreen.
- Run the focused renderer/UI suites, `py_compile`, relevant doctests, and
  `git diff --check`; finish with a real themed-app visual pass when UI changes.
