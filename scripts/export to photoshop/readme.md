# Photoshop Bridge & Integration Guide for AI Agents

This document is the single source of truth for the **BallonsTranslator Photoshop Bridge** (`BallonTranslator_PS_Bridge.jsx`) and its integration tools in `scripts/export to photoshop/` and `ballontranslator/ui/ps_bridge_dialog.py`.

---

## 1. Architecture Overview

```
BallonsTranslator (Qt UI)
   │
   ├─ Tools Menu -> Photoshop Bridge (`Ctrl+Shift+P`)
   │    └─ PhotoshopBridgeDialog (ballontranslator/ui/ps_bridge_dialog.py)
   │         ├─ Status detector: Registry lookup & version/MD5 check
   │         ├─ One-click launcher: Launches photoshop.exe with -r script
   │         └─ Installer: Non-elevated JSX copy with manual fallback
   │
   └─ scripts/export to photoshop/
        ├─ BallonTranslator_PS_Bridge.jsx (Standalone ExtendScript v2.5+)
        └─ install_manual.md (User guide)
```

---

## 2. ExtendScript & Photoshop DOM Gotchas (Critical Invariants)

When editing or extending `BallonTranslator_PS_Bridge.jsx`, you MUST adhere to the following rules:

### A. Newlines in Text Layers (`\r` vs `\n`)
- **CRITICAL:** Photoshop ExtendScript `TextItem.contents` requires **`\r` (Carriage Return `0x0D`)** for line breaks.
- Passing `\n` (Line Feed `0x0A`) causes Photoshop on Windows to render missing-glyph boxes `▯` (*tofu*) and keep the entire text on a single line.
- **Rule:** Always normalize text before assigning to `contents`:
  ```javascript
  var psText = transText.replace(/\r\n/g, "\r").replace(/\n/g, "\r");
  tItem.contents = psText;
  ```

### B. Leading / Line Spacing (`useAutoLeading`)
- BallonsTranslator stores `line_spacing` as a dimensionless multiplier (e.g. `1.0`, `1.15`, or `0`).
- Feeding `fmt.line_spacing` directly as pixels into `tItem.leading` collapses multiline text into a 1px overlapping pile.
- **Rule:** Use `tItem.useAutoLeading = true;` unless `fmt.line_spacing > 5` explicit pixel value is provided.

### C. Color Profile & Missing Profile Dialogs
- Manga scans and web images often lack ICC profiles (or use sRGB), while Photoshop may be set to Adobe RGB (1998).
- Calling `app.open(file)` can trigger a blocking modal dialog (*"Embedded Profile Mismatch"*).
- **Rule:** Open files silently via ActionManager with `DialogModes.NO`:
  ```javascript
  function openSilent(fileObj) {
      try {
          var desc = new ActionDescriptor();
          desc.putPath(stringIDToTypeID("null"), fileObj);
          executeAction(stringIDToTypeID("open"), desc, DialogModes.NO);
          return app.activeDocument;
      } catch (e) {
          return app.open(fileObj);
      }
  }
  ```

### D. User Cancellation Handling
- When a user closes a dialog, presses Escape, or cancels an action, ExtendScript throws error numbers `8007` or `-128` ("User cancelled").
- **Rule:** Catch and filter user cancellation silently so Photoshop does not display scary debugger error boxes:
  ```javascript
  function isUserCancelled(e) {
      if (!e) return false;
      if (e.number === 8007 || e.number === -128) return true;
      var msg = (e.message || String(e)).toLowerCase();
      return msg.indexOf("cancel") !== -1 || msg.indexOf("отмен") !== -1;
  }
  ```

---

## 3. ActionManager Descriptors (Layer Effects / Stroke FX)

Photoshop DOM does not have a direct `artLayer.stroke` property. Strokes (white manga outlines) are created via the low-level ActionManager `frameFX` descriptor:

```javascript
function applyStrokeFX(layer, strokeWidthPx, r, g, b, opacity, position) {
    if (!strokeWidthPx || strokeWidthPx <= 0) return;
    try {
        app.activeDocument.activeLayer = layer;
        var s2t = stringIDToTypeID;
        var posKey = (position === "inside") ? "InsF" : (position === "center" ? "CtrF" : "OutF");

        var desc = new ActionDescriptor();
        var ref = new ActionReference();
        ref.putProperty(s2t("property"), s2t("layerEffects"));
        ref.putEnumerated(s2t("layer"), s2t("ordinal"), s2t("targetEnum"));
        desc.putReference(s2t("null"), ref);

        var strokeDesc = new ActionDescriptor();
        strokeDesc.putBoolean(s2t("enabled"), true);
        strokeDesc.putBoolean(s2t("present"), true);
        strokeDesc.putBoolean(s2t("showInDialog"), true);
        strokeDesc.putEnumerated(s2t("style"), s2t("frameStyle"), s2t(posKey));
        strokeDesc.putEnumerated(s2t("paintType"), s2t("fillFrameStyle"), s2t("solidColorLayer"));

        var colorDesc = new ActionDescriptor();
        colorDesc.putDouble(s2t("red"), (r !== undefined) ? r : 255);
        colorDesc.putDouble(s2t("grain"), (g !== undefined) ? g : 255);
        colorDesc.putDouble(s2t("blue"), (b !== undefined) ? b : 255);
        strokeDesc.putObject(s2t("color"), s2t("RGBColor"), colorDesc);

        strokeDesc.putUnitDouble(s2t("size"), s2t("pixelsUnit"), strokeWidthPx);
        strokeDesc.putUnitDouble(s2t("opacity"), s2t("percentUnit"), (opacity !== undefined) ? opacity : 100);

        var lefxDesc = new ActionDescriptor();
        lefxDesc.putObject(s2t("frameFX"), s2t("frameFX"), strokeDesc);
        desc.putObject(s2t("to"), s2t("layerEffects"), lefxDesc);

        executeAction(s2t("set"), desc, DialogModes.NO);
    } catch (e) {}
}
```

---

## 4. Multilingual i18n & Unicode Escapes

ExtendScript on Windows evaluates `.jsx` source files in the local ANSI codepage (CP1251, CP1252, etc.).
- **Rule:** Never put raw non-ASCII Cyrillic or CJK characters directly in ExtendScript string literals.
- **Rule:** Encode all non-ASCII strings using standard 4-digit Unicode escape sequences (`\u0420\u0443\u0441\u0441\u043a\u0438\u0439` for "Русский").
- The `BT_I18N` table automatically detects `app.locale` and provides instant in-dialog translation switching between English, Russian, and Chinese.

---

## 5. Robust 4-Property Font Matching Engine

Photoshop `app.fonts` objects expose 4 properties:
- `font.family`: Font family name (e.g. `CC Rumble`, `v_CCRumble`, `Arial`)
- `font.name`: Full display name (e.g. `CC Rumble Regular`, `Arial Bold`)
- `font.style`: Subfamily style (e.g. `Regular`, `Bold`, `Italic`, `Bold Italic`)
- `font.postScriptName`: System internal PostScript identifier (e.g. `CCRumble-Regular`, `Arial-BoldMT`)

**Resolution Pipeline (`BT_PS.resolveFontPostScript`):**
1. **Normalization:** Strips spaces, hyphens, underscores, and common vertical/vendor prefixes (`v_`, `@`).
2. **Level 1 (Exact Match):** Direct lookup by sanitized `postScriptName` or `name`.
3. **Level 2 (Family & Weight Match):** Matches `family`, then selects weight candidate matching requested `isBold` / `isItalic` flags, with fallback to `Regular`.
4. **Level 3 (Fuzzy Substring):** Substring search across all installed font names.
5. **Safe Fallback:** If absent, text layout coordinates and sizes are 100% preserved with default font fallback, and missing fonts are summarized upon completion.

---

## 6. Geometry, Centering & Bounding Boxes

- BallonsTranslator stores block geometries in `_bounding_rect = [x, y, width, height]` and `xyxy = [x1, y1, x2, y2]`.
- **DPI / Resolution Scaling (`scaleFactor`):**
  Photoshop ExtendScript `TextItem.size` internally treats points based on `doc.resolution` ($\text{Rendered Pixels} = \text{Points} \times \frac{\text{doc.resolution}}{72}$).
  On high-resolution prints (e.g. 293 DPI / 300 DPI / 600 DPI), setting unscaled pixels blows up font size by $\frac{\text{DPI}}{72}$.
  **Rule:** Always scale font size and manual leading by `scaleFactor = 72.0 / doc.resolution`:
  ```javascript
  var docRes = (doc.resolution && doc.resolution > 0) ? doc.resolution : 72;
  var scaleFactor = 72.0 / docRes;
  tItem.size = new UnitValue(fontSize * scaleFactor, "pt");
  ```
  And when syncing back in `Save PSD to JSON`:
  ```javascript
  var sizeInPx = Math.round(tLayer.textItem.size.as("pt") * (activeDocRes / 72.0));
  ```
- **Vertical Centering Math:**
  When `chkVCenter` is enabled, the script calculates:
  $$\text{centerBoxY} = \text{posY} + \frac{\text{targetH}}{2}$$
  $$\text{centerTextY} = \text{actualTop} + \frac{\text{actualH}}{2}$$
  $$\Delta Y = \text{centerBoxY} - \text{centerTextY}$$
  and calls `tLayer.translate(0, deltaY)`.
- It then trims `tItem.height = actualH + padding` so the paragraph frame tightly hugs the rendered text without dangling tails.

---

## 6. Official Adobe API References & Documentation

1. **Adobe Photoshop JavaScript Scripting Reference:**
   - [Adobe Photoshop Scripting Documentation (GitHub)](https://github.com/Adobe-CEP/Photoshop-Scripts)
   - [ExtendScript Toolkit Reference Manual](https://extendscript.docsforadobe.dev/)
2. **ActionManager (ActionDescriptor / ActionReference):**
   - [ActionManager Primer (Clean-SVG / Photoshop Scripting Guide)](https://github.com/r-freivald/photoshop-scripting-docs)
3. **Adobe UXP Documentation (for future UXP panel development):**
   - [Photoshop UXP API Reference](https://developer.adobe.com/photoshop/uxp/2022/ps_reference/)
