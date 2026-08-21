# BallonTranslator ↔ Photoshop Bridge

Advanced bidirectional bridge between **BallonsTranslator** and **Adobe Photoshop** (CC 2019 – CC 2025+).

---

## ⚡ Features / Возможности

1. **Automatic Multi-Layer PSD Assembly**:
   - Opens the raw manga/comic scan as the base layer (`[BT] Original Scan`).
   - Inserts clean inpainted plate (`inpainted/<page>`) as `[BT] Clean Inpaint`.
   - Inserts binary text mask (`mask/<page>`) as `[BT] Text Mask` (hidden).
   - Generates editable translated text layers in `[BT] Translations`.
   - Generates original OCR reference text layers in `[BT] Original OCR` (hidden).

2. **Typography & Styling (ActionManager Engine)**:
   - Pixel-perfect Font Size, Line Spacing (Leading), Letter Spacing (Tracking), Alignment (Left, Center, Right).
   - Automatic Outside Outline/Stroke effect (`Stroke FX`) matching project color and stroke width.
   - Text Fill color (`RGB`).
   - Font matching with fallback: matches PostScriptName and font families without runtime exceptions.
   - Paragraph box text wrapping within speech bubble bounds.

3. **Bidirectional Synchronization (Round-Trip Sync)**:
   - Edit, reformat, or reposition text blocks directly in Photoshop.
   - Click `💾 Save PSD back to JSON` to export typography changes back to `imgtrans_*.json` in real time.

4. **Batch Processing**:
   - Select one, several, or all pages in a chapter to import into layered Photoshop documents in a single click.

---

## 📥 Installation / Установка

### Method 1: Automatic Installer (Recommended)
Just double-click **`install_ps_script.bat`** (or right-click `install_ps_script.ps1` -> *Run with PowerShell*).
It will request Admin elevation and automatically install `BallonTranslator_PS_Bridge.jsx` into `Presets/Scripts`.

### Method 2: Manual Installation
Copy `BallonTranslator_PS_Bridge.jsx` directly into:
- **Windows**: `C:\Program Files\Adobe\Adobe Photoshop [Version]\Presets\Scripts\`
- **macOS**: `/Applications/Adobe Photoshop [Version]/Presets/Scripts/`

Restart Photoshop or run directly via `File -> Scripts -> Browse...` (Ctrl+F12).

---

## 🚀 Usage / Использование

1. Translate & Inpaint your chapter in **BallonsTranslator** and save the project.
2. Open **Adobe Photoshop**.
3. Go to **File -> Scripts -> BallonTranslator_PS_Bridge** (or `File -> Scripts -> Browse...` and select `BallonTranslator_PS_Bridge.jsx`).
4. Select your project file `imgtrans_*.json`.
5. In the dialog:
   - Choose pages to import.
   - Toggle inpaint, mask, translation, stroke FX, and paragraph box options.
   - Click **🚀 Import Selected Pages**.
6. Polish text/art in Photoshop.
7. To push changes back to BallonsTranslator, reopen the script and click **💾 Save PSD back to JSON**.
