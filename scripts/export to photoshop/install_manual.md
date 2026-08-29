# BallonTranslator Photoshop Bridge

Advanced bidirectional bridge between **BallonsTranslator** and **Adobe Photoshop** (CC 2019 - CC 2026+).

---

## Features

1. **Automatic Multi-Layer PSD Assembly**:
   - Opens the raw manga/comic scan as the base layer (`[BT] Original Scan`).
   - Inserts clean inpainted plate (`inpainted/<page>`) as `[BT] Clean Inpaint`.
   - Inserts binary text mask (`mask/<page>`) as `[BT] Text Mask` (hidden).
   - Generates editable translated text layers in `[BT] Translations`.
   - Generates original OCR reference text layers in `[BT] Original OCR` (hidden).

2. **Typography & Styling (ActionManager Engine)**:
   - Font size, auto-leading, and alignment (Left, Center, Right).
   - Automatic outside outline/stroke effect (`Stroke FX`) with custom color and stroke width.
   - Text fill color (`RGB`).
   - Font resolution with fuzzy fallback matching PostScriptName and font families.
   - Paragraph box text wrapping and optional vertical centering within speech bubble bounds.

3. **Bidirectional Synchronization (Round-Trip Sync)**:
   - Edit or reformat text blocks directly in Photoshop.
   - Click **Send Changes to BallonsTranslator** to return validated changes through the open application.

4. **Batch Processing**:
   - Select one, several, or all pages in a chapter to import into layered Photoshop documents in a single click.

---

## Installation

### Method 1: Via BallonsTranslator UI (Recommended)
1. Open BallonsTranslator.
2. Go to **Tools -> Photoshop Bridge** (`Ctrl+Shift+P`).
3. Click **Install / Update Script in Photoshop**.
4. If Photoshop's installation directory is protected, BallonsTranslator opens the source and destination folders for a manual copy. Windows may request permission when you copy through Explorer.

### Method 2: Manual Installation
Copy `BallonTranslator_PS_Bridge.jsx` directly into:
- **Windows**: `C:\Program Files\Adobe\Adobe Photoshop [Version]\Presets\Scripts\`
- **macOS**: `/Applications/Adobe Photoshop [Version]/Presets/Scripts/`

Restart Photoshop (or run directly via `File -> Scripts -> Browse...` / `Ctrl+F12`).

---

## Usage

1. Translate and inpaint your project in **BallonsTranslator** and save the project.
2. Open **Adobe Photoshop**.
3. Go to **File -> Scripts -> BallonTranslator_PS_Bridge** (or `File -> Scripts -> Browse...` and select `BallonTranslator_PS_Bridge.jsx`).
4. Select your project file `imgtrans_*.json`.
5. In the dialog:
   - Choose pages to import.
   - Toggle clean plate, mask, translations, stroke FX, and vertical centering.
   - Click **Import Selected Pages**.
6. Polish text and art in Photoshop.
7. To push text updates back to BallonsTranslator, run the script and click **Send Changes to BallonsTranslator**.
