# BallonsTranslator API Documentation

This API allows external applications, such as a Photoshop UXP plugin, to leverage BallonsTranslator's core engines for OCR, Translation, and Inpainting.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run the API server: `python launch.py --api` (or `python app_api.py`)
3. Default address: `http://localhost:5000`

## Endpoints

### 1. List Available Modules
- **URL:** `/modules`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "textdetectors": ["ctd", ...],
    "ocr": ["mit48px", "none_ocr", ...],
    "translators": ["google", "chatgpt", ...],
    "inpainters": ["lama_large_512px", "patchmatch", ...]
  }
  ```

### 2. OCR (Detection + Recognition)
- **URL:** `/ocr`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
    "image": "<base64_encoded_image>",
    "detector": "ctd",
    "ocr": "mit48px",
    "lang_source": "日本語"
  }
  ```
- **Response:**
  ```json
  {
    "blocks": [
      {
        "text": "こんにちは",
        "box": [x1, y1, x2, y2],
        "lines_xyxy": [[...], [...]]
      }
    ],
    "mask": "<base64_encoded_mask_image>"
  }
  ```

### 3. Translation
- **URL:** `/translate`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
    "queries": ["こんにちは", "元気ですか？"],
    "translator": "google",
    "lang_source": "日本語",
    "lang_target": "English"
  }
  ```
- **Response:**
  ```json
  {
    "translations": ["Hello", "How are you?"]
  }
  ```

### 4. Inpainting (Text Removal)
- **URL:** `/inpaint`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
    "image": "<base64_encoded_image>",
    "mask": "<base64_encoded_mask>",
    "inpainter": "lama_large_512px"
  }
  ```
- **Response:**
  ```json
  {
    "image": "<base64_encoded_inpainted_image>"
  }
  ```

## Photoshop UXP Integration Guide

To connect a Photoshop UXP plugin:
1. Capture the selection in Photoshop.
2. Use Photoshop's Imaging API to get the pixel data as a Base64 string or binary buffer.
3. Send the data to the corresponding endpoint.
4. For Inpainting:
   - Create a mask based on the selection.
   - Send the image and mask to `/inpaint`.
   - Receive the processed Base64 image and place it as a new layer in Photoshop.
5. Example UXP `fetch` call:
   ```javascript
   const response = await fetch('http://localhost:5000/inpaint', {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({
       image: base64Image,
       mask: base64Mask,
       inpainter: 'lama_large_512px'
     })
   });
   const data = await response.json();
   // Use data.image to create a new layer
   ```
