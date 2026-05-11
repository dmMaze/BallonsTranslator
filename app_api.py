import os
import io
import base64
import numpy as np
import cv2
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

import utils.shared as shared
shared.HEADLESS = True

from utils.config import load_config
load_config() # Load default config

from modules.base import init_module_registries
from modules.ocr import OCR
from modules.translators import TRANSLATORS
from modules.inpaint import INPAINTERS
from modules.textdetector import TEXTDETECTORS
from utils.textblock import TextBlock
from utils.proj_imgtrans import ProjImgTrans

# Global registry initialization
init_module_registries()

app = FastAPI(title="BallonsTranslator API")

# Helper to decode base64 image
def decode_image(b64_str: str) -> np.ndarray:
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        data = base64.b64decode(b64_str)
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to decode image")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

# Helper to encode image to base64
def encode_image(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

class OCRRequest(BaseModel):
    image: str  # base64
    detector: str = "ctd"
    ocr: str = "mit48px"
    lang_source: str = "日本語"
    params: Optional[Dict] = None

class TranslateRequest(BaseModel):
    queries: List[str]
    translator: str = "google"
    lang_source: str = "日本語"
    lang_target: str = "English"
    params: Optional[Dict] = None

class InpaintRequest(BaseModel):
    image: str  # base64
    mask: str   # base64
    inpainter: str = "lama_large_512px"
    params: Optional[Dict] = None

@app.get("/modules")
async def list_modules():
    return {
        "textdetectors": list(TEXTDETECTORS.module_dict.keys()),
        "ocr": list(OCR.module_dict.keys()),
        "translators": list(TRANSLATORS.module_dict.keys()),
        "inpainters": list(INPAINTERS.module_dict.keys())
    }

@app.post("/ocr")
async def run_ocr(req: OCRRequest):
    img = decode_image(req.image)

    det_cls = TEXTDETECTORS.get(req.detector)
    if not det_cls:
        raise HTTPException(status_code=400, detail="Detector not found")

    det_params = req.params or {}
    det = det_cls(**det_params)

    ocr_cls = OCR.get(req.ocr)
    if not ocr_cls:
        raise HTTPException(status_code=400, detail="OCR not found")

    ocr_params = req.params or {} # Simplified, usually params are separate per module
    ocr = ocr_cls(**ocr_params)

    mask, blk_list = det.detect(img)
    ocr.run_ocr(img, blk_list)

    results = []
    for blk in blk_list:
        results.append({
            "text": blk.get_text(),
            "box": blk.xyxy.tolist(),
            "lines_xyxy": [line.tolist() for line in blk.lines_xyxy],
            "translation": blk.translation
        })

    return {"blocks": results, "mask": encode_image(mask)}

@app.post("/translate")
async def run_translate(req: TranslateRequest):
    trans_cls = TRANSLATORS.get(req.translator)
    if not trans_cls:
        raise HTTPException(status_code=400, detail="Translator not found")

    trans_params = req.params or {}
    trans = trans_cls(lang_source=req.lang_source, lang_target=req.lang_target, **trans_params)
    translated = trans.translate(req.queries)

    return {"translations": translated}

@app.post("/inpaint")
async def run_inpaint(req: InpaintRequest):
    img = decode_image(req.image)
    mask = decode_image(req.mask)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    inp_cls = INPAINTERS.get(req.inpainter)
    if not inp_cls:
        raise HTTPException(status_code=400, detail="Inpainter not found")

    inp_params = req.params or {}
    inp = inp_cls(**inp_params)
    inpainted = inp.inpaint(img, mask)

    return {"image": encode_image(inpainted)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
