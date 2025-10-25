import os
import math
import json
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from bs4 import BeautifulSoup
import requests
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

import requests as httpx
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

APP_MODE = os.environ.get("CLASSIFIER_MODE", "LOCAL").upper()  # LOCAL or HF_API
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_MODEL = os.environ.get("HF_MODEL", "ProtectAI/deberta-v3-base-prompt-injection-v2")  # for HF API or local model path

# Frontend templates
templates = Jinja2Templates(directory="templates")
app = FastAPI(title="Prompt Injection Page Scanner")

# ---------- classifier setup ----------
classifier_local = None
if APP_MODE == "LOCAL":
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained("./my_prompt_injection_model")
    model = AutoModelForSequenceClassification.from_pretrained("./my_prompt_injection_model")
    classifier_local = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512,
        device=device
    )

class ScanRequest(BaseModel):
    url: str
    max_chars_per_segment: Optional[int] = 800
    max_segments: Optional[int] = 60

# ---------- helpers ----------
def fetch_dynamic_page(url: str, timeout_s: int = 15) -> str:
    """
    Try Playwright first to fetch JS-rendered content, fallback to requests.
    Returns page body HTML string.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=timeout_s * 1000)
            page.wait_for_load_state("networkidle", timeout=timeout_s * 1000)
            content = page.content()
            browser.close()
            return content
    except Exception as e:
        # fallback to requests
        try:
            resp = requests.get(url, headers={"User-Agent": "web-agent/1.0"}, timeout=timeout_s)
            resp.raise_for_status()
            return resp.text
        except Exception as e2:
            raise

def extract_text_segments(html: str, max_chars: int = 800, max_segments: int = 60) -> List[str]:
    """
    Parse HTML, take <article>, <main>, or body text. Split into paragraph-like segments
    with a cap on chars per segment and max segments in total.
    """
    soup = BeautifulSoup(html, "html.parser")

    # prefer main/article
    container = soup.find("main") or soup.find("article") or soup.body
    if not container:
        container = soup

    # gather paragraphs and headings
    nodes = container.find_all(["p", "li", "h1", "h2", "h3", "h4"])
    segments = []
    for n in nodes:
        txt = n.get_text(separator=" ", strip=True)
        if not txt:
            continue
        # split too long paragraphs into chunks
        if len(txt) <= max_chars:
            segments.append(txt)
        else:
            # break by sentences approximated by ". " or newline
            parts = []
            cur = ""
            for part in txt.split(". "):
                if cur:
                    cand = cur + ". " + part
                else:
                    cand = part
                if len(cand) > max_chars:
                    if cur:
                        parts.append(cur + ".")
                    # if single sentence too long, break hard
                    if len(part) > max_chars:
                        # break by words
                        words = part.split()
                        chunk = ""
                        for w in words:
                            if len(chunk) + len(w) + 1 > max_chars:
                                parts.append(chunk.strip() + ".")
                                chunk = w
                            else:
                                chunk += " " + w if chunk else w
                        if chunk:
                            parts.append(chunk.strip() + ".")
                        cur = ""
                    else:
                        cur = part
                else:
                    cur = cand
            if cur:
                parts.append(cur if cur.endswith(".") else cur + ".")
            segments.extend(parts)
        if len(segments) >= max_segments:
            break

    # fallback: if no nodes, use body text split by paragraphs
    if not segments:
        text = container.get_text(separator="\n")
        for chunk in text.split("\n\n"):
            s = chunk.strip()
            if s:
                segments.append(s[:max_chars])
            if len(segments) >= max_segments:
                break

    # trim to max_segments
    return segments[:max_segments]

def classify_segments_hf_api(segments: List[str]) -> List[Dict[str, Any]]:
    """
    Call HF hosted inference API in batch (sequentially).
    """
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
    api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    results = []
    for seg in segments:
        payload = {"inputs": seg}
        resp = httpx.post(api_url, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            # treat as unknown
            results.append({"text": seg, "label": "error", "score": 0.0, "raw": resp.text})
            continue
        out = resp.json()
        # out is likely list of dicts [{'label':..., 'score':...}]
        if isinstance(out, list) and len(out) > 0:
            best = out[0]
            results.append({"text": seg, "label": best.get("label"), "score": float(best.get("score", 0.0)), "raw": best})
        else:
            results.append({"text": seg, "label": "unknown", "score": 0.0, "raw": out})
    return results

def classify_segments_local(segments: List[str]) -> List[Dict[str, Any]]:
    results = []
    for seg in segments:
        out = classifier_local(seg)[0]
        results.append({"text": seg, "label": out.get("label"), "score": float(out.get("score", 0.0)), "raw": out})
    return results

def normalize_label(label: str) -> str:
    """Map model label names to 'benign' or 'injection' where possible."""
    if label is None:
        return "unknown"
    l = str(label).lower()
    if "inject" in l or "attack" in l or "jail" in l:
        return "injection"
    if "benign" in l or "safe" in l or "clean" in l:
        return "benign"
    # try label like LABEL_0/LABEL_1 where 1 = injection for many models
    if "label_1" in l or l.endswith("_1"):
        return "injection"
    if "label_0" in l or l.endswith("_0"):
        return "benign"
    return "unknown"

# ---------- routes ----------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "mode": APP_MODE})


@app.post("/scan", response_class=JSONResponse)
def scan(req: ScanRequest):
    url = req.url
    max_chars = req.max_chars_per_segment or 800
    max_segments = req.max_segments or 60

    # 1) Fetch
    try:
        html = fetch_dynamic_page(url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch page: {e}")

    # 2) Extract text
    segments = extract_text_segments(html, max_chars=max_chars, max_segments=max_segments)
    if not segments:
        raise HTTPException(status_code=400, detail="No text segments found on page.")

    # 3) Classify
    if APP_MODE == "HF_API":
        if not HF_API_TOKEN:
            raise HTTPException(status_code=400, detail="HF_API_TOKEN not set for HF_API mode.")
        raw_results = classify_segments_hf_api(segments)
    else:
        raw_results = classify_segments_local(segments)

    # 4) Normalize + filter
    annotated = []
    for r in raw_results:
        norm = normalize_label(r.get("label"))
        if norm == "injection":  # ✅ keep only suspicious segments
            annotated.append({
                "text": r.get("text"),
                "raw_label": r.get("label"),
                "label": norm,
                "score": r.get("score"),
                "raw": r.get("raw")
            })

    # 5) Sort by descending score
    annotated_sorted = sorted(annotated, key=lambda x: x["score"], reverse=True)

    return {
        "url": url,
        "total_segments": len(segments),
        "injection_count": len(annotated_sorted),
        "segments": annotated_sorted
    }

# health
@app.get("/health")
def health():
    return {"status": "ok", "mode": APP_MODE}