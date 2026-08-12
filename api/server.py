"""
ContentPulse — FastAPI Backend
Serves D3.js interactive dashboard with causal analysis and cross-platform comparison results.
"""
import os
import json
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
import uvicorn

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("data/outputs")
RESULTS_FILE = OUTPUT_DIR / "results.json"

app = FastAPI(title="ContentPulse", description="Content Consumption Intelligence Dashboard")
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")
templates = Jinja2Templates(directory="dashboard/templates")


def load_results() -> dict:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/causal")
async def get_causal_results():
    results = load_results()
    return JSONResponse(results.get("causal", {}))


@app.get("/api/crossplatform")
async def get_crossplatform_results():
    results = load_results()
    return JSONResponse(results.get("crossplatform", {}))


@app.get("/api/sessions")
async def get_session_stats():
    results = load_results()
    return JSONResponse(results.get("session_stats", {}))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
