"""Persian web UI for the Stable Diffusion repository.

The browser/mobile phone is the client; inference runs on the host/server.
Set SD_MODEL_ID to a Diffusers-compatible model repository or local model path.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import torch
from diffusers import StableDiffusionPipeline
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
logger = logging.getLogger(__name__)
DATA_DIR = Path(os.getenv("SD_DATA_DIR", str(ROOT / "data")))
OUTPUT_DIR = Path(os.getenv("SD_OUTPUT_DIR", str(DATA_DIR / "outputs")))
HISTORY_FILE = DATA_DIR / "history.json"
MODEL_ID = os.getenv("SD_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
MAX_HISTORY = int(os.getenv("SD_MAX_HISTORY", "500"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="پنل تولید تصویر — Stable Diffusion", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

_pipeline: StableDiffusionPipeline | None = None
_pipeline_lock = threading.Lock()
_history_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_history() -> list[dict[str, Any]]:
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _write_history(items: list[dict[str, Any]]) -> None:
    temp = HISTORY_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(items[:MAX_HISTORY], ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(HISTORY_FILE)


def _get_pipeline() -> StableDiffusionPipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
            )
            pipe = pipe.to(device)
            if device == "cuda":
                try:
                    pipe.enable_attention_slicing()
                except (AttributeError, RuntimeError):
                    pass
            _pipeline = pipe
        return _pipeline


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    steps: int = Field(default=25, ge=1, le=60)
    width: int = Field(default=512, ge=256, le=768)
    height: int = Field(default=512, ge=256, le=768)
    seed: int | None = Field(default=None, ge=0, le=4_294_967_295)


def _run_generation(job_id: str, request: GenerateRequest) -> None:
    with _history_lock:
        _jobs[job_id]["status"] = "loading"
        _jobs[job_id]["updated_at"] = _now()
    try:
        pipe = _get_pipeline()
        with _history_lock:
            _jobs[job_id]["status"] = "generating"
            _jobs[job_id]["updated_at"] = _now()
        generator = None
        if request.seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(request.seed)
        with torch.inference_mode():
            result = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or None,
                num_inference_steps=request.steps,
                width=request.width,
                height=request.height,
                generator=generator,
            )
        filename = f"{job_id}.png"
        result.images[0].save(OUTPUT_DIR / filename)
        item = {
            "id": job_id,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "steps": request.steps,
            "width": request.width,
            "height": request.height,
            "seed": request.seed,
            "status": "completed",
            "image_url": f"/outputs/{filename}",
            "created_at": _jobs[job_id]["created_at"],
            "updated_at": _now(),
        }
        with _history_lock:
            _jobs[job_id].update(item)
            history = _read_history()
            history.insert(0, item)
            _write_history(history)
    except Exception as exc:  # Keep the job state useful to the UI; details go to server logs.
        logger.exception("Image generation failed for job %s", job_id)
        with _history_lock:
            _jobs[job_id].update({"status": "failed", "error": str(exc)[:500], "updated_at": _now()})


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_loaded": _pipeline is not None,
    }


@app.post("/api/generate", status_code=202)
def generate(request: GenerateRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    if not request.prompt.strip():
        raise HTTPException(status_code=422, detail="متن پرامپت نمی‌تواند خالی باشد.")
    if request.width % 8 or request.height % 8:
        raise HTTPException(status_code=422, detail="عرض و ارتفاع باید مضرب ۸ باشند.")
    job_id = uuid.uuid4().hex
    now = _now()
    with _history_lock:
        _jobs[job_id] = {
            "id": job_id,
            "prompt": request.prompt,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
        }
    background_tasks.add_task(_run_generation, job_id, request)
    return {"id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with _history_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="کار پیدا نشد یا سرور پس از شروع دوباره راه‌اندازی شده است.")
        return dict(job)


@app.get("/api/history")
def history() -> list[dict[str, Any]]:
    return _read_history()


@app.delete("/api/history")
def clear_history() -> dict[str, int]:
    with _history_lock:
        _write_history([])
    return {"deleted": 1}


@app.get("/api/gallery")
def gallery() -> list[dict[str, Any]]:
    items = _read_history()
    return [item for item in items if item.get("status") == "completed" and item.get("image_url")]
