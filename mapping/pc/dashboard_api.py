"""Realtime API for the Go2 mapping console.

The live ingestor should call ``publish`` after accepting a verified keyframe.
Until a robot session is active, this module emits a clearly labelled simulated
state so the visual console remains demonstrable without corrupting map data.
"""
import asyncio
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parent / "dashboard"
app = FastAPI(title="Go2 Mapping Console API")
app.mount("/assets", StaticFiles(directory=ROOT / "src"), name="assets")


def simulated_state(elapsed: float) -> dict:
    return {
        "source": "SIMULATED / awaiting verified LiDAR",
        "points": int(elapsed * 18000),
        "area": round(elapsed * 0.09, 2),
        "drift": 2.4,
        "latency": 82,
        "confidence": min(94, round(elapsed * 0.25, 1)),
        "distance": round(elapsed * 0.035, 2),
        "as_built_authority": True,
    }


@app.get("/")
async def index():
    return FileResponse(ROOT / "index.html")


@app.get("/api/state")
async def state():
    return simulated_state(time.monotonic() % 3600)


@app.websocket("/ws/live")
async def live(websocket: WebSocket):
    await websocket.accept()
    started = time.monotonic()
    try:
        while True:
            await websocket.send_json(simulated_state(time.monotonic() - started))
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return
