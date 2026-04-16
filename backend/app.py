from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

from .engine import SubtitleEngine

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

engine: SubtitleEngine | None = None
startup_error: str | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine, startup_error
    try:
        engine = SubtitleEngine(language="ja")
        await engine.start()
        LOGGER.info("Subtitle engine started")
    except Exception as exc:  # pragma: no cover - depends on local hardware/deps.
        startup_error = str(exc)
        LOGGER.exception("Failed to start subtitle engine")
    yield
    if engine is not None:
        await engine.stop()


app = FastAPI(title="Live Subtitles", lifespan=lifespan)


@app.get("/")
def root() -> HTMLResponse:
    msg = "Live subtitles backend is running"
    if startup_error:
        msg = f"Backend is up, but engine failed to start: {startup_error}"
    return HTMLResponse(
        f"""
<!doctype html>
<html>
  <head><meta charset='utf-8'><title>Live Subtitles</title></head>
  <body>
    <h2>{msg}</h2>
    <p>Connect frontend to <code>ws://localhost:8000/ws/subtitles</code>.</p>
  </body>
</html>
"""
    )


@app.websocket("/ws/subtitles")
async def subtitles_socket(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            if startup_error:
                await ws.send_json({"error": startup_error, "tokens": []})
            elif engine is None:
                await ws.send_json({"error": "engine not initialized", "tokens": []})
            else:
                await ws.send_json(await engine.snapshot())
            await asyncio.sleep(0.2)
    except Exception:
        await ws.close()
