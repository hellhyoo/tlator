"""Minimal live-subtitles prototype backend.

This server exposes:
- /ws/subtitles websocket for pushing subtitle state to UI
- a simple in-process engine skeleton that demonstrates
  fast/refine merge with stable/unstable tokens

The audio capture + Whisper calls are intentionally stubbed in this first
prototype so the project can run everywhere and be extended incrementally.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections import deque
from typing import Deque, Iterable, List

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse


@dataclasses.dataclass
class TokenState:
    text: str
    stable: bool


class RingBuffer:
    """Time-based audio ring buffer skeleton for 16 kHz mono frames."""

    def __init__(self, seconds: int = 10, sample_rate: int = 16_000) -> None:
        self.capacity = seconds * sample_rate
        self._buf: Deque[float] = deque(maxlen=self.capacity)

    def push(self, samples: Iterable[float]) -> None:
        self._buf.extend(samples)

    def latest(self, count: int) -> List[float]:
        if count <= 0:
            return []
        return list(self._buf)[-count:]


class StabilityEngine:
    """Locks tokens once they reappear several times in same position."""

    def __init__(self, lock_after: int = 2) -> None:
        self.lock_after = lock_after
        self.prev_tokens: List[str] = []
        self.repeats: List[int] = []

    def update(self, new_tokens: List[str]) -> List[TokenState]:
        next_repeats: List[int] = []
        out: List[TokenState] = []

        for i, token in enumerate(new_tokens):
            repeated = 1
            if i < len(self.prev_tokens) and token == self.prev_tokens[i]:
                repeated = self.repeats[i] + 1
            next_repeats.append(repeated)
            out.append(TokenState(text=token, stable=repeated >= self.lock_after))

        self.prev_tokens = new_tokens
        self.repeats = next_repeats
        return out


class MergeEngine:
    """Replace only the tail after common prefix, reducing visual jitter."""

    @staticmethod
    def merge_tail(prev_text: str, new_text: str) -> str:
        common = 0
        limit = min(len(prev_text), len(new_text))
        while common < limit and prev_text[common] == new_text[common]:
            common += 1
        return prev_text[:common] + new_text[common:]


class DemoInferenceLoop:
    """Deterministic stub for fast/refine Whisper passes.

    Replace `_fast_pass` and `_refine_pass` with faster-whisper calls.
    """

    def __init__(self) -> None:
        self.buffer = RingBuffer()
        self.stability = StabilityEngine(lock_after=2)
        self.current_text = ""
        self.started = time.time()

    async def tick(self) -> List[TokenState]:
        # Simulated evolving hypotheses (JP phrases as in target use case).
        elapsed = int(time.time() - self.started)
        fast = self._fast_pass(elapsed)
        refine = self._refine_pass(elapsed)

        merged = MergeEngine.merge_tail(fast, refine)
        self.current_text = MergeEngine.merge_tail(self.current_text, merged)

        # crude whitespace tokenization for prototype; replace with Sudachi/MeCab.
        tokens = [t for t in self.current_text.split(" ") if t]
        return self.stability.update(tokens)

    def _fast_pass(self, t: int) -> str:
        samples = [
            "今日は",
            "今日は いい",
            "今日は いい 天気",
            "今日は いい 天気 です",
            "今日は いい 天気 ですね",
        ]
        return samples[min(t, len(samples) - 1)]

    def _refine_pass(self, t: int) -> str:
        samples = [
            "今日は",
            "今日は いい",
            "今日は とても いい 天気",
            "今日は とても いい 天気 です",
            "今日は とても いい 天気 ですね",
        ]
        return samples[min(max(t - 1, 0), len(samples) - 1)]


app = FastAPI(title="Live Subtitles Prototype")
engine = DemoInferenceLoop()


@app.get("/")
def root() -> HTMLResponse:
    return HTMLResponse(
        """
<!doctype html>
<html>
  <head><meta charset=\"utf-8\"><title>Live Subtitles Backend</title></head>
  <body>
    <h2>Live Subtitles backend is running.</h2>
    <p>Open <code>/frontend/index.html</code> from a static server and connect to <code>/ws/subtitles</code>.</p>
  </body>
</html>
"""
    )


@app.websocket("/ws/subtitles")
async def subtitles_socket(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            token_states = await engine.tick()
            await ws.send_json(
                {
                    "tokens": [dataclasses.asdict(t) for t in token_states],
                    "full_text": " ".join([t.text for t in token_states]),
                }
            )
            await asyncio.sleep(0.6)
    except Exception:
        await ws.close()
