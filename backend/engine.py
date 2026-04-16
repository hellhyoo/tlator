from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - import availability depends on host audio stack.
    sd = None

try:
    import webrtcvad
except Exception:  # pragma: no cover - import availability depends on package install.
    webrtcvad = None

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - import availability depends on package install.
    WhisperModel = None

from .tokenize import tokenize_for_ui

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class TimedToken:
    text: str
    start: float
    end: float
    stable: bool = False


class RingBuffer:
    def __init__(self, seconds: int = 10, sample_rate: int = 16_000) -> None:
        self.capacity = seconds * sample_rate
        self._buf: Deque[int] = deque(maxlen=self.capacity)

    def push(self, samples: np.ndarray) -> None:
        self._buf.extend(samples.astype(np.int16).tolist())

    def latest(self, seconds: float, sample_rate: int = 16_000) -> np.ndarray:
        count = int(seconds * sample_rate)
        if count <= 0:
            return np.array([], dtype=np.int16)
        return np.array(list(self._buf)[-count:], dtype=np.int16)


class AudioCapture:
    """Microphone capture pipeline (16k/mono/int16)."""

    def __init__(self, sample_rate: int = 16_000, block_ms: int = 30) -> None:
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.blocksize = int(sample_rate * block_ms / 1000)
        self._stream = None
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=256)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed; pip install -r backend/requirements.txt")
        self._loop = loop

        def callback(indata, frames, _time, status):
            if status:
                LOGGER.warning("Audio callback status: %s", status)
            arr = np.frombuffer(indata, dtype=np.int16).copy()
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._push_frame, arr)

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=1,
            dtype="int16",
            callback=callback,
        )
        self._stream.start()

    def _push_frame(self, arr: np.ndarray) -> None:
        if self._queue.full():
            _ = self._queue.get_nowait()
        self._queue.put_nowait(arr)

    async def read(self) -> np.ndarray:
        return await self._queue.get()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class VADChunker:
    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_ms: int = 30,
        speech_chunk_sec: float = 0.6,
        pause_to_finalize_ms: int = 400,
        aggressiveness: int = 2,
    ) -> None:
        if webrtcvad is None:
            raise RuntimeError("webrtcvad is not installed; pip install -r backend/requirements.txt")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.speech_chunk_samples = int(sample_rate * speech_chunk_sec)
        self.pause_frames = max(1, pause_to_finalize_ms // frame_ms)
        self.vad = webrtcvad.Vad(aggressiveness)

        self.pending = np.array([], dtype=np.int16)
        self.speech_buffer = np.array([], dtype=np.int16)
        self.silent_frames = 0

    def accept(self, pcm: np.ndarray) -> List[tuple[np.ndarray, bool]]:
        self.pending = np.concatenate([self.pending, pcm])
        out: List[tuple[np.ndarray, bool]] = []

        while len(self.pending) >= self.frame_samples:
            frame = self.pending[: self.frame_samples]
            self.pending = self.pending[self.frame_samples :]
            voiced = self.vad.is_speech(frame.tobytes(), self.sample_rate)

            if voiced:
                self.silent_frames = 0
                self.speech_buffer = np.concatenate([self.speech_buffer, frame])
                if len(self.speech_buffer) >= self.speech_chunk_samples:
                    out.append((self.speech_buffer.copy(), False))
            else:
                self.silent_frames += 1
                if len(self.speech_buffer) > 0 and self.silent_frames >= self.pause_frames:
                    out.append((self.speech_buffer.copy(), True))
                    self.speech_buffer = np.array([], dtype=np.int16)
                    self.silent_frames = 0

        return out


class StabilityEngine:
    def __init__(self, lock_after: int = 3, tail_unstable: int = 2) -> None:
        self.lock_after = lock_after
        self.tail_unstable = tail_unstable
        self.prev_tokens: List[str] = []
        self.repeats: List[int] = []

    def apply(self, tokens: List[TimedToken]) -> List[TimedToken]:
        texts = [t.text for t in tokens]
        next_repeats: List[int] = []

        for i, text in enumerate(texts):
            repeated = 1
            if i < len(self.prev_tokens) and text == self.prev_tokens[i]:
                repeated = self.repeats[i] + 1
            next_repeats.append(repeated)

        for i, token in enumerate(tokens):
            tail_guard = i >= len(tokens) - self.tail_unstable
            token.stable = (next_repeats[i] >= self.lock_after) and not tail_guard

        self.prev_tokens = texts
        self.repeats = next_repeats
        return tokens


class WhisperPipeline:
    def __init__(
        self,
        language: str = "ja",
        fast_model: str = "small",
        refine_model: str = "medium",
        device: str = "auto",
    ) -> None:
        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed; pip install -r backend/requirements.txt")

        compute = "float16" if device != "cpu" else "int8"
        self.language = language
        self.fast = WhisperModel(fast_model, device=device, compute_type=compute)
        self.refine = WhisperModel(refine_model, device=device, compute_type=compute)

    def _to_float32(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return np.array([], dtype=np.float32)
        return audio.astype(np.float32) / 32768.0

    def transcribe_fast(self, audio: np.ndarray) -> List[TimedToken]:
        return self._transcribe(self.fast, audio, beam_size=1)

    def transcribe_refine(self, audio: np.ndarray) -> List[TimedToken]:
        return self._transcribe(self.refine, audio, beam_size=4)

    def _transcribe(self, model, audio: np.ndarray, beam_size: int) -> List[TimedToken]:
        if len(audio) == 0:
            return []
        segments, _ = model.transcribe(
            self._to_float32(audio),
            language=self.language,
            beam_size=beam_size,
            temperature=0,
            word_timestamps=True,
            vad_filter=False,
        )

        out: List[TimedToken] = []
        for seg in segments:
            for word in seg.words or []:
                for ui_token in tokenize_for_ui(word.word, language=self.language):
                    out.append(TimedToken(text=ui_token, start=float(word.start), end=float(word.end)))
        return out


class SubtitleEngine:
    """Coordinates capture -> vad/chunking -> fast/refine -> stability."""

    def __init__(self, language: str = "ja") -> None:
        self.sample_rate = 16_000
        self.ring = RingBuffer(seconds=10, sample_rate=self.sample_rate)
        self.capture = AudioCapture(sample_rate=self.sample_rate, block_ms=30)
        self.chunker = VADChunker(sample_rate=self.sample_rate)
        self.stability = StabilityEngine(lock_after=3, tail_unstable=2)
        self.pipeline = WhisperPipeline(language=language)

        self.running = False
        self.current_tokens: List[TimedToken] = []
        self._lock = asyncio.Lock()
        self._bg_tasks: List[asyncio.Task] = []
        self._last_refine = 0.0

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        loop = asyncio.get_running_loop()
        self.capture.start(loop)
        self._bg_tasks = [
            asyncio.create_task(self._capture_loop(), name="capture_loop"),
        ]

    async def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self.capture.stop()
        for task in self._bg_tasks:
            task.cancel()
        await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        self._bg_tasks.clear()

    async def _capture_loop(self) -> None:
        while self.running:
            frame = await self.capture.read()
            self.ring.push(frame)
            chunks = self.chunker.accept(frame)

            for _chunk, finalized in chunks:
                await self._run_fast_pass()
                if finalized or (time.time() - self._last_refine) > 1.2:
                    await self._run_refine_pass()
                    self._last_refine = time.time()

    async def _run_fast_pass(self) -> None:
        window = self.ring.latest(seconds=4.0, sample_rate=self.sample_rate)
        tokens = await asyncio.to_thread(self.pipeline.transcribe_fast, window)
        async with self._lock:
            self.current_tokens = self.stability.apply(tokens)

    async def _run_refine_pass(self) -> None:
        window = self.ring.latest(seconds=6.0, sample_rate=self.sample_rate)
        tokens = await asyncio.to_thread(self.pipeline.transcribe_refine, window)
        async with self._lock:
            self.current_tokens = self.stability.apply(tokens)

    async def snapshot(self) -> dict:
        async with self._lock:
            tokens = [dataclasses.asdict(t) for t in self.current_tokens]
        return {
            "tokens": tokens,
            "emitted_at": time.time(),
            "full_text": "".join([t["text"] for t in tokens]),
        }
