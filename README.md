# tlator — live subtitles (Whisper)

Реализован рабочий локальный web-app для "живых" субтитров с архитектурой из дорожной карты:

- аудио с микрофона (16 kHz mono)
- VAD + чанкер (WebRTC VAD)
- sliding window (fast 4s / refine 6s)
- fast + refine распознавание (faster-whisper)
- стабилизация токенов (stable/unstable)
- WebSocket state-store + UI с мягкой отрисовкой

## Структура

- `backend/app.py` — FastAPI app + websocket `/ws/subtitles`
- `backend/engine.py` — capture/vad/asr/stability core
- `backend/tokenize.py` — японская токенизация (fugashi; fallback по символам)
- `frontend/index.html` — UI (stable/unstable слои + fade + timestamp pacing)

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Откройте `frontend/index.html` в браузере.

## Что реализовано по roadmap

1. **Audio capture**: `sounddevice.RawInputStream`, int16, 30ms блоки.
2. **VAD + segmentation**: `webrtcvad`, черновые апдейты каждые ~0.6 сек, finalize на паузе ~400ms.
3. **Sliding windows**: fast=4s, refine=6s из ring buffer.
4. **Double inference**:
   - fast: `small`, beam=1
   - refine: `medium`, beam=4, реже/на паузе
5. **Stability engine**: фиксирует токены после повторов, хвост оставляет редактируемым.
6. **Merge behavior**: итог в UI идёт как стабильный+нестабильный слой.
7. **Japanese support**: токенизация через fugashi/MeCab-словарь `unidic-lite`.
8. **Word timestamps**: включены, UI добавляет токены с тайминговым pace.

## Ограничения

- Для `faster-whisper` желательно GPU; на CPU latency выше.
- Если зависимости/микрофон недоступны, backend стартует и отдаёт ошибку в websocket (`payload.error`).
