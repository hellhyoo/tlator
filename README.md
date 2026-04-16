# tlator — live subtitles prototype

Минимальный локальный прототип desktop/web-утилиты с "живыми" субтитрами по вашей схеме:

- backend: FastAPI + WebSocket
- ядро: ring buffer + fast/refine pipeline (заглушка) + стабилизация токенов
- frontend: HTML/CSS/JS с отдельным рендером стабильного/нестабильного текста

## Что уже реализовано

1. **Skeleton backend** (`backend/app.py`)
   - `RingBuffer` для последних N секунд аудио.
   - `StabilityEngine` с lock-after-N повторов.
   - `MergeEngine` (общий префикс + замена хвоста).
   - `DemoInferenceLoop` как безопасная заглушка fast/refine-проходов.
   - `ws://localhost:8000/ws/subtitles` отдает состояние токенов.

2. **Skeleton frontend** (`frontend/index.html`)
   - Два слоя: stable (яркий), unstable (серый + blur).
   - Легкий fade-in у новых слов.

## Быстрый запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload
```

Откройте `frontend/index.html` в браузере (или отдайте statically через любой http-server).

## Следующие шаги (чтобы дойти до production-поведения)

1. Подключить **реальный микрофон** (`sounddevice`) в поток 20–50 мс.
2. Добавить **VAD** (`webrtcvad`) и чанки 0.5–0.8 сек.
3. Заменить `DemoInferenceLoop` на `faster-whisper`:
   - fast pass: `small`, beam=1
   - refine pass: `medium`, реже, окно длиннее
4. Для японского добавить **Sudachi/MeCab** токенизацию перед стабилизацией.
5. Включить `word_timestamps=True` и рендер "печати" по таймкодам.

