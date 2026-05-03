# distanceDetection

CLI-пайплайн для real-time детекции и инстанс-сегментации на базе Ultralytics YOLO/YOLOE. Приложение умеет читать камеру, экран, изображения, видеофайлы и URL/стримы, показывать bounding boxes, masks и FPS в OpenCV-окне.

## Возможности

- Инстанс-сегментация объектов через YOLO/YOLOE `*-seg.pt` модели.
- Детекция объектов bounding boxes.
- FPS overlay для видеопотока, камеры и захвата экрана.
- YOLOE text prompt: поиск объектов по текстовому описанию через `--text`.
- Архитектура разделена на слои `domain`, `use_cases`, `adapters`, `infrastructure`, чтобы позже добавить FastAPI или другой UI без переписывания ядра.

Visual/reference-image prompt заложен в конфигурации, но CLI-выбор bbox-примера пока не реализован. При `--prompt-image` приложение выводит понятное сообщение о необходимости bbox provider.

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Положите веса модели в `models/`. По умолчанию используется:

```text
models/yoloe-26x-seg.pt
```

Можно передать другой путь через `--model`.

Настройки можно задать через `.env`. Скопируйте `.env.example` в `.env` и при необходимости измените:

```env
MODELS_DIR=models
YOLOE_WEIGHTS_FILE=yoloe-26x-seg.pt
YOLOE_TEXT_MODEL_FILE=mobileclip2_b.ts
UPLOADS_DIR=uploads
```

## Быстрый Старт

Камера с bbox + masks + FPS:

```powershell
python main.py --source 0 --mode both
```

Только bounding boxes:

```powershell
python main.py --source 0 --mode detect
```

Только segmentation masks:

```powershell
python main.py --source 0 --mode segment
```

Выход из OpenCV-окна: нажмите `q`.

## Источники

```powershell
# Камера
python main.py --source 0 --mode both

# Экран
python main.py --source screen --mode both

# Изображение
python main.py --source "path\to\image.jpg" --mode both

# Видео
python main.py --source "path\to\video.mp4" --mode both

# RTSP/RTMP/HTTP URL
python main.py --source "rtsp://example.com/media.mp4" --mode both
```

## YOLOE Text Prompt

YOLOE может искать классы по текстовому prompt. Для первого запуска может потребоваться время на загрузку text encoder весов Ultralytics.

```powershell
python main.py --source 0 --mode both --text "person,bus"
```

По умолчанию text encoder берётся из локального файла, собранного из `MODELS_DIR` и `YOLOE_TEXT_MODEL_FILE`:

```text
models/mobileclip2_b.ts
```

Если файл лежит в другом месте, укажите путь явно:

```powershell
python main.py --source 0 --mode both --text "person,bus" --text-model "models\mobileclip2_b.ts"
```

Для неизвестных или редких объектов лучше пробовать несколько объектных формулировок:

```powershell
python main.py --source 0 --mode both --text "tree,tree trunk,branch,log" --conf 0.1 --imgsz 960
```

Важно: text prompt лучше работает с объектами, а не материалами. Например, `tree`, `log`, `wooden chair` обычно лучше, чем `wood`.

## Аргументы CLI

```text
--model PATH              путь к весам модели
--source SOURCE           камера, screen, изображение, видео или URL
--mode segment|detect|both
--text "a,b,c"            YOLOE text prompts
--text-model PATH         путь к локальному text encoder `mobileclip2_b.ts`
--prompt-image PATH       заготовка под visual prompt, bbox UI пока не реализован
--conf FLOAT              confidence threshold, default 0.25
--imgsz INT               размер inference, default 640
--device DEVICE           cpu, 0, cuda:0 или авто-выбор Ultralytics
--window-name NAME        имя OpenCV-окна
```

Полная справка:

```powershell
python main.py --help
```

## Тесты

```powershell
python -m pytest
```

Проверяются парсинг CLI/config, FPS meter, определение типа source и ожидаемый отказ для `--prompt-image`.

## Структура

```text
app/
  domain/          # entities, value objects, ports
  use_cases/       # pipeline loop, FPS meter
  adapters/        # OpenCV sources/renderer, Ultralytics engine
  infrastructure/  # CLI/config mapping
main.py            # composition root
tests/             # focused unit tests
```

## Диагностика

Если OpenCV/Ultralytics/Torch не установлены в активном окружении:

```text
Missing runtime dependency: <package>. Install dependencies from requirements.txt.
```

Убедитесь, что активировано `.venv` и зависимости установлены:

```powershell
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```