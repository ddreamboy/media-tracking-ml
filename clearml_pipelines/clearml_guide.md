# ClearML: Руководство по запуску

## 1. Получить credentials

Зайди на [app.clear.ml](https://app.clear.ml) -> Settings -> Workspace -> **Create new credentials**.

Затем инициализируй ClearML на машине:

```bash
clearml-init
```

Вводишь:
- API host: `https://api.clear.ml`
- Access key и Secret key из UI

Это создаст `~/.clearml/clearml.conf`.

---

## 2. Заполнить .env

```bash
cp clearml_pipelines/.env.example clearml_pipelines/.env
```

### Обязательные

```env
# ClearML - те же ключи что вводил в clearml-init
CLEARML_API_ACCESS_KEY=xxxxxxxxxx
CLEARML_API_SECRET_KEY=xxxxxxxxxx

# HuggingFace - нужен для скачивания ddreamboy/media-tracking-topics-dataset
HF_TOKEN=hf_xxxxxxxxxx

# LLM - нужен для t06 (разметка тем)
LLM_API_KEY=xxxxxxxxxx
```

### Под эксперименты на 750k

```env
N_FULL_CORPUS=750000
```

### Опциональные (есть дефолты)

```env
# Провайдер эмбеддингов: local_hf | openai | gemini | qwen
# (всё кроме local_hf идёт через OpenAI-совместимый API)
EMBEDDING_PROVIDER=local_hf
EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-4B
EMBEDDING_BATCH_SIZE=2
EMBEDDING_MAX_SEQ_LENGTH=1024
EMBEDDING_TORCH_DTYPE=float16
EMBEDDING_DIMENSIONS=0             # 0 = нативные 2560; иначе MRL-усечение
EMBEDDING_MAX_WORKERS=8            # параллельные запросы, только для API-провайдеров

# LLM модель (OpenAI-совместимый API)
LLM_BASE_URL=https://routerai.ru/api/v1
LLM_MODEL=google/gemini-2.5-flash-lite

# БД для мониторинга (t09) - можно не заполнять пока нет интеграции
DB_CONNECTION_STRING=postgresql://user:password@localhost:5432/media_tracking
```

### Про эмбеддер

Обучение (t03) считает эмбеддинги **локально на GPU**, прод-инференс берёт **ту же модель
по API** через Router AI (`qwen/qwen3-embedding-4b`, переменная `ROUTER_AI_EMBEDDING_MODEL`
в конфиге воркера). Это должна быть одна и та же модель: UMAP и HDBSCAN обучены в конкретном
пространстве, и эмбеддинг другой моделью на инференсе разъезжается с обучением, даже если
размерности совпали.

Отсюда же требования к `Qwen/Qwen3-Embedding-4B`:

- **`EMBEDDING_TORCH_DTYPE=float16` обязателен.** В fp32 модель занимает ~16 ГБ и падает по
  OOM. В fp16 — ~8 ГБ, помещается на 16 ГБ карте.
- **`EMBEDDING_MAX_SEQ_LENGTH` обязателен.** В конфиге модели контекст 32768; без явного
  лимита активации не помещаются в память. 1024 обрезает менее 0.5% постов корпуса.
- **`EMBEDDING_BATCH_SIZE=2`** — не опечатка. Веса в fp16 занимают 7.5 ГБ, а Windows/WDDM
  отдаёт процессу лишь ~9 ГБ из 16, так что на активации остаётся около 1.2 ГБ. Замер
  рабочих точек на RTX 5060 Ti: `seq=1024` проходит только с `batch=2`, `seq=512` — с
  `batch=4`, `seq=256` — с `batch=16`. При этом батч почти не влияет на скорость
  (упор в саму модель): 18.8 док/с при 1024/2 против 23 док/с при 256/8 — поэтому берём
  максимальную длину. Ориентир по времени: ~1.5 ч на 100k документов, ~11 ч на 750k.
  Аллокатор `expandable_segments` на Windows не поддерживается, обойти фрагментацию им нельзя.
- Размерность — 2560 против 1024 у прежней BGE, то есть `embeddings.npy` в 2.5 раза тяжелее,
  а UMAP заметно дороже. Если станет узким местом — `EMBEDDING_DIMENSIONS` включает
  MRL-усечение, но менять его нужно **синхронно** с воркером инференса (там параметр
  `dimensions` у API).
- 8B-версия на 16 ГБ не помещается даже в fp16 (веса ~15.1 ГБ).

---

## 3. Установить зависимости

```bash
# GPU-машина
pixi install -e cuda

# CPU-машина
pixi install -e cpu
```

---

## 4. Запустить агентов

Нужно минимум два агента - под две очереди.

**default** - CPU, для t01/t02/t05/t06/t07/t08/t09 и контроллеров пайплайнов:

```bash
pixi run -e cpu clearml-agent daemon --queue default --detached
```

**gpu** - для t03 (эмбеддинги), t04 (обучение BERTopic) и HPO:

```bash
pixi run -e cuda clearml-agent daemon --queue gpu --detached
```

Проверить что агенты подключились:

```bash
clearml-agent list
```

---

## 5. Запустить пайплайн

```bash
cd clearml_pipelines

# Обучение на всём датасете
python pipelines/training_pipeline.py

# Обучение на 750k записях
python pipelines/training_pipeline.py --sample-size 750000

# С явными датами
python pipelines/training_pipeline.py --start-date 2025-01-01 --end-date 2025-06-30 --sample-size 750000
```

После запуска скрипт отправит задачу в очередь и завершится. Дальше всё выполняется на агентах.

### HPO

```bash
# Автоматически берёт артефакты из последних t02/t03 задач, сэмплирует 100k записей
python hpo/run_hpo.py --n-trials 30

# Указать размер выборки явно
python hpo/run_hpo.py --n-trials 30 --sample-size 150000

# Из конкретных ClearML task ID (можно найти в UI)
python hpo/run_hpo.py --n-trials 30 --preprocess-task-id <id> --embed-task-id <id>

# Из локальных файлов
python hpo/run_hpo.py --n-trials 30 \
  --data-path /path/to/preprocessed.parquet \
  --embeddings-path /path/to/embeddings.npy
```

Задача уйдёт на gpu-агент. Найденные гиперпараметры сохранятся в ClearML как артефакт и будут автоматически подхвачены следующим запуском training pipeline.

> **По выборке:** HPO по умолчанию берёт 100k записей из датасета - этого достаточно для репрезентативного поиска, и каждый trial занимает минуты, а не часы. На полном датасете (750k+) один trial может идти 30-60 минут.

### Validation & Promotion

```bash
python pipelines/validation_pipeline.py \
  --training-meta /tmp/training_meta.json \
  --topic-map /tmp/topic_map_llm.csv \
  --evolution-report /tmp/evolution_report.json \
  --model-path /tmp/bertopic_model \
  --topic-emb /tmp/topic_embeddings.npy
```

### Monitoring (разовый запуск)

```bash
python pipelines/monitoring_pipeline.py
```

### Monitoring (по расписанию, каждый день в 3:00 UTC)

```bash
python pipelines/monitoring_pipeline.py --schedule
```

---

## 6. Что появится в ClearML UI

| Раздел | Содержимое |
|---|---|
| **Pipelines** | Training Pipeline, Validation & Promotion Pipeline, Monitoring Pipeline |
| **Experiments** | Tasks t01–t09, HPO_BERTopic |
| **Scalars** (внутри каждого эксперимента) | DBCV, noise_ratio, cv_coherence, num_topics, training_duration |
| **Artifacts** (внутри каждого эксперимента) | модель, эмбеддинги, parquet-файлы, JSON-отчёты |
| **Models** | bertopic_model с тегами `production` / `archived` / `failed_validation` |

---

## Очереди и задачи

| Задача | Очередь | Описание |
|---|---|---|
| t01 data_fetch | default | Загрузка датасета с HF Hub |
| t02 preprocess | default | Очистка и лемматизация |
| t03 embed | **gpu** | Генерация эмбеддингов |
| t04 train_bertopic | **gpu** | Обучение BERTopic |
| t05 topic_evolution | default | Анализ эволюции тем |
| t06 topic_labeling | default | LLM-разметка тем |
| t07 validate_model | default | Проверка метрик по порогам |
| t08 promote_model | default | Продвижение модели в production |
| t09 collect_metrics | default | Детектирование дрейфа |
| HPO_BERTopic | **gpu** | Подбор гиперпараметров (Optuna) |
| Pipeline controllers | default | Оркестрация шагов |
