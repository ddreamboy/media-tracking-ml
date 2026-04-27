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
# Если хочешь API-эмбеддинги вместо локальной модели
EMBEDDING_PROVIDER=local_hf        # или: api
EMBEDDING_MODEL_NAME=deepvk/USER-bge-m3
EMBEDDING_BATCH_SIZE=512

# LLM модель (OpenAI-совместимый API)
LLM_BASE_URL=https://routerai.ru/api/v1
LLM_MODEL=google/gemini-2.5-flash-lite

# БД для мониторинга (t09) - можно не заполнять пока нет интеграции
DB_CONNECTION_STRING=postgresql://user:password@localhost:5432/media_tracking
```

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

> **По выборке:** HPO по умолчанию берёт 100k записей из датасета — этого достаточно для репрезентативного поиска, и каждый trial занимает минуты, а не часы. На полном датасете (750k+) один trial может идти 30-60 минут.

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
