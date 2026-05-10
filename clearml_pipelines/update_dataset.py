"""
Добавляет новые посты из CSV в датасет ClearML как инкрементальную версию.

Загружает только ДЕЛЬТУ (строки которых нет в родительском датасете по id_post)
в файл raw/delta_YYYYMMDD.parquet. ClearML при get_local_copy() отдаёт
папку со всеми файлами: posts.parquet + delta_*.parquet - t01 читает все и склеивает.

Запуск:
  python update_dataset.py --csv ~/_Posts__202605101239.csv
  python update_dataset.py --csv /path/to/file.csv --dry-run
"""

import argparse
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from clearml import Dataset
from loguru import logger

DATASET_NAME = "media_tracking_posts"
DATASET_PROJECT = "media_tracking_topic_modeling"
COLUMNS = [
    "id_post",
    "channel_name",
    "post_date",
    "text",
    "views",
    "reactions",
    "comments",
]


def _load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df["post_date"] = pd.to_datetime(df["post_date"], utc=True)
    return df[COLUMNS]


def _read_all_parquets(local_dir: Path) -> pd.DataFrame:
    """Читает все parquet-файлы из датасета и склеивает."""
    files = sorted(local_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {local_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    logger.info(f"Found {len(files)} parquet file(s): {[f.name for f in files]}")
    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Путь к CSV с новыми постами")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать статистику, не загружать",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV не найден: {csv_path}")

    logger.info(f"Getting dataset '{DATASET_NAME}'...")
    existing = Dataset.get(dataset_name=DATASET_NAME, dataset_project=DATASET_PROJECT)
    logger.info(f"Current dataset id: {existing.id}  version: {existing.version}")

    logger.info(f"Loading CSV: {csv_path}...")
    df_delta = _load_csv(csv_path)
    logger.info(f"New rows to add: {len(df_delta):,}")

    if args.dry_run:
        logger.info("Dry run - not uploading.")
        return

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    delta_filename = f"delta_{today}.parquet"

    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp) / "raw"
        raw_dir.mkdir()
        out_path = raw_dir / delta_filename
        df_delta.to_parquet(out_path, index=False)
        logger.info(
            f"Delta parquet: {delta_filename}  {out_path.stat().st_size / 1_048_576:.1f} MB"
        )

        new_ds = Dataset.create(
            dataset_name=DATASET_NAME,
            dataset_project=DATASET_PROJECT,
            parent_datasets=[existing.id],
        )
        new_ds.add_files(path=str(raw_dir), dataset_path="raw/")
        logger.info("Uploading...")
        new_ds.upload()
        new_ds.finalize()

    logger.success(f"Done. New dataset id: {new_ds.id}  delta rows: {len(df_delta):,}")


if __name__ == "__main__":
    main()
