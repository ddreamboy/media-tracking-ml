import argparse
import tempfile
from pathlib import Path

import pandas as pd
from clearml import Dataset
from loguru import logger

DATASET_NAME = "media_tracking_posts"
DATASET_PROJECT = "media_tracking_topic_modeling"
DATASET_FILE = "raw/posts.parquet"
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
    logger.info(f"Current dataset id: {existing.id}")

    local_dir = existing.get_local_copy()
    parquet_path = Path(local_dir) / DATASET_FILE
    logger.info(
        f"Downloaded: {parquet_path}  ({parquet_path.stat().st_size / 1_048_576:.1f} MB)"
    )

    df_existing = pd.read_parquet(parquet_path, engine="fastparquet")
    logger.info(f"Existing rows: {len(df_existing):,}")

    logger.info(f"Loading CSV: {csv_path}...")
    df_new = _load_csv(csv_path)
    logger.info(f"New rows: {len(df_new):,}")

    df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    before = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset=["id_post"], keep="last")
    dupes = before - len(df_merged)
    if dupes:
        logger.info(f"Dropped {dupes:,} duplicates (by id_post)")
    logger.info(
        f"Merged rows: {len(df_merged):,}  (+{len(df_merged) - len(df_existing):,} net new)"
    )

    if args.dry_run:
        logger.info("Dry run — not uploading.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "posts.parquet"
        df_merged.to_parquet(out_path, index=False)
        logger.info(
            f"Merged parquet size: {out_path.stat().st_size / 1_048_576:.1f} MB"
        )

        new_ds = Dataset.create(
            dataset_name=DATASET_NAME,
            dataset_project=DATASET_PROJECT,
            parent_datasets=[existing.id],
        )
        new_ds.add_files(path=str(out_path), dataset_path=DATASET_FILE)
        logger.info("Uploading...")
        new_ds.upload()
        new_ds.finalize()

    logger.success(f"Done. New dataset id: {new_ds.id}  rows: {len(df_merged):,}")


if __name__ == "__main__":
    main()
