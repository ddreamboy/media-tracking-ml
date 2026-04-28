"""Task t01: Get data from HF Hub and ClearML Dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from clearml import Dataset, Task
from huggingface_hub import hf_hub_download
from shared.config import CLEARML_PROJECT_NAME

HF_REPO_ID = "ddreamboy/media-tracking-topics-dataset"
HF_FILENAME = "raw/posts.parquet"
CLEARML_DATASET_NAME = "media_tracking_posts"
TARGET_YEAR = 2025


def _find_parquet_file(base_dir: str) -> str:
    candidates = sorted(Path(base_dir).rglob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet files found in {base_dir}")

    for path in candidates:
        if path.as_posix().endswith("raw/posts.parquet"):
            return str(path)
    return str(candidates[0])


def _get_or_create_clearml_dataset() -> tuple[Dataset, str, bool]:
    try:
        dataset = Dataset.get(
            dataset_project=CLEARML_PROJECT_NAME,
            dataset_name=CLEARML_DATASET_NAME,
            only_published=False,
        )
        if not dataset.is_final():
            raise ValueError(f"Dataset {dataset.id} is not finalized — will recreate")
        local_copy = dataset.get_local_copy()
        parquet_path = _find_parquet_file(local_copy)
        print(f"Using existing ClearML Dataset: {dataset.id}")
        return dataset, parquet_path, False
    except Exception as e:
        print(f"Could not use existing dataset ({e}), downloading from HF Hub...")
        Path("data").mkdir(parents=True, exist_ok=True)
        file_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="dataset",
            local_dir="data",
            force_download=False,
        )
        print(f"Downloaded HF dataset to {file_path}")

        dataset = Dataset.create(
            dataset_name=CLEARML_DATASET_NAME,
            dataset_project=CLEARML_PROJECT_NAME,
            dataset_tags=["source:hf", f"year:{TARGET_YEAR}"],
        )
        dataset.add_files(file_path, dataset_path="raw")
        dataset.upload()
        dataset.finalize()

        print(f"Created ClearML Dataset from HF: {dataset.id}")
        return dataset, file_path, True


def main():
    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name="t01_data_fetch",
        task_type=Task.TaskTypes.data_processing,
    )
    logger = task.get_logger()

    import os

    _default_sample = int(os.environ.get("T01_SAMPLE_SIZE", "0"))
    params = task.connect({"sample_size": _default_sample})
    sample_size = int(params["sample_size"])

    dataset, parquet_path, dataset_created = _get_or_create_clearml_dataset()

    df = pd.read_parquet(parquet_path)

    df = df.rename(columns={
        "id_post": "post_id",
        "channel_name": "channel",
        "post_date": "created_at",
    })

    required_cols = {"post_id", "channel", "text", "created_at"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise RuntimeError(
            f"Dataset is missing required columns: {sorted(missing_cols)}"
        )

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])
    df = df[df["created_at"].dt.year == TARGET_YEAR].copy()

    num_records = len(df)
    if num_records == 0:
        raise RuntimeError(f"No records found for {TARGET_YEAR} in dataset")

    if sample_size > 0 and sample_size < num_records:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled {sample_size:,} records from {num_records:,}")
        num_records = len(df)

    start_date = datetime(TARGET_YEAR, 1, 1, tzinfo=timezone.utc).date().isoformat()
    end_date = datetime(TARGET_YEAR, 12, 31, tzinfo=timezone.utc).date().isoformat()
    channels_count = df["channel"].nunique()
    date_range_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1

    logger.report_scalar("dataset", "num_records", value=num_records, iteration=0)
    logger.report_scalar(
        "dataset", "date_range_days", value=date_range_days, iteration=0
    )
    logger.report_scalar("dataset", "channels_count", value=channels_count, iteration=0)
    logger.report_scalar(
        "dataset", "dataset_created", value=int(dataset_created), iteration=0
    )

    # Plot monthly distribution
    df["month"] = pd.to_datetime(df["created_at"]).dt.to_period("M").astype(str)
    monthly = df["month"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(monthly.index, monthly.values, color="#4C72B0")
    ax.set_title("Публикации по месяцам")
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Количество постов")
    plt.xticks(rotation=45)
    plt.tight_layout()
    dist_path = os.path.join(tempfile.gettempdir(), "monthly_distribution.png")
    fig.savefig(dist_path)
    plt.close(fig)
    logger.report_image("distribution", "monthly", iteration=0, local_path=dist_path)

    # Save artifacts
    parquet_path = os.path.join(tempfile.gettempdir(), "raw_data.parquet")
    df.to_parquet(parquet_path, index=False)
    task.upload_artifact("raw_data.parquet", artifact_object=parquet_path)

    month_dist = df["month"].value_counts().sort_index().to_dict()
    channel_dist = df["channel"].value_counts().head(20).to_dict()
    meta = {
        "start_date": start_date,
        "end_date": end_date,
        "num_records": num_records,
        "channels_count": channels_count,
        "date_range_days": date_range_days,
        "monthly_distribution": {str(k): int(v) for k, v in month_dist.items()},
        "top_channels": {str(k): int(v) for k, v in channel_dist.items()},
        "clearml_dataset_id": dataset.id,
        "source": "existing_clearml_dataset" if not dataset_created else "hf_hub",
    }
    meta_path = os.path.join(tempfile.gettempdir(), "dataset_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    task.upload_artifact("dataset_meta.json", artifact_object=meta_path)

    print(
        f"Done. Year: {TARGET_YEAR}, Records: {num_records}, "
        f"Channels: {channels_count}, Dataset: {dataset.id}"
    )
    task.close()


if __name__ == "__main__":
    main()
