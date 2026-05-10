"""Task t09: Collect inference metrics and trigger retraining if drift detected"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pandas as pd
from clearml import Task
from shared.clearml_utils import is_training_in_progress
from shared.config import (
    CLEARML_PROJECT_NAME,
    DB_CONNECTION_STRING,
    DRIFT_WINDOW_DAYS,
    TAG_TRAINING_IN_PROGRESS,
)

DRIFT_THRESHOLDS = {
    "avg_probability_min": 0.50,
    "low_confidence_ratio_max": 0.30,
    "noise_ratio_inference_max": 0.35,
}


def fetch_inference_log(
    connection_string: str,
    window_days: int,
) -> pd.DataFrame:
    import sqlalchemy

    engine = sqlalchemy.create_engine(connection_string)
    since = datetime.now(timezone.utc) - timedelta(days=window_days)
    query = """
        SELECT predicted_at, probability, topic_id
        FROM inference_log
        WHERE predicted_at >= :since
        ORDER BY predicted_at
    """
    return pd.read_sql(query, engine, params={"since": since})


def compute_drift_metrics(df: pd.DataFrame) -> dict:
    import numpy as np

    if len(df) == 0:
        return {}

    probs = df["probability"].dropna()
    avg_prob = float(probs.mean()) if len(probs) > 0 else 0.0
    low_conf_ratio = float((probs < 0.4).mean()) if len(probs) > 0 else 0.0
    noise_ratio = float((df["topic_id"].isna() | (df["topic_id"] < 0)).mean())

    # Topic entropy
    topic_counts = df["topic_id"].dropna().value_counts()
    if len(topic_counts) > 0:
        p = topic_counts / topic_counts.sum()
        topic_entropy = float(-np.sum(p * np.log2(p + 1e-10)))
    else:
        topic_entropy = 0.0

    return {
        "avg_probability": avg_prob,
        "low_confidence_ratio": low_conf_ratio,
        "noise_ratio_inference": noise_ratio,
        "topic_entropy": topic_entropy,
        "total_predictions": len(df),
    }


def check_drift_trigger(
    df: pd.DataFrame,
    window_days: int,
    thresholds: dict,
) -> tuple[bool, list[str]]:
    """Return (triggered, reasons) if ANY single drift condition holds for >= window_days consecutive days"""
    if len(df) == 0:
        return False, ["no_data"]

    import datetime as _dt

    df = df.copy()
    df["date"] = pd.to_datetime(df["predicted_at"]).dt.date
    dates_sorted = sorted(df["date"].unique())

    if len(dates_sorted) < window_days:
        return False, []

    # Per-day flags for each condition
    per_day: dict[_dt.date, dict[str, bool]] = {}
    for date, day_df in df.groupby("date"):
        m = compute_drift_metrics(day_df)
        per_day[date] = {
            "avg_prob": m.get("avg_probability", 1.0)
            < thresholds["avg_probability_min"],
            "low_conf": m.get("low_confidence_ratio", 0.0)
            > thresholds["low_confidence_ratio_max"],
            "noise": m.get("noise_ratio_inference", 0.0)
            > thresholds["noise_ratio_inference_max"],
        }

    # Spec: "на протяжении 3+ дней" - each condition checked independently for consecutive days
    triggered_reasons: list[str] = []
    for condition_key in ("avg_prob", "low_conf", "noise"):
        condition_label = {
            "avg_prob": "avg_probability",
            "low_conf": "low_confidence_ratio",
            "noise": "noise_ratio_inference",
        }[condition_key]

        # Sliding window over consecutive calendar days
        streak = 0
        streak_start = None
        for i, date in enumerate(dates_sorted):
            if date not in per_day:
                streak = 0
                streak_start = None
                continue
            if per_day[date][condition_key]:
                if streak == 0:
                    streak_start = date
                streak += 1
                if streak >= window_days:
                    triggered_reasons.append(
                        f"{condition_label} violated for {streak} consecutive days starting {streak_start}"
                    )
                    break
            else:
                streak = 0
                streak_start = None

    return bool(triggered_reasons), triggered_reasons


def trigger_training_pipeline(training_days: int = 180, sample_size: int = 0):
    """Запускает training pipeline на последних training_days днях данных."""
    from pipelines.training_pipeline import run_pipeline  # noqa: PLC0415

    end_date = datetime.now(timezone.utc).date().isoformat()
    start_date = (datetime.now(timezone.utc) - timedelta(days=training_days)).date().isoformat()

    print(f"Triggering Training Pipeline: {start_date} -> {end_date}, sample_size={sample_size or 'all'}")
    run_pipeline(start_date=start_date, end_date=end_date, sample_size=sample_size)


def main():
    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name="t09_collect_metrics",
        task_type=Task.TaskTypes.monitor,
    )
    logger = task.get_logger()

    params = task.connect(
        {
            "drift_window_days": DRIFT_WINDOW_DAYS,
            "avg_probability_min": DRIFT_THRESHOLDS["avg_probability_min"],
            "low_confidence_ratio_max": DRIFT_THRESHOLDS["low_confidence_ratio_max"],
            "noise_ratio_inference_max": DRIFT_THRESHOLDS["noise_ratio_inference_max"],
            "retrain_days": 180,   # обучать на последних N днях при триггере
            "retrain_sample_size": 0,  # 0 = все записи за retrain_days
        }
    )

    window_days = int(params["drift_window_days"])
    retrain_days = int(params["retrain_days"])
    retrain_sample_size = int(params["retrain_sample_size"])
    thresholds = {
        "avg_probability_min": float(params["avg_probability_min"]),
        "low_confidence_ratio_max": float(params["low_confidence_ratio_max"]),
        "noise_ratio_inference_max": float(params["noise_ratio_inference_max"]),
    }

    conn_str = task.get_parameter("Args/db_connection_string") or DB_CONNECTION_STRING
    if not conn_str:
        print("ERROR: DB_CONNECTION_STRING not set. Cannot collect inference metrics.")
        task.close()
        return

    try:
        df = fetch_inference_log(conn_str, window_days)
        print(f"Fetched {len(df):,} inference records from last {window_days} days")
    except Exception as e:
        print(f"ERROR fetching inference log: {e}")
        task.close()
        return

    metrics = compute_drift_metrics(df)
    print(f"Drift metrics: {metrics}")

    for metric_name, value in metrics.items():
        logger.report_scalar("inference", metric_name, value=float(value), iteration=0)

    drift_triggered, reasons = check_drift_trigger(df, window_days, thresholds)

    logger.report_scalar(
        "drift", "drift_triggered", value=int(drift_triggered), iteration=0
    )

    if drift_triggered:
        print(f"DRIFT DETECTED: {reasons}")
        task.connect(
            {"drift_triggered": True, "reasons": "; ".join(reasons)}, name="drift"
        )

        if is_training_in_progress():
            print("Training already in progress - skipping trigger")
        else:
            trigger_training_pipeline(training_days=retrain_days, sample_size=retrain_sample_size)
    else:
        print("No drift detected")
        task.connect({"drift_triggered": False}, name="drift")

    report = {
        "window_days": window_days,
        "metrics": metrics,
        "drift_triggered": drift_triggered,
        "reasons": reasons,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    report_path = os.path.join(tempfile.gettempdir(), "monitoring_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    task.upload_artifact("monitoring_report.json", artifact_object=report_path)

    task.close()


if __name__ == "__main__":
    main()
