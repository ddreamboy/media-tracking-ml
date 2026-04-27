"""On-demand Optuna HPO for BERTopic hyperparameters.

Usage:
    python hpo/run_hpo.py --data-path /path/to/preprocessed.parquet \
        --embeddings-path /path/to/embeddings.npy \
        --n-trials 30
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import optuna
import pandas as pd
from clearml import Task
from shared.bertopic_utils import build_bertopic, compute_metrics
from shared.config import CLEARML_PROJECT_NAME, RANDOM_STATE

SEARCH_SPACE = {
    "n_neighbors": {"type": "int", "low": 15, "high": 100},
    "n_components": {"type": "int", "low": 5, "high": 15},
    "min_cluster_size": {"type": "int", "low": 50, "high": 200},
    "min_samples": {"type": "int", "low": 5, "high": 150},
}

N_RANDOM = 10
N_TPE = 20


def objective(trial, docs: list[str], embeddings: np.ndarray) -> float:
    hparams = {
        "n_neighbors": trial.suggest_int("n_neighbors", 15, 100),
        "n_components": trial.suggest_int("n_components", 5, 15),
        "min_cluster_size": trial.suggest_int("min_cluster_size", 50, 200),
        "min_samples": trial.suggest_int("min_samples", 5, 150),
        "umap_metric": "cosine",
        "hdbscan_metric": "euclidean",
        "top_n_words": 10,
    }

    try:
        model = build_bertopic(hparams, corpus_size=len(docs))
        topics, _ = model.fit_transform(docs, embeddings=embeddings)
        metrics = compute_metrics(model, topics, docs)

        dbcv = metrics["dbcv_score"]
        noise_ratio = metrics["noise_ratio"]

        if np.isnan(dbcv):
            return -999.0

        # L = DBCV - 0.5 * NoiseRatio
        return dbcv - 0.5 * noise_ratio
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return -999.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--embeddings-path", required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--study-name", default="bertopic_hpo")
    args = parser.parse_args()

    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name="HPO_BERTopic",
        task_type=Task.TaskTypes.optimizer,
    )
    task.execute_remote(queue_name="gpu")
    logger = task.get_logger()

    task.connect(
        {
            "n_trials": args.n_trials,
            "n_random": N_RANDOM,
            "n_tpe": N_TPE,
            "search_space": SEARCH_SPACE,
        }
    )

    df = pd.read_parquet(args.data_path)
    embeddings = np.load(args.embeddings_path)
    docs = df["text_lemm"].tolist()

    print(f"HPO: {len(docs):,} docs, {args.n_trials} trials")

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=N_RANDOM,
        seed=RANDOM_STATE,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=args.study_name,
    )
    study.optimize(
        lambda trial: objective(trial, docs, embeddings),
        n_trials=args.n_trials,
    )

    best_params = study.best_params
    best_value = study.best_value

    print(f"\nBest params (L={best_value:.4f}):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    best_hparams = {
        **best_params,
        "umap_metric": "cosine",
        "hdbscan_metric": "euclidean",
        "top_n_words": 10,
    }

    for k, v in best_hparams.items():
        logger.report_scalar("best_hparams", k, value=float(v), iteration=0)
    logger.report_scalar("hpo", "best_objective", value=best_value, iteration=0)
    logger.report_scalar("hpo", "n_trials", value=len(study.trials), iteration=0)

    hparams_path = "/tmp/best_hparams.json"
    with open(hparams_path, "w") as f:
        json.dump(best_hparams, f, indent=2)
    task.upload_artifact("best_hparams.json", artifact_object=hparams_path)

    # Save trials CSV
    trials_df = study.trials_dataframe()
    trials_path = "/tmp/hpo_trials.csv"
    trials_df.to_csv(trials_path, index=False)
    task.upload_artifact("hpo_trials.csv", artifact_object=trials_path)

    print("\nSaved best_hparams.json to ClearML artifact")
    print("Next Training Pipeline run will automatically use these parameters.")

    task.close()


if __name__ == "__main__":
    main()
