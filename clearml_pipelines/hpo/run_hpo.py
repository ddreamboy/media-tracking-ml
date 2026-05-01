"""On-demand Optuna HPO for BERTopic hyperparameters.

Usage:
    # Автоматически берёт артефакты из последних t02/t03 задач ClearML
    python hpo/run_hpo.py --n-trials 30 --sample-size 100000

    # Из конкретных ClearML task ID
    python hpo/run_hpo.py --n-trials 30 --sample-size 100000 \
        --preprocess-task-id <t02_task_id> --embed-task-id <t03_task_id>

    # Из локальных файлов
    python hpo/run_hpo.py --n-trials 30 \
        --data-path /path/to/preprocessed.parquet \
        --embeddings-path /path/to/embeddings.npy
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from clearml import Task
from shared.config import CLEARML_PROJECT_NAME, RANDOM_STATE

SEARCH_SPACE = {
    "n_neighbors": {"type": "int", "low": 15, "high": 100},
    "n_components": {"type": "int", "low": 5, "high": 15},
    "min_cluster_size": {"type": "int", "low": 50, "high": 200},
    "min_samples": {"type": "int", "low": 5, "high": 150},
}

N_RANDOM = 10
N_TPE = 20


def load_from_clearml(
    preprocess_task_id: str = None, embed_task_id: str = None
) -> tuple[str, str]:
    """Fetch preprocessed.parquet and embeddings.npy from ClearML artifacts."""
    if preprocess_task_id:
        preprocess_task = Task.get_task(task_id=preprocess_task_id)
    else:
        preprocess_task = Task.get_task(
            project_name=CLEARML_PROJECT_NAME,
            task_name="t02_preprocess",
        )
    print(f"Using preprocess task: {preprocess_task.id} ({preprocess_task.name})")
    data_path = preprocess_task.artifacts["preprocessed.parquet"].get_local_copy()

    if embed_task_id:
        embed_task = Task.get_task(task_id=embed_task_id)
    else:
        embed_task = Task.get_task(
            project_name=CLEARML_PROJECT_NAME,
            task_name="t03_embed",
        )
    print(f"Using embed task: {embed_task.id} ({embed_task.name})")
    embeddings_path = embed_task.artifacts["embeddings.npy"].get_local_copy()

    return data_path, embeddings_path


def run_with_args(args) -> None:
    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name="HPO_BERTopic",
        task_type=Task.TaskTypes.optimizer,
    )

    # Fix: point remote agent directly at run_hpo.py, not main.py
    task.set_script(
        entry_point="hpo/run_hpo.py",
        working_dir="clearml_pipelines",
    )

    params = {
        "n_trials": args.n_trials if args.n_trials is not None else 30,
        "sample_size": args.sample_size if args.sample_size is not None else 100000,
        "study_name": args.study_name if args.study_name is not None else "bertopic_hpo",
        "n_random": N_RANDOM,
        "n_tpe": N_TPE,
        "search_space": SEARCH_SPACE,
    }
    task.connect(params)

    if task.execute_remotely(queue_name="gpu"):
        return

    # Read back possibly overridden values from ClearML UI
    args.n_trials = int(params["n_trials"])
    args.sample_size = int(params["sample_size"])
    args.study_name = str(params["study_name"])

    import json

    import numpy as np
    import optuna
    import pandas as pd
    from shared.bertopic_utils import build_bertopic, compute_metrics

    logger = task.get_logger()

    def objective(trial, docs: list[str], embeddings) -> float:
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
            return dbcv - 0.5 * noise_ratio
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return -999.0

    if args.data_path and args.embeddings_path:
        data_path = args.data_path
        embeddings_path = args.embeddings_path
        print("Loading data from local files")
    else:
        data_path, embeddings_path = load_from_clearml(
            preprocess_task_id=args.preprocess_task_id,
            embed_task_id=args.embed_task_id,
        )

    df = pd.read_parquet(data_path)
    embeddings = np.load(embeddings_path)
    assert len(df) == len(embeddings), "Mismatch between df and embeddings length"

    sample_size = args.sample_size
    total = len(df)
    if sample_size > 0 and sample_size < total:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(total, size=sample_size, replace=False)
        idx.sort()
        df = df.iloc[idx].reset_index(drop=True)
        embeddings = embeddings[idx]
        print(f"HPO sample: {sample_size:,} / {total:,} docs")
    else:
        print(f"HPO: using full dataset {total:,} docs")

    docs = df["text_lemm"].tolist()
    print(f"Running {args.n_trials} trials")

    sampler = optuna.samplers.TPESampler(n_startup_trials=N_RANDOM, seed=RANDOM_STATE)
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
    logger.report_scalar("hpo", "sample_size", value=len(docs), iteration=0)

    hparams_path = "/tmp/best_hparams.json"
    with open(hparams_path, "w") as f:
        json.dump(best_hparams, f, indent=2)
    task.upload_artifact("best_hparams.json", artifact_object=hparams_path)

    trials_df = study.trials_dataframe()
    trials_path = "/tmp/hpo_trials.csv"
    trials_df.to_csv(trials_path, index=False)
    task.upload_artifact("hpo_trials.csv", artifact_object=trials_path)

    print("\nSaved best_hparams.json to ClearML artifact")
    print("Next Training Pipeline run will automatically use these parameters.")

    task.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess-task-id", dest="preprocess_task_id", default=None)
    parser.add_argument("--embed-task-id", dest="embed_task_id", default=None)
    parser.add_argument("--data-path", dest="data_path", default=None)
    parser.add_argument("--embeddings-path", dest="embeddings_path", default=None)
    parser.add_argument("--n-trials", dest="n_trials", type=int, default=30)
    parser.add_argument("--sample-size", dest="sample_size", type=int, default=100000)
    parser.add_argument("--study-name", dest="study_name", default="bertopic_hpo")
    run_with_args(parser.parse_args())


if __name__ == "__main__":
    main()
