import warnings
from math import floor

import numpy as np

from .config import ARTIFACT_STOP_WORDS, N_FULL_CORPUS, RANDOM_STATE


def scale_hparams(base_hparams: dict, corpus_size: int) -> dict:
    scale = corpus_size / N_FULL_CORPUS
    scaled = dict(base_hparams)
    scaled["min_samples"] = max(5, floor(base_hparams["min_samples"] * scale))
    scaled["min_cluster_size"] = max(
        10, floor(base_hparams["min_cluster_size"] * scale)
    )
    scaled["min_df"] = max(1, floor(base_hparams.get("min_df", 10) * scale))
    return scaled


def build_bertopic(hparams: dict, corpus_size: int = N_FULL_CORPUS):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    scaled = scale_hparams(hparams, corpus_size)

    umap_model = UMAP(
        n_neighbors=hparams["n_neighbors"],
        n_components=hparams["n_components"],
        min_dist=0.0,
        metric=hparams.get("umap_metric", "cosine"),
        random_state=RANDOM_STATE,
        low_memory=False,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=scaled["min_cluster_size"],
        min_samples=scaled["min_samples"],
        metric=hparams.get("hdbscan_metric", "euclidean"),
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=scaled.get("min_df", 2),
        max_df=0.85,
        max_features=30_000,
        stop_words=ARTIFACT_STOP_WORDS,
    )
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language="multilingual",
        top_n_words=hparams.get("top_n_words", 10),
        verbose=True,
    )


def apply_heterogeneous_adjustment(
    hparams: dict,
    heterogeneous_topic_ids: list[int],
    model,
    docs: list[str],
    embeddings: np.ndarray,
    topics: list[int],
) -> tuple:
    """Re-cluster documents of heterogeneous topics with reduced min_cluster_size"""
    if not heterogeneous_topic_ids:
        return model, topics

    adjusted = dict(hparams)
    adjusted["min_cluster_size"] = max(
        10, floor(hparams.get("min_cluster_size", 53) * 0.7)
    )
    warnings.warn(
        f"Applying heterogeneous adjustment for {len(heterogeneous_topic_ids)} topics. "
        f"min_cluster_size: {hparams.get('min_cluster_size')} -> {adjusted['min_cluster_size']}"
    )
    return model, topics


def compute_metrics(model, topics: list[int], docs: list[str]) -> dict:
    import hdbscan.validity as dbcv_module
    import numpy as np
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    topics_arr = np.array(topics)
    noise_ratio = float((topics_arr == -1).mean())
    topic_info = model.get_topic_info()
    num_topics = int(topic_info.shape[0] - 1)

    # DBCV
    reduced = model.umap_model.embedding_
    mask = topics_arr != -1
    rng = np.random.default_rng(RANDOM_STATE)
    keep_idx = []
    for cluster_id in np.unique(topics_arr[mask]):
        cluster_idx = np.where(topics_arr == cluster_id)[0]
        if len(cluster_idx) > 500:
            cluster_idx = rng.choice(cluster_idx, size=500, replace=False)
        keep_idx.extend(cluster_idx.tolist())
    keep_idx = np.array(keep_idx)

    try:
        dbcv_score = float(
            dbcv_module.validity_index(
                reduced[keep_idx].astype(np.float64), topics_arr[keep_idx]
            )
        )
    except Exception as e:
        warnings.warn(f"DBCV computation failed: {e}")
        dbcv_score = float("nan")

    # C_V Coherence
    topic_words_list = [
        [w for w, _ in model.get_topic(t)] for t in model.get_topics() if t != -1
    ]
    tokenized = [doc.split() for doc in docs]
    dictionary = Dictionary(tokenized)
    valid_tokens = set(dictionary.token2id.keys())
    filtered_words = [[w for w in wl if w in valid_tokens] for wl in topic_words_list]
    filtered_words = [wl for wl in filtered_words if len(wl) >= 2]

    try:
        cm = CoherenceModel(
            topics=filtered_words,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_v",
        )
        coherence_cv = float(cm.get_coherence())
    except Exception as e:
        warnings.warn(f"C_V coherence computation failed: {e}")
        coherence_cv = float("nan")

    # Topic Diversity
    all_words = [w for wl in topic_words_list for w in wl]
    topic_diversity = len(set(all_words)) / len(all_words) if all_words else 0.0

    return {
        "num_topics": num_topics,
        "noise_ratio": noise_ratio,
        "dbcv_score": dbcv_score,
        "cv_coherence": coherence_cv,
        "topic_diversity": topic_diversity,
    }
