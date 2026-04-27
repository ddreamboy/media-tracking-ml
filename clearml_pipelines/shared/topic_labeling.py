import json

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from .config import RANDOM_STATE


def compute_ring_boundaries(
    model,
    topic_ids: list[int],
    doc_embeddings: np.ndarray,
    topics: list[int],
    quantiles: tuple[float, float] = (0.25, 0.60),
    sample_per_topic: int = 200,
) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    all_distances: list[float] = []

    for tid in topic_ids:
        topic_emb = model.topic_embeddings_[tid + 1].reshape(1, -1)
        mask = np.array(topics) == tid
        if mask.sum() < 2:
            continue
        doc_embs = doc_embeddings[mask]
        if len(doc_embs) > sample_per_topic:
            idx = rng.choice(len(doc_embs), size=sample_per_topic, replace=False)
            doc_embs = doc_embs[idx]
        dists = cosine_distances(doc_embs, topic_emb).flatten()
        all_distances.extend(dists.tolist())

    all_distances_arr = np.array(all_distances)
    q_low = float(np.quantile(all_distances_arr, quantiles[0]))
    q_high = float(np.quantile(all_distances_arr, quantiles[1]))
    return q_low, q_high


def get_ring_samples(
    model,
    topic_id: int,
    doc_embeddings: np.ndarray,
    topics: list[int],
    docs: list[str],
    q_low: float,
    q_high: float,
    core_docs: int = 5,
    main_docs: int = 3,
    periphery_docs: int = 2,
) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    topic_emb = model.topic_embeddings_[topic_id + 1].reshape(1, -1)
    topic_mask = np.array(topics) == topic_id
    if topic_mask.sum() == 0:
        return {}

    topic_doc_embs = doc_embeddings[topic_mask]
    topic_docs = [d for d, m in zip(docs, topic_mask) if m]
    distances = cosine_distances(topic_doc_embs, topic_emb).flatten()

    rings_cfg = [
        (0.0, q_low, "Ядро", core_docs),
        (q_low, q_high, "Основная", main_docs),
        (q_high, 1.0, "Периферия", periphery_docs),
    ]

    result: dict = {}
    for ring_idx, (low, high, name, n_sample) in enumerate(rings_cfg):
        mask = (distances >= low) & (distances < high)
        ring_docs = [d for d, m in zip(topic_docs, mask) if m]
        ring_dists = distances[mask]

        if ring_docs:
            n = min(n_sample, len(ring_docs))
            idx = rng.choice(len(ring_docs), size=n, replace=False)
            sampled_docs = [ring_docs[i] for i in idx]
        else:
            sampled_docs = []

        result[f"ring_{ring_idx + 1}"] = {
            "name": name,
            "range": (low, high),
            "total_docs": int(mask.sum()),
            "sample_docs": sampled_docs,
        }

    result["_total_count"] = int(topic_mask.sum())
    return result


def make_label_prompt(top_words: list[str], ring_data: dict) -> str:
    words_str = ", ".join(top_words[:15])
    rings_str = ""
    for r_key in ["ring_1", "ring_2", "ring_3"]:
        rd = ring_data.get(r_key, {})
        if not rd or rd.get("total_docs", 0) == 0:
            continue
        rings_str += f"\n{'─' * 45}\n"
        rings_str += f"{rd['name'].upper()} ({rd['total_docs']} документов):\n"
        for doc in rd.get("sample_docs", []):
            rings_str += f"  • {doc[:200]}\n"

    return (
        f"Тематический кластер из русскоязычных Telegram-новостей.\n\n"
        f"Ключевые слова кластера: [{words_str}]\n\n"
        f"Документы по зонам (от центра к периферии):{rings_str}\n\n"
        "Создай трехуровневую иерархическую метку:\n\n"
        "l1 - ШИРОКИЙ: максимально общая область (Политика, Экономика, Общество, "
        "Безопасность, Культура, Наука, Спорт, Происшествия и т.д.) "
        "ЗАПРЕЩЕНО: страны, имена, конкретные события.\n\n"
        "l2 - СРЕДНИЙ: подобласть БЕЗ привязки к конкретным странам/событиям. "
        "Должна покрывать ВСЕ зоны, не только ядро. "
        "ЗАПРЕЩЕНО: конкретные страны, конфликты, имена людей.\n\n"
        "l3 - КОНКРЕТНЫЙ: специфика. Можно указать географию/контекст ТОЛЬКО если "
        "он присутствует в >70% документов ВСЕХ колец. Иначе - обобщи через явление.\n\n"
        'Примеры ПРАВИЛЬНО:\n{"l1": "Безопасность", "l2": "Вооруженные конфликты", "l3": "Боевые действия на постсоветском пространстве"}\n'
        '{"l1": "Общество", "l2": "Происшествия на транспорте", "l3": "ДТП с тяжелыми последствиями"}\n\n'
        "Также укажи:\n"
        "- core_topic: о чем ядро (1 предложение)\n"
        "- periphery_topic: о чем периферия (1 предложение, или null если похожа на ядро)\n"
        "- heterogeneous: true если ядро и периферия про разное\n"
        "- coverage: доля документов которую покрывает метка (0.0–1.0)\n\n"
        'Ответь строго JSON:\n{"l1": "...", "l2": "...", "l3": "...", "core_topic": "...", '
        '"periphery_topic": "...", "heterogeneous": false, "coverage": 0.9}'
    )


def call_llm_label(
    model,
    topic_id: int,
    ring_data: dict,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
    top_words = [w for w, _ in model.get_topic(topic_id)]
    prompt = make_label_prompt(top_words, ring_data)

    system = (
        "Ты эксперт по тематическому моделированию русскоязычных новостей. "
        "Создай иерархическую метку для тематического кластера на основе его документов. "
        "Документы разбиты по близости к центру темы: ядро -> основная -> периферия. "
        "Отвечай строго в формате JSON."
    )

    try:
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content.strip())
        assert "l1" in result and "l2" in result and "l3" in result
        return result
    except Exception as e:
        return {
            "l1": "ERROR",
            "l2": str(e)[:50],
            "l3": "",
            "heterogeneous": None,
            "coverage": None,
        }


def label_all_topics(
    model,
    topic_ids: list[int],
    doc_embeddings: np.ndarray,
    topics: list[int],
    docs: list[str],
    q_low: float,
    q_high: float,
    core_docs: int,
    main_docs: int,
    periphery_docs: int,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    inherited_labels: dict[int, dict] | None = None,
    stable_topic_ids: set[int] | None = None,
) -> list[dict]:
    if inherited_labels is None:
        inherited_labels = {}
    if stable_topic_ids is None:
        stable_topic_ids = set()

    topic_info = model.get_topic_info().set_index("Topic")
    records = []
    llm_calls = 0

    for tid in topic_ids:
        count = int(topic_info.loc[tid, "Count"]) if tid in topic_info.index else 0

        # Compute ring_data once per topic - reuse for both label source and zone counts
        ring_data = get_ring_samples(
            model,
            tid,
            doc_embeddings,
            topics,
            docs,
            q_low,
            q_high,
            core_docs,
            main_docs,
            periphery_docs,
        )

        if tid in stable_topic_ids and tid in inherited_labels:
            label = inherited_labels[tid]
            label_source = "inherited"
        else:
            label = call_llm_label(
                model, tid, ring_data, llm_model, llm_base_url, llm_api_key
            )
            llm_calls += 1
            label_source = "llm"

        records.append(
            {
                "Topic": tid,
                "Human_Label": f"{label.get('l1', '')}:{label.get('l2', '')}:{label.get('l3', '')}",
                "is_heterogeneous": bool(label.get("heterogeneous", False)),
                "coverage": label.get("coverage"),
                # evolution_type will be filled from evolution_report in the calling task
                "evolution_type": "",
                # label_source: "inherited" (from prod) or "llm" (freshly generated)
                "label_source": label_source,
                "zone_boundaries_q25_q60": f"{q_low:.4f},{q_high:.4f}",
                "l1": label.get("l1", ""),
                "l2": label.get("l2", ""),
                "l3": label.get("l3", ""),
                "core_topic": label.get("core_topic", ""),
                "periphery_topic": label.get("periphery_topic", ""),
                "count": count,
                "ring_1_count": ring_data.get("ring_1", {}).get("total_docs", 0),
                "ring_2_count": ring_data.get("ring_2", {}).get("total_docs", 0),
                "ring_3_count": ring_data.get("ring_3", {}).get("total_docs", 0),
                "_llm_calls": llm_calls,
            }
        )

    return records
