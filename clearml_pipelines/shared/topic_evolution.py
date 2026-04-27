import json
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_top_words(model, topic_id: int, n: int = 10) -> set[str]:
    return {word for word, _ in model.get_topic(topic_id)[:n]}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def call_llm_judge(
    words_a: set,
    words_b: set,
    topic_id_a: int,
    topic_id_b: int,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
    system = (
        "Ты эксперт по тематическому моделированию текстов. "
        "Тебе будут даны два набора ключевых слов, представляющих две темы из новостных Telegram-каналов. "
        "Твоя задача - определить, описывают ли они одну и ту же тему или разные темы. "
        "Отвечай строго в формате JSON, без пояснений и markdown."
    )
    user = (
        f"Проанализируй две темы из русскоязычных новостных Telegram-каналов.\n\n"
        f"Тема A (topic_id={topic_id_a}):\nКлючевые слова: [{', '.join(sorted(words_a)[:10])}]\n\n"
        f"Тема B (topic_id={topic_id_b}):\nКлючевые слова: [{', '.join(sorted(words_b)[:10])}]\n\n"
        "Определи:\n"
        "1. Описывают ли эти две темы одну и ту же тематику? (true/false)\n"
        "2. Уверенность в ответе: high (очевидно), medium (вероятно), low (неясно)\n"
        "3. Краткое обоснование на русском языке (1-2 предложения)\n\n"
        'Ответь строго в формате JSON:\n{"same_topic": true/false, "confidence": "high"/"medium"/"low", "reasoning": "..."}'
    )
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=200,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content.strip())
        assert "same_topic" in result and "confidence" in result
        return result
    except Exception as e:
        return {"same_topic": False, "confidence": "low", "reasoning": f"ERROR: {e}"}


def run_topic_evolution(
    new_model,
    prod_model,
    cosine_threshold_high: float = 0.7,
    cosine_threshold_low: float = 0.4,
    jaccard_threshold: float = 0.3,
    top_n_words: int = 10,
    llm_model: str = "",
    llm_base_url: str = "",
    llm_api_key: str = "",
) -> dict:
    new_topic_ids = sorted([t for t in new_model.get_topics().keys() if t != -1])
    prod_topic_ids = sorted([t for t in prod_model.get_topics().keys() if t != -1])

    emb_new = np.array([new_model.topic_embeddings_[t + 1] for t in new_topic_ids])
    emb_prod = np.array([prod_model.topic_embeddings_[t + 1] for t in prod_topic_ids])

    sim_matrix = cosine_similarity(emb_new, emb_prod)

    words_new = {t: get_top_words(new_model, t, top_n_words) for t in new_topic_ids}
    words_prod = {t: get_top_words(prod_model, t, top_n_words) for t in prod_topic_ids}

    jaccard_matrix = np.array(
        [
            [jaccard_similarity(words_new[a], words_prod[b]) for b in prod_topic_ids]
            for a in new_topic_ids
        ]
    )

    combined_matrix = 0.7 * sim_matrix + 0.3 * jaccard_matrix

    # best match: new -> prod
    best_new2prod: dict[int, dict] = {}
    for i, tn in enumerate(new_topic_ids):
        j_best = int(np.argmax(combined_matrix[i]))
        tp_best = prod_topic_ids[j_best]
        best_new2prod[tn] = {
            "matched_prod": tp_best,
            "cosine": float(sim_matrix[i, j_best]),
            "jaccard": float(jaccard_matrix[i, j_best]),
            "combined": float(combined_matrix[i, j_best]),
        }

    # best match: prod -> new
    best_prod2new: dict[int, dict] = {}
    for j, tp in enumerate(prod_topic_ids):
        i_best = int(np.argmax(combined_matrix[:, j]))
        tn_best = new_topic_ids[i_best]
        best_prod2new[tp] = {
            "matched_new": tn_best,
            "cosine": float(sim_matrix[i_best, j]),
            "jaccard": float(jaccard_matrix[i_best, j]),
            "combined": float(combined_matrix[i_best, j]),
        }

    # LLM for border zone
    border_pairs = [
        (tn, best_new2prod[tn]["matched_prod"], best_new2prod[tn]["cosine"])
        for tn in new_topic_ids
        if cosine_threshold_low <= best_new2prod[tn]["cosine"] < cosine_threshold_high
    ]

    llm_results: dict[tuple[int, int], dict] = {}
    if llm_api_key and border_pairs:
        for tn, tp, cosine_val in border_pairs:
            llm_results[(tn, tp)] = call_llm_judge(
                words_new[tn],
                words_prod[tp],
                tn,
                tp,
                llm_model,
                llm_base_url,
                llm_api_key,
            )

    def is_match(tn: int, tp: int, cosine_val: float) -> bool:
        if cosine_val >= cosine_threshold_high:
            return True
        if cosine_threshold_low <= cosine_val < cosine_threshold_high:
            llm_res = llm_results.get((tn, tp))
            if llm_res is not None:
                return bool(llm_res["same_topic"])
        return False

    matched_new2prod: dict[int, int] = {}
    for tn in new_topic_ids:
        tp_best = best_new2prod[tn]["matched_prod"]
        if is_match(tn, tp_best, best_new2prod[tn]["cosine"]):
            matched_new2prod[tn] = tp_best

    matched_prod2new: dict[int, int] = {}
    for tp in prod_topic_ids:
        tn_best = best_prod2new[tp]["matched_new"]
        if is_match(tn_best, tp, best_prod2new[tp]["cosine"]):
            matched_prod2new[tp] = tn_best

    # Topology analysis
    prod_sources: dict[int, list[int]] = defaultdict(list)  # tp -> [tn, ...]
    for tn, tp in matched_new2prod.items():
        prod_sources[tp].append(tn)

    new_targets: dict[int, list[int]] = defaultdict(list)  # tn -> [tp, ...]
    for tn, tp in matched_new2prod.items():
        new_targets[tn].append(tp)

    topic_mapping = []
    max_prod_id = max(prod_topic_ids) if prod_topic_ids else -1
    next_new_id = max_prod_id + 1

    # Disappeared
    for tp in prod_topic_ids:
        if tp not in matched_prod2new:
            topic_mapping.append(
                {
                    "new_topic_id": None,
                    "evolution_type": "Disappeared",
                    "matched_old_topic_id": tp,
                    "similarity_score": None,
                    "inherited_topic_id": None,
                    "llm_verdict": None,
                }
            )

    # Emerged, Stable, Merged, Split
    for tn in new_topic_ids:
        if tn not in matched_new2prod:
            inherited_id = next_new_id
            next_new_id += 1
            topic_mapping.append(
                {
                    "new_topic_id": tn,
                    "evolution_type": "Emerged",
                    "matched_old_topic_id": None,
                    "similarity_score": None,
                    "inherited_topic_id": inherited_id,
                    "llm_verdict": None,
                }
            )
        else:
            tp = matched_new2prod[tn]
            cosine_val = best_new2prod[tn]["cosine"]
            llm_verdict = (
                llm_results.get((tn, tp), {}).get("same_topic")
                if (tn, tp) in llm_results
                else None
            )

            # Stable: 1:1 bidirectional
            if len(prod_sources[tp]) == 1 and len(new_targets[tn]) == 1:
                topic_mapping.append(
                    {
                        "new_topic_id": tn,
                        "evolution_type": "Stable",
                        "matched_old_topic_id": tp,
                        "similarity_score": cosine_val,
                        "inherited_topic_id": tp,
                        "llm_verdict": llm_verdict,
                    }
                )
            # Merged: multiple new -> one prod
            elif len(prod_sources[tp]) > 1:
                topic_mapping.append(
                    {
                        "new_topic_id": tn,
                        "evolution_type": "Merged",
                        "matched_old_topic_id": tp,
                        "similarity_score": cosine_val,
                        "inherited_topic_id": tp,
                        "llm_verdict": llm_verdict,
                    }
                )
            # Split: one new -> multiple prod
            elif len(new_targets[tn]) > 1:
                # Senior sibling = largest cluster by cosine similarity to prod topic
                siblings = new_targets[tn]
                senior = max(siblings, key=lambda s: best_new2prod[s]["cosine"])
                is_senior = senior == tn
                inh = tp if is_senior else next_new_id
                if not is_senior:
                    next_new_id += 1
                topic_mapping.append(
                    {
                        "new_topic_id": tn,
                        "evolution_type": "Split",
                        "matched_old_topic_id": tp,
                        "similarity_score": cosine_val,
                        "inherited_topic_id": inh,
                        "llm_verdict": llm_verdict,
                    }
                )
            else:
                topic_mapping.append(
                    {
                        "new_topic_id": tn,
                        "evolution_type": "Stable",
                        "matched_old_topic_id": tp,
                        "similarity_score": cosine_val,
                        "inherited_topic_id": tp,
                        "llm_verdict": llm_verdict,
                    }
                )

    type_counts: dict[str, int] = defaultdict(int)
    for rec in topic_mapping:
        type_counts[rec["evolution_type"]] += 1

    llm_same = sum(1 for r in llm_results.values() if r.get("same_topic"))
    llm_diff = len(llm_results) - llm_same

    return {
        "embedding_changed": False,
        "cold_start": False,
        "summary": dict(type_counts),
        "llm_border_cases_total": len(llm_results),
        "llm_same_topic_verdict": llm_same,
        "llm_different_topic_verdict": llm_diff,
        "topic_mapping": topic_mapping,
        "_sim_matrix": sim_matrix,
        "_new_topic_ids": new_topic_ids,
        "_prod_topic_ids": prod_topic_ids,
    }


def make_cold_start_evolution(new_model) -> dict:
    new_topic_ids = sorted([t for t in new_model.get_topics().keys() if t != -1])
    topic_mapping = []
    # Cold start: no existing IDs -> assign topic_id == new_topic_id
    for tn in new_topic_ids:
        topic_mapping.append(
            {
                "new_topic_id": tn,
                "evolution_type": "Emerged",
                "matched_old_topic_id": None,
                "similarity_score": None,
                "inherited_topic_id": tn,
                "llm_verdict": None,
            }
        )
    return {
        "embedding_changed": False,
        "cold_start": True,
        "summary": {"Emerged": len(new_topic_ids)},
        "llm_border_cases_total": 0,
        "llm_same_topic_verdict": 0,
        "llm_different_topic_verdict": 0,
        "topic_mapping": topic_mapping,
    }


def make_embedding_changed_evolution(new_model) -> dict:
    result = make_cold_start_evolution(new_model)
    result["embedding_changed"] = True
    result["cold_start"] = False
    return result
