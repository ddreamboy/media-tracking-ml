import os
from pathlib import Path

from .config import HF_TOKEN, HF_USERNAME, HF_MODEL_REPO, HF_DATASET_REPO


def push_artifacts_to_hub(
    local_files: dict[str, str],
    tag: str,
    repo_id: str | None = None,
) -> str:
    from huggingface_hub import HfApi

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is not set")

    api = HfApi(token=HF_TOKEN)
    full_repo_id = repo_id or f"{HF_USERNAME}/{HF_MODEL_REPO}"

    try:
        api.create_repo(repo_id=full_repo_id, exist_ok=True, repo_type="model")
    except Exception:
        pass

    for remote_name, local_path in local_files.items():
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{tag}/{remote_name}",
            repo_id=full_repo_id,
            repo_type="model",
        )

    return f"https://huggingface.co/{full_repo_id}/tree/{tag}"


def push_dataset_meta_to_hub(meta_path: str, tag: str) -> None:
    from huggingface_hub import HfApi

    if not HF_TOKEN:
        return

    api = HfApi(token=HF_TOKEN)
    full_repo_id = f"{HF_USERNAME}/{HF_DATASET_REPO}"

    try:
        api.create_repo(repo_id=full_repo_id, exist_ok=True, repo_type="dataset")
    except Exception:
        pass

    api.upload_file(
        path_or_fileobj=meta_path,
        path_in_repo=f"{tag}/dataset_meta.json",
        repo_id=full_repo_id,
        repo_type="dataset",
    )


def pull_latest_artifacts(local_dir: str, repo_id: str | None = None) -> dict[str, str]:
    from huggingface_hub import HfApi, hf_hub_download
    import json

    full_repo_id = repo_id or f"{HF_USERNAME}/{HF_MODEL_REPO}"

    try:
        api = HfApi(token=HF_TOKEN or None)
        refs = api.list_repo_refs(repo_id=full_repo_id, repo_type="model")
        tags = [t.name for t in refs.tags]
        if not tags:
            return {}
        latest_tag = sorted(tags)[-1]
    except Exception as e:
        return {}

    Path(local_dir).mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, str] = {}

    for filename in ["topic_map_llm.csv", "topic_embeddings.npy"]:
        try:
            path = hf_hub_download(
                repo_id=full_repo_id,
                filename=f"{latest_tag}/{filename}",
                repo_type="model",
                local_dir=local_dir,
                token=HF_TOKEN or None,
            )
            downloaded[filename] = path
        except Exception:
            pass

    return downloaded
