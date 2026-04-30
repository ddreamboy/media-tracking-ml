import os

from clearml import Task


def main():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
    LLM_MODEL = os.getenv("LLM_MODEL")
    LLM_API_KEY = os.getenv("LLM_API_KEY")

    if not LLM_API_KEY or not LLM_BASE_URL or not LLM_MODEL:
        print(
            "LLM configuration is incomplete. Please set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL in the environment."
        )
        return

    task = Task.init(
        project_name="media_tracking_topic_modeling",
        task_name="test_llm_api",
        task_type=Task.TaskTypes.inference,
    )
    task.execute_remotely(queue_name="default")
    logger = task.get_logger()

    print("Testing LLM API connectivity with the following configuration:")
    print(
        f"LLM_BASE_URL={LLM_BASE_URL}\n"
        f"LLM_MODEL={LLM_MODEL}\n"
        f"LLM_API_KEY={LLM_API_KEY[:3] + '......' + LLM_API_KEY[-3:]}"
    )

    chat = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
    )
    response = chat.chat.completions.create(
        messages=[
            {"role": "user", "content": "Hello, can you respond to this test message?"}
        ]
    )

    logger.report_text(f"LLM API response: {response}")
