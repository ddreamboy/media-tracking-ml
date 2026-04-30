from clearml import Task
from shared.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


def main():
    from openai import OpenAI

    task = Task.init(
        project_name="media_tracking_topic_modeling",
        task_name="test_llm_api",
        task_type=Task.TaskTypes.inference,
    )
    task.execute_remotely(queue_name="default")
    logger = task.get_logger()

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
