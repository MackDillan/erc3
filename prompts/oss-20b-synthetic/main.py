from dotenv import load_dotenv

from prompts.common import get_tasks_from_yaml, generate_synthetic_examples

load_dotenv(override=True)

if __name__ == '__main__':
    tasks = get_tasks_from_yaml(file_path="configs/tasks.yaml")
    filtered_tasks = [t for t in tasks if t not in []]

    for t in filtered_tasks:
        generate_synthetic_examples(t)
