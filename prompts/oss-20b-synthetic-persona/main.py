from concurrent.futures import ProcessPoolExecutor, as_completed

from dotenv import load_dotenv

from prompts.common import get_tasks_from_yaml, process_task

load_dotenv(override=True)

if __name__ == '__main__':
    tasks = get_tasks_from_yaml(file_path="configs/tasks.yaml")
    filtered_tasks = [t for t in tasks if t not in ["coding_expert", "constraint_expert", "error_handling_expert", "planning_expert", "tool_expert"]]

    for t in tasks:
        process_task(t, f"../oss-20b-synthetic/{t}_train_synthetic.jsonl")

    # with ProcessPoolExecutor(max_workers=8) as executor:
    #     futures = {executor.submit(process_task, t): t for t in filtered_tasks}
    #
    #     for future in as_completed(futures):
    #         task_name = futures[future]
    #         try:
    #             future.result()
    #         except Exception as e:
    #             print(f"Error processing {task_name}: {e}")
    #
    # print("All tasks completed")
