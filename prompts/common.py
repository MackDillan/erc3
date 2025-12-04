import os
from re import findall
from typing import Any

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from PromptWizard.promptwizard.glue.common.utils.file import save_jsonlist
from PromptWizard.promptwizard.glue.promptopt.instantiate import GluePromptOpt
from PromptWizard.promptwizard.glue.promptopt.techniques.common_logic import DatasetSpecificProcessing

load_dotenv(override=True)


class GSM8k(DatasetSpecificProcessing):
    TEXT_DELIMITER_PATTERN = r"(?s)(?<=<START>)(.*?)(?=(?:<END>|</START>|</END>))"
    TEXT_DELIMITER_PATTERN_MUTATION = r"(?s)(?<=<START>)(.*?)(?=(?:<END>|</START>|</END>))"

    def dataset_to_jsonl(self, dataset_jsonl: str, **kwargs: Any) -> None:
        def extract_answer_from_output(completion):
            # Your functions for metrics and prompt building
            ans_re = compile(r"#### (\-?[0-9\.\,]+)")
            self.INVALID_ANS = "[invalid]"

            match = ans_re.search(completion)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                return match_str
            else:
                return self.INVALID_ANS

        examples_set = []

        for _, sample in tqdm(enumerate(kwargs["dataset"]), desc="Evaluating samples"):
            example = {
                DatasetSpecificProcessing.QUESTION_LITERAL: sample['question'],
                DatasetSpecificProcessing.ANSWER_WITH_REASON_LITERAL: sample['answer'],
                DatasetSpecificProcessing.FINAL_ANSWER_LITERAL: extract_answer_from_output(sample["answer"])
            }
            examples_set.append(example)

        save_jsonlist(dataset_jsonl, examples_set, "w")

    def extract_final_answer(self, answer: str):

        if not answer:
            return self.INVALID_ANS

        model_pred = answer.lower()
        preds = model_pred.split(self.ANSWER_START.lower())
        answer_flag = True if len(preds) > 1 else False

        pred = preds[-1].replace(",", "")
        pred = [s for s in findall(r'-?\d+\.?\d*', pred)]

        if len(pred) == 0:
            return self.INVALID_ANS

        if answer_flag:
            # choose the first element in list
            pred = pred[0]
        else:
            # choose the last element in list
            pred = pred[-1]

        # (For arithmetic tasks) if a word ends with period, it will be omitted ...
        if pred[-1] == ".":
            pred = pred[:-1]
        return pred


def update_yaml_file(file_path, config_dict, prefix=""):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    for field, value in config_dict.items():
        data[field] = value

    output_path = f"tmp/{file_path.replace('/', f'/{prefix}')}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    print("YAML file updated successfully!")
    return output_path


def get_tasks_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def save_prompt_to_file(expert, expert_profile, best_prompt):
    print(f"{"=" * 20} saving prompt for {expert} into file {"=" * 20}")
    print("-" * 60)
    print(expert_profile)
    print("-" * 60)
    print(best_prompt)
    print("=" * 60)
    with open(f"{expert}_system.txt", 'w') as file:
        file.write(expert_profile)
    with open(f"{expert}_user.txt", 'w') as file:
        file.write(best_prompt)


def generate_synthetic_examples(t):
    tasks = get_tasks_from_yaml(file_path="configs/tasks.yaml")
    path_to_config = "configs"
    promptopt_config_path = os.path.join(path_to_config, "promptopt_config_synthetic.yaml")
    setup_config_path = os.path.join(path_to_config, "setup_config_synthetic.yaml")

    file_path = 'configs/promptopt_config_synthetic.yaml'
    # Set the following based on the use case
    config_dict = {
        "task_description": tasks[t]['task_description'],
        "base_instruction": tasks[t]['base_instruction'],
    }
    promptopt_config_path = update_yaml_file(file_path, config_dict, t)

    # generate synthetic examples
    gp = GluePromptOpt(promptopt_config_path,
                       setup_config_path,
                       dataset_jsonl=None,
                       data_processor=None)

    _, _ = gp.get_best_prompt(
        use_examples=False,
        run_without_train_examples=False,
        generate_synthetic_examples=True,
        t=t,
    )


def process_task(t, train_dataset_jsonl):
    tasks = get_tasks_from_yaml(file_path="configs/tasks.yaml")
    path_to_config = "configs"
    # generate expert prompts examples
    config_dict_eval = {
        "task_description": tasks[t]['task_description'],
        "base_instruction": tasks[t]['base_instruction'],
    }
    setup_config_path_eval = os.path.join(path_to_config, "setup_config_evaluator.yaml")
    file_path_eval = 'configs/promptopt_config_evaluator.yaml'

    promptopt_config_path_eval = update_yaml_file(file_path_eval, config_dict_eval, t)
    gsm8k_processor = GSM8k()
    # Trick to replace regexes in DatasetSpecificProcessing class to avoid ValueError
    DatasetSpecificProcessing.TEXT_DELIMITER_PATTERN = gsm8k_processor.TEXT_DELIMITER_PATTERN
    DatasetSpecificProcessing.TEXT_DELIMITER_PATTERN_MUTATION = gsm8k_processor.TEXT_DELIMITER_PATTERN

    gp = GluePromptOpt(promptopt_config_path_eval,
                       setup_config_path_eval,
                       dataset_jsonl=train_dataset_jsonl,
                       data_processor=gsm8k_processor)

    expert_profile, best_prompt = gp.get_best_prompt(
        use_examples=True,
        run_without_train_examples=False,
        generate_synthetic_examples=False,
    )

    save_prompt_to_file(t, expert_profile, best_prompt)


def run_task_pipeline(t):
    generate_synthetic_examples(t)
    process_task(t, f"{t}_train_synthetic.jsonl")
