import argparse
import json
import logging
import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from utils import load_json, StructuredDataLinearize
from main.config import DATASETS, get_heuristics, get_requests, get_end_prompt
from transformers import GPT2TokenizerFast

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

class BabelConvertor:
    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.linearizer = StructuredDataLinearize()
        self.linearizer.end_prompt = ""

    def set_split_obj(self, task, structured_type, split, objective, instruction, linearize_func, use_partition_mark, use_format_explanation, heuristic):
        self.prompt_input, self.split, self.objective, self.task, self.data_type = [], split, objective, task, structured_type
        self.instruct, self.linearize_func, self.use_partition_mark, self.use_format_explanation, self.heuristic = instruction, linearize_func, use_partition_mark, use_format_explanation, heuristic
        self.dataset = load_dataset(f"./scripts/dataset_collection/{task}.py")
        self.request = get_requests(task)
        self.end_prompt = "The answer is \n" if "heur" not in objective else "The structural information is \n"
        if "heur" in objective:
            self.request = get_heuristics(structured_type)[objective]

    def fit_heuristics_constraints(self, sequence, max_token_length=4000):
        return len(self.tokenizer(sequence).input_ids) < max_token_length

    def get_one_shot_example(self):
        return self.dataset['train'][np.random.randint(0, len(self.dataset['train']))]

    def retrieve_sample_list(self):
        return getattr(self, f"retrieve_{self.task}")()

    def to_linearized_data(self, _example, context_key=None):
        data = {
            "title": "",
            "context": _example.get(context_key, ""),
            "table": {
                "header": _example['table']['header'][0] if context_key else _example['table']['header'],
                "rows": _example['table']['rows'][0] if context_key else _example['table']['rows'],
                "caption": _example['table'].get('caption', "")
            }
        }
        return self.linearizer.retrieve_linear_function(self.linearize_func, self.use_partition_mark, self.use_format_explanation, False, data)

    def process_example(self, example, context_key=None):
        try:
            table_info = self.to_linearized_data(example, context_key)
        except:
            return None
        content = {
            "prompt": f"{self.instruct}{table_info}<request>\n{self.request}<statement>\n{example['statement']}\n{self.end_prompt}",
            "completion": "0" if example["label"] == "REFUTES" else "1"
        }
        return content if self.fit_heuristics_constraints("".join(content.values()), 4000 - 1024 - 500) else None

    def retrieve_feverous(self):
        return [self.process_example(example) for example in self.dataset[self.split] if self.process_example(example)]

    def retrieve_hybridqa(self):
        return [self.process_example(example, "context") for example in tqdm(self.dataset[self.split], leave=False) if self.process_example(example, "context")]

    def retrieve_sqa(self):
        return [self.process_example(example) for example in tqdm(self.dataset[self.split], leave=False) if self.process_example(example)]

    def retrieve_tabfact(self):
        return [self.process_example(example) for example in tqdm(self.dataset[self.split], leave=False) if self.process_example(example)]

    def retrieve_totto(self):
        return [self.process_example(example) for example in tqdm(self.dataset[self.split], leave=False) if self.process_example(example)]

def get_keys(dict, value):
    return [k for k, v in dict.items() if value in v]

def save_jsonl(output_path, content_list):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as outfile:
        for content in content_list:
            outfile.write(json.dumps(content) + "\n")

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=["formlm_opt", "formlm_qa", "formlm_block_type"], nargs="+")
    parser.add_argument("--objective", default=["zero"], nargs="+")
    parser.add_argument("--split", default=["validation"], nargs="+")
    parser.add_argument("--linearize_list", default=["html"], nargs="+")
    parser.add_argument("--use_partition_mark", default=False, action="store_true")
    parser.add_argument("--use_format_explanation", default=False, action="store_true")
    parser.add_argument("--change_order", default=False, action="store_true")
    parser.add_argument("--use_role_prompting", default=False, action="store_true")
    parser.add_argument("--heuristic", default=None, type=str)
    parser.add_argument("--unified", default=False, action="store_true")
    parser.add_argument("--unified_file_output", default="./exps/downstream_tasks_20230113_log/", type=str)
    return parser.parse_args()

def task_specific_babel_convertor():
    args = get_arguments()
    logging.info(args)
    if args.unified:
        unified_dict = {"content": [], "task": [], "objective": [], "choices": []}
    babel_convertor = BabelConvertor()
    split, obj = args.split[0], args.objective[0]

    for task in args.task:
        for linear_func in args.linearize_list:
            structured_type = get_keys(DATASETS, task)[0]
            instruction = f"You are a brilliant {structured_type} executor with the capabilities [retrieve], [input parsing], [metadata inference], [pattern understanding] who can understand the structural information of the {structured_type}.\n" if args.use_role_prompting else ""
            babel_convertor.set_split_obj(task, structured_type, split, obj, instruction, linear_func, args.use_partition_mark, args.use_format_explanation, args.heuristic)
            mode = f"{linear_func}_{int(args.use_partition_mark)}_{int(args.use_format_explanation)}_{int(args.use_role_prompting)}"
            content_list = babel_convertor.retrieve_sample_list()
            if args.unified:
                unified_dict["content"].append(content_list)
                unified_dict["task"].append(task)
                unified_dict["objective"].append(obj)
                unified_dict["choices"].append(mode)
            logging.info(f"Task-{task} Objective-{obj} Split-{split} Linear-{linear_func} has been saved..")
            save_jsonl(f"./generated/{task}/{obj}/{split}_{mode}.jsonl", content_list)

    if args.unified:
        save_jsonl(f"{args.unified_file_output}/validation_{mode}.jsonl", unified_dict)
        logging.info(f"unified version has been saved")

if __name__ == "__main__":
    task_specific_babel_convertor()
