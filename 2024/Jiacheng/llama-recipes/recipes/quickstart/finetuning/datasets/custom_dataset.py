import json
import itertools
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

B_INST, E_INST = "[INST]", "[/INST]"

def load_new_dataset(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def get_custom_dataset(dataset_config, tokenizer, split=None):
    dataset_config = '/home/ljc/representation-engineering/llama-recipes/src/llama_recipes/datasets/couterdata/counterfact_first_100.json'
    
    data = load_new_dataset(dataset_config)
    dataset = Dataset.from_list(data)
    
    PROMPT_DICT = {
        "prompt_counter_fact": (
            "{statement} {target_new}"
        ),
    }
    
    def format_example(example, tokenizer, prompt_dict):
        requested_rewrite = example['requested_rewrite']
        statement = requested_rewrite['prompt']
        target_new = requested_rewrite['target_new']['str']
        
        # Configure prompt
        prompts = example['generation_prompts'][:3]
        formatted_prompts = [
            prompt_dict["prompt_counter_fact"].format(statement=prompt, target_new=target_new)
            for prompt in prompts
        ]

        # Tokenize
        input_ids = []
        labels = []
        for prompt in formatted_prompts:
            # Encode the prompt and the target separately
            prompt_tokens = tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {prompt.strip()} ", add_special_tokens=False)
            target_tokens = tokenizer.encode(f"{target_new} {E_INST}", add_special_tokens=False)

            # Combine the tokens
            full_tokens = prompt_tokens + target_tokens

            # Create the labels
            prompt_labels = [-100] * len(prompt_tokens)  # Ignore the prompt part in the loss
            target_labels = target_tokens[:len(target_new)] + [-100] * (len(target_tokens) - len(target_new))

            input_ids.append(full_tokens)
            labels.append(prompt_labels + target_labels)

        combined_tokens = {
            "input_ids": list(itertools.chain(*input_ids)),
            "labels": list(itertools.chain(*labels)),
            "attention_mask": [1] * len(list(itertools.chain(*input_ids)))
        }
        
        return combined_tokens

    formatted_dataset = dataset.map(lambda sample: format_example(sample, tokenizer, PROMPT_DICT), remove_columns=list(dataset.features))
    
    # Print a few samples to verify
    # for i in range(3):
    #     print(f"Sample {i} - Input IDs: {formatted_dataset[i]['input_ids']}")
    #     print(f"Sample {i} - Labels: {formatted_dataset[i]['labels']}")
    
    return formatted_dataset

# Example usage
if __name__ == "__main__":
    dataset_config = '/home/ljc/representation-engineering/llama-recipes/src/llama_recipes/datasets/couterdata/counterfact_first_100.json'
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Update with your actual model

    # split = "train"  # or "test"
    formatted_data = get_custom_dataset(dataset_config, tokenizer )

# import json
# import itertools
# from transformers import AutoTokenizer
# from datasets import load_dataset, Dataset

# B_INST, E_INST = "[INST]", "[/INST]"

# def load_new_dataset(file_path):
#     with open(file_path, "r") as file:
#         data = json.load(file)
#     return data

# def format_example(example, tokenizer, prompt_dict):
#     requested_rewrite = example['requested_rewrite']
#     statement = requested_rewrite['prompt']
#     target_new = requested_rewrite['target_new']['str']
    
#     # Configure prompt
#     prompts = example['paraphrase_prompts'] + example['neighborhood_prompts'] + example['attribute_prompts'] + example['generation_prompts']
#     formatted_prompts = [
#         prompt_dict["prompt_counter_fact"].format(statement=prompt, target_new=target_new)
#         for prompt in prompts
#     ]

#     # Tokenize
#     input_ids = []
#     labels = []
#     for prompt in formatted_prompts:
#         prompt_tokens = tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {prompt.strip()} {E_INST}", add_special_tokens=False)
#         input_ids.append(prompt_tokens)
#         labels.append(len(prompt_tokens) * [-100])

#     combined_tokens = {
#         "input_ids": list(itertools.chain(*input_ids)),
#         "labels": list(itertools.chain(*labels)),
#         "attention_mask": [1] * len(list(itertools.chain(*input_ids)))
#     }
    
#     return combined_tokens

# def get_custom_dataset(dataset_config, tokenizer,split):
#     dataset_config = '/home/ljc/representation-engineering/llama-recipes/src/llama_recipes/datasets/couterdata/counterfact_first_100.json'
#     data = load_new_dataset(dataset_config)
#     dataset = Dataset.from_list(data)

#     PROMPT_DICT = {
#         "prompt_counter_fact": (
#             "Below is a statement with a fact. Answer the true fact based on the provided information.\n\n"
#             "### Statement:\n{statement}\n\n### Answer True Fact: {target_new}"
#         ),
#     }

#     formatted_dataset = dataset.map(lambda example: format_example(example, tokenizer, PROMPT_DICT), remove_columns=dataset.column_names)
    
#     return formatted_dataset

# # Example usage
# if __name__ == "__main__":
#     dataset_config = '/home/ljc/representation-engineering/llama-recipes/src/llama_recipes/datasets/couterdata/counterfact.json'
#     tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Update with your actual model

#     split = "train"  # or "test"
#     formatted_data = get_custom_dataset(dataset_config, tokenizer )

#     Now `formatted_data` can be used for training your model

# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# # For dataset details visit: https://huggingface.co/datasets/samsum

# import copy
# import datasets
# import itertools


# B_INST, E_INST = "[INST]", "[/INST]"

# def tokenize_dialog(dialog, tokenizer):
#     if tokenizer.vocab_size >= 128000:
#         dialog_tokens = tokenizer.apply_chat_template(dialog)
#         dialog_tokens = dialog_tokens[:-4] # Remove generation prompt <|start_header_id|>assistant<|end_header_id|>\n\n
#         eot_indices = [i for i,n in enumerate(dialog_tokens) if n == 128009]
#         labels = copy.copy(dialog_tokens)
#         last_idx = 0
#         for n, idx in enumerate(eot_indices):
#             if n % 2 == 1:
#                 last_idx = idx
#             else:
#                 labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)

#         dialog_tokens = [dialog_tokens]
#         labels_tokens = [labels]
#     else:
#         prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}", add_special_tokens=False) for prompt in dialog[::2]]
#         answer_tokens = [tokenizer.encode(f"{answer['content'].strip()} {tokenizer.eos_token}", add_special_tokens=False) for answer in dialog[1::2]]
#         dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))

#         #Add labels, convert prompt token to -100 in order to ignore in loss function
#         labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i,c in enumerate(dialog_tokens)]

#     combined_tokens = {
#         "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
#         "labels": list(itertools.chain(*(t for t in labels_tokens))),
#     }

#     return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


# def get_custom_dataset(dataset_config, tokenizer, split):
#     dataset = datasets.load_dataset("OpenAssistant/oasst1", split=split)

#     dataset = dataset.map(lambda sample: {
#         "message_id": sample["message_id"],
#         "parent_id": sample["parent_id"],
#         "text": sample["text"],
#         },
#         batched=True,
#         remove_columns=list(dataset.features),)

#     nodes = {}

#     messages = {}
#     root_ids = []

#     for data in dataset:
#         if data["parent_id"]:
#             nodes[data["parent_id"]] = nodes.get(data["parent_id"], []) + [data["message_id"]]
#         else:
#             root_ids.append(data["message_id"])
#         messages[data["message_id"]]=data["text"]

#     def follow(thread, current_id):
#         thread = copy.copy(thread) + [messages[current_id]]
#         if current_id in nodes:
#             new_threads = []
#             for next_id in nodes[current_id]:
#                 new_threads += follow(thread, next_id)
#             return new_threads
#         else:
#             return [thread]

#     def get_threads_from_root(root_id):
#         all_threads = []
#         thread = [messages[root_id]]
#         for cid in nodes[root_id]:
#             all_threads += follow(thread, cid)
#         return all_threads

#     dataset = dataset.filter(lambda x: x["message_id"] in root_ids)
#     dataset = dataset.map(lambda x: {"thread": get_threads_from_root(x["message_id"])}, remove_columns=list(dataset.features))
#     dataset = dataset.map(lambda x: {"thread": [i for row in x["thread"] for i in row]}, batched=True)

#     def to_dialog(thread):
#         dialog = []
#         for i, content in enumerate(thread):
#             dialog.append({
#                 "role": "user" if i % 2 == 0 else "assistant",
#                 "content": content,
#             })
#         return {"dialog": dialog}

#     dataset = dataset.map(lambda x: to_dialog(x["thread"]), remove_columns=list(dataset.features))
#     dataset = dataset.map(lambda x: tokenize_dialog(x["dialog"], tokenizer), remove_columns=list(dataset.features))

#     return dataset
