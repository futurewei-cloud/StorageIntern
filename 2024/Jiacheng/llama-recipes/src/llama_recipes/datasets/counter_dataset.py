import copy
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

PROMPT_DICT = {
    "prompt_counter_fact": (
        "Below is a statement with a fact. Determine the true fact based on the provided information.\n\n"
        "### Statement:\n{statement}\n\n### True Fact:"
    ),
}

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        self.ann = json.load(open(dataset_config['data_path']))
        # Use 5% of the dataset for evaluation
        eval_length = int(len(self.ann) / 20)
        if partition == "train":
            self.ann = self.ann[eval_length:]
        else:
            self.ann = self.ann[:eval_length]

        self.tokenizer = tokenizer

    def _configure_prompt(self, ann):
        # Select the prompt type and content as needed
        statement = ann["requested_rewrite"]["prompt"].format(ann["requested_rewrite"]["subject"], ann["requested_rewrite"]["target_new"]["str"])
        response = ann["requested_rewrite"]["target_true"]["str"]

        return statement, response

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]
        statement, response = self._configure_prompt(ann)

        full_prompt = PROMPT_DICT["prompt_counter_fact"].format(statement=statement)
        example = full_prompt + response
        prompt = torch.tensor(
            self.tokenizer.encode(full_prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": example_mask.tolist(),
        }

if __name__ == "__main__":
    # Configuration for the dataset
    dataset_config = {
        'data_path': '/home/ljc/representation-engineering/llama-recipes/src/llama_recipes/datasets/couterdata/counterfact.json'  # Replace with the actual path to your dataset
    }
    
    # Initialize the tokenizer, e.g., using a pretrained model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Replace with the actual tokenizer you want to use
    
    # Initialize the dataset
    train_dataset = InstructionDataset(dataset_config, tokenizer, partition="train")
    eval_dataset = InstructionDataset(dataset_config, tokenizer, partition="eval")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=2, shuffle=False)

    # Iterate over the dataset and print some sample outputs
    print("Training samples:")
    for i, batch in enumerate(train_loader):
        if i > 2:  # Print only the first few batches for brevity
            break
        print(batch)
    
    print("\nEvaluation samples:")
    for i, batch in enumerate(eval_loader):
        if i > 2:  # Print only the first few batches for brevity
            break
        print(batch)
