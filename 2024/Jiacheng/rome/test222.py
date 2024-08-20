import json
import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import os

# Set the environment variable to use only GPU 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# print(os.environ.get("CUDA_VISIBLE_DEVICES"))
# device="cpu"
# Load the JSON dataset
with open('dataset.json', 'r') as f:
    dataset = json.load(f)

task_prefix = "Here is a sentence, please extract the keyword of the sentence into (entity,entity,relation) format. "

# Shuffle the dataset
random.shuffle(dataset)

# Split the dataset into training and testing sets
train_data = dataset[:80]
test_data = dataset[80:]

# Function to write the datasets to files
def write_dataset_to_file(data, file_path):
    with open(file_path, 'w') as f:
        for entry in data:
            sentence = entry['sentence']
            output = " | ".join(entry['output'])
            f.write(f"{task_prefix} Sentence: {sentence} Output: {output}\n")

# Write the train and test datasets to respective files
write_dataset_to_file(train_data, 'train.txt')
write_dataset_to_file(test_data, 'test.txt')

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
tokenizer.pad_token = tokenizer.eos_token
# Function to create a dataset
def create_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

# Function to create a data collator
def create_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # Ensuring padding is applied properly
    )

# Load the datasets
train_dataset = create_dataset("train.txt", tokenizer)
test_dataset = create_dataset("test.txt", tokenizer)
data_collator = create_data_collator(tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    report_to=[],  # Disables wandb logging
    dataloader_pin_memory=False,  # To avoid potential CUDA memory issues
  
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./finetuned_gpt2')
tokenizer.save_pretrained('./finetuned_gpt2')

# Example function to generate relation extractions
def generate_relation_extraction(model, tokenizer, sentence):
    inputs = tokenizer.encode(sentence, return_tensors="pt", padding='longest', truncation=True).to(device)
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# Example usage
sentence = "Here is a sentence, please extract the keyword of the sentence into (entity,entity,relation) format. Sentence: Spotify launched a new music streaming service. Output:"
result = generate_relation_extraction(model, tokenizer, sentence)
print(result)
