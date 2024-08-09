import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
import fitz  # PyMuPDF
from typing import List 

# Constants
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True

# Function to read text from PDF using PyMuPDF
def read_pdf(file_path: str) -> str:
    document = fitz.open(file_path)
    text = ""

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()

    return text

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)
# Set the pad token ID only once
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.eos_token_id


# Apply PEFT (Parameter Efficient Fine-Tuning) configuration
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0. Suggested values: 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None, # And LoftQ
)

FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# Function to generate text from a given prompt
def generate_text(prompt: str, max_new_tokens: int = 6000) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)  # Adjust max_new_tokens as needed
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input text from the generated text
        generated_text_without_prompt = generated_text[len(prompt):]
        return generated_text_without_prompt

# Function to split text into paragraphs
def split_text_into_paragraphs(text: str) -> list:
    return text.split('\n')

# Function to group paragraphs into chunks of `group_size` paragraphs each
def group_paragraphs(paragraphs: list, group_size: int = 3) -> list:
    grouped_paragraphs = ['\n'.join(paragraphs[i:i + group_size]) for i in range(0, len(paragraphs), group_size)]
    return grouped_paragraphs

# Function to generate text for a long article by processing it in chunks of paragraphs
def generate_text_for_long_article(article: str, max_new_tokens: int = 6000) -> List[str]:
    paragraphs = split_text_into_paragraphs(article)
    chunks = group_paragraphs(paragraphs, group_size=3)
    # generated_chunks = []

    # for chunk in chunks:
    #     generated_text = generate_text(chunk, max_new_tokens=max_new_tokens)
    #     generated_chunks.append(generated_text.strip())
    
    return chunks


