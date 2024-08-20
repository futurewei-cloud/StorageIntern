from repe import repe_pipeline_registry # register 'rep-reading' and 'rep-control' tasks into Hugging Face pipelines
repe_pipeline_registry()
from transformers import pipeline
# ... initializing model and tokenizer ....
# import transformers
# import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model="meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer="meta-llama/Meta-Llama-3-8B-Instruct"
# pipeline = transformers.pipeline(
#   "text-generation",
#   model="meta-llama/Meta-Llama-3-8B-Instruct",
#   model_kwargs={"torch_dtype": torch.bfloat16},
#   device="cuda",
# )
rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer)
rep_control_pipeline =  pipeline("rep-control", model=model, tokenizer=tokenizer, **control_kwargs)