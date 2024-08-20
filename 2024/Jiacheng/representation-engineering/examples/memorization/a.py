from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_interpret import QuestionAnsweringExplainer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.cls_token = tokenizer.eos_token
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

qa_explainer = QuestionAnsweringExplainer(
    model,
    tokenizer,
)

context = """
In Artificial Intelligence and machine learning, Natural Language Processing relates to the usage of machines to process and understand human language.
Many researchers currently work in this space.
"""

word_attributions = qa_explainer(
    "What is natural language processing ?",
    context,
    
)
qa_explainer.visualize("bert_qa_viz.html")