import argparse
import pandas as pd
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5EncoderModel, GPT2Tokenizer, GPT2Model
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import json
from tabulate import tabulate
import fitz  

class DocumentConvertor:
    def __init__(self):
        self.models = {
            'bert-base': (BertTokenizer.from_pretrained('bert-base-uncased'), BertModel.from_pretrained('bert-base-uncased')),
            'bert-large': (BertTokenizer.from_pretrained('bert-large-uncased'), BertModel.from_pretrained('bert-large-uncased')),
            't5-small': (T5Tokenizer.from_pretrained('t5-small'), T5EncoderModel.from_pretrained('t5-small')),
            't5-large': (T5Tokenizer.from_pretrained('t5-large'), T5EncoderModel.from_pretrained('t5-large')),
            'gpt2': (GPT2Tokenizer.from_pretrained('gpt2'), GPT2Model.from_pretrained('gpt2')),
            'gpt2-medium': (GPT2Tokenizer.from_pretrained('gpt2-medium'), GPT2Model.from_pretrained('gpt2-medium'))
        }
        for name, (tokenizer, model) in self.models.items():
            if name.startswith('gpt2'):
                tokenizer.pad_token = tokenizer.eos_token

    def read_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def to_linearized_data(self, dataframe):
        header = [str(h) for h in dataframe.columns.tolist()]
        rows = dataframe.values.tolist()
        
        linearized_data = " | ".join(header) + "\n"
        for row in rows:
            linearized_data += " | ".join(map(str, row)) + "\n"
        return linearized_data

    def to_json_format(self, dataframe):
        return dataframe.to_json(orient="index")

    def to_data_matrix_format(self, dataframe):
        header = [str(h) for h in dataframe.columns.tolist()]
        rows = dataframe.values.tolist()
        data_matrix = [header] + rows
        return json.dumps(data_matrix)

    def to_comma_separated_format(self, dataframe):
        return dataframe.to_csv(index=True)

    def to_html_format(self, dataframe):
        return dataframe.to_html(index=True)

    def to_markdown_format(self, dataframe):
        return dataframe.to_markdown(index=True)

    def to_dfloader_format(self, dataframe):
        df_dict = dataframe.to_dict(orient='list')
        df_str = "pd.DataFrame({\n"
        for key, value in df_dict.items():
            df_str += f"    '{key}': {value},\n"
        df_str += "}, index=" + str(list(dataframe.index)) + ")"
        return df_str

    def get_embedding(self, text, model_name='bert-base'):
        tokenizer, model = self.models[model_name]
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='longest', max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        if 'bert' in model_name or 't5' in model_name:
            return outputs.last_hidden_state.mean(dim=1).detach().numpy()
        elif 'gpt2' in model_name:
            return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def calculate_similarity(self, original_text, transformed_text, model_name='bert-base'):
        original_embedding = self.get_embedding(original_text, model_name=model_name)
        transformed_embedding = self.get_embedding(transformed_text, model_name=model_name)
        
        original_embedding = normalize(original_embedding)
        transformed_embedding = normalize(transformed_embedding)

        similarity = cosine_similarity(original_embedding, transformed_embedding)
        return similarity[0][0]

def main(pdf_path):
    convertor = DocumentConvertor()
    
    pdf_text = convertor.read_pdf(pdf_path)
    
    rows = pdf_text.split('\n')
    data = [row.split() for row in rows if row]
    df = pd.DataFrame(data)

    formats = {
        "Linearized Data": convertor.to_linearized_data(df),
        "JSON Format": convertor.to_json_format(df),
        "Data-Matrix Format": convertor.to_data_matrix_format(df),
        "Comma Separated Format": convertor.to_comma_separated_format(df),
        "HTML Format": convertor.to_html_format(df),
        "Markdown Format": convertor.to_markdown_format(df),
        "DFLoader Format": convertor.to_dfloader_format(df)
    }

    original_text = pdf_text
    
    print("Original Data:")
    print(original_text)


    model_names = ['bert-base', 'bert-large', 't5-small', 't5-large', 'gpt2', 'gpt2-medium']
    
    similarity_scores = {model_name: {} for model_name in model_names}
    for model_name in model_names:
        for format_name, transformed_data in formats.items():
            similarity_score = convertor.calculate_similarity(original_text, transformed_data, model_name=model_name)
            similarity_scores[model_name][format_name] = similarity_score
            if model_name == 'bert-base':  
                print(f"\n{format_name}:")
                print(transformed_data)
                print(f"\nSemantic Similarity ({model_name}): {similarity_score}\n")

    table_data = []
    headers = ["Format"] + [f"{model_name} Similarity" for model_name in model_names]

    all_formats = sorted(list(formats.keys()))

    for fmt in all_formats:
        row = [fmt]
        for model_name in model_names:
            score = similarity_scores[model_name].get(fmt, None)
            row.append(f"{score:.6f}")
        table_data.append(row)

    print("\nSemantic Similarity Comparison (sorted by BERT similarity):")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", default="/home/shengkun/FW/wsk/FWProject/PDF/huawei_fact_sheet_en.pdf", type=str, help="Path to the PDF file to be converted")
    args = parser.parse_args()
    main(args.pdf_path)
