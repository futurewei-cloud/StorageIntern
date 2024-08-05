import argparse
import pandas as pd
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5EncoderModel, GPT2Tokenizer, GPT2Model, RobertaTokenizer, RobertaModel, ElectraTokenizer, ElectraModel, XLNetTokenizer, XLNetModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import json
from tabulate import tabulate
import fitz  

class DocumentConvertor:
    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.t5_model = T5EncoderModel.from_pretrained('t5-small')
        
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2Model.from_pretrained('gpt2')
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')

        self.electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        self.electra_model = ElectraModel.from_pretrained('google/electra-small-discriminator')

        self.xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.xlnet_model = XLNetModel.from_pretrained('xlnet-base-cased')

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

    def get_embedding(self, text, model_name='bert'):
        if model_name == 'bert':
            tokenizer, model = self.bert_tokenizer, self.bert_model
        elif model_name == 't5':
            tokenizer, model = self.t5_tokenizer, self.t5_model
        elif model_name == 'gpt2':
            tokenizer, model = self.gpt2_tokenizer, self.gpt2_model
        elif model_name == 'roberta':
            tokenizer, model = self.roberta_tokenizer, self.roberta_model
        elif model_name == 'electra':
            tokenizer, model = self.electra_tokenizer, self.electra_model
        elif model_name == 'xlnet':
            tokenizer, model = self.xlnet_tokenizer, self.xlnet_model
        else:
            raise ValueError("Unsupported model name")

        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='longest', max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        if model_name in ['bert', 'roberta', 'electra', 'xlnet']:
            return outputs.last_hidden_state.mean(dim=1).detach().numpy()
        elif model_name == 't5':
            return outputs.last_hidden_state.mean(dim=1).detach().numpy()
        elif model_name == 'gpt2':
            return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def calculate_similarity(self, original_text, transformed_text, model_name='bert'):
        original_embedding = self.get_embedding(original_text, model_name=model_name)
        transformed_embedding = self.get_embedding(transformed_text, model_name=model_name)
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


    similarity_scores = {'bert': {}, 't5': {}, 'gpt2': {}, 'roberta': {}, 'electra': {}, 'xlnet': {}}
    for model_name in similarity_scores.keys():
        for format_name, transformed_data in formats.items():
            similarity_score = convertor.calculate_similarity(original_text, transformed_data, model_name=model_name)
            similarity_scores[model_name][format_name] = similarity_score
            if model_name == 'bert':  
                print(f"\n{format_name}:")
                print(transformed_data)
                print(f"\nSemantic Similarity ({model_name}): {similarity_score}\n")


    table_data = []
    headers = ["Format", "BERT Similarity", "T5 Similarity", "GPT2 Similarity", "RoBERTa Similarity", "Electra Similarity", "XLNet Similarity"]


    all_formats = sorted(list(formats.keys()))

    for fmt in all_formats:
        bert_score = similarity_scores['bert'].get(fmt, None)
        t5_score = similarity_scores['t5'].get(fmt, None)
        gpt2_score = similarity_scores['gpt2'].get(fmt, None)
        roberta_score = similarity_scores['roberta'].get(fmt, None)
        electra_score = similarity_scores['electra'].get(fmt, None)
        xlnet_score = similarity_scores['xlnet'].get(fmt, None)
        table_data.append([fmt, f"{bert_score:.6f}", f"{t5_score:.6f}", f"{gpt2_score:.6f}", f"{roberta_score:.6f}", f"{electra_score:.6f}", f"{xlnet_score:.6f}"])

    print("\nSemantic Similarity Comparison (sorted by BERT similarity):")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", default="/home/shengkun/FW/wsk/FWProject/PDF/huawei_fact_sheet_en.pdf", type=str, help="Path to the PDF file to be converted")
    args = parser.parse_args()
    main(args.pdf_path)
