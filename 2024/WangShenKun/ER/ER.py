import argparse
import pandas as pd
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5Model, GPT2Tokenizer, GPT2Model
import torch
from sklearn.metrics.pairwise import cosine_similarity
import json
from tabulate import tabulate

class TableConvertor:
    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.t5_model = T5Model.from_pretrained('t5-small')
        
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2Model.from_pretrained('gpt2')
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token

    def to_linearized_data(self, dataframe, table_name, primary_key=None, foreign_keys=None):
        header = dataframe.columns.tolist()
        rows = dataframe.values.tolist()
        
        linearized_data = f"Table: {table_name}\n"
        if primary_key:
            linearized_data += f"Primary Key: {primary_key}\n"
        if foreign_keys:
            linearized_data += f"Foreign Keys: {', '.join(foreign_keys)}\n"
        linearized_data += " | ".join(header) + "\n"
        for row in rows:
            linearized_data += " | ".join(map(str, row)) + "\n"
        return linearized_data

    def to_json_format(self, dataframe, table_name, primary_key=None, foreign_keys=None):
        data = dataframe.to_dict(orient="index")
        metadata = {
            "table_name": table_name,
            "primary_key": primary_key,
            "foreign_keys": foreign_keys
        }
        return json.dumps({"metadata": metadata, "data": data})

    def to_data_matrix_format(self, dataframe, table_name, primary_key=None, foreign_keys=None):
        header = dataframe.columns.tolist()
        rows = dataframe.values.tolist()
        data_matrix = [header] + rows
        metadata = {
            "table_name": table_name,
            "primary_key": primary_key,
            "foreign_keys": foreign_keys
        }
        return json.dumps({"metadata": metadata, "data": data_matrix})

    def to_comma_separated_format(self, dataframe, table_name, primary_key=None, foreign_keys=None):
        csv_data = dataframe.to_csv(index=False)
        metadata = f"Table: {table_name}\n"
        if primary_key:
            metadata += f"Primary Key: {primary_key}\n"
        if foreign_keys:
            metadata += f"Foreign Keys: {', '.join(foreign_keys)}\n"
        return metadata + csv_data

    def to_html_format(self, dataframe, table_name, primary_key=None, foreign_keys=None):
        html_data = dataframe.to_html(index=False)
        metadata = f"<h3>Table: {table_name}</h3>\n"
        if primary_key:
            metadata += f"<p>Primary Key: {primary_key}</p>\n"
        if foreign_keys:
            metadata += f"<p>Foreign Keys: {', '.join(foreign_keys)}</p>\n"
        return metadata + html_data

    def to_markdown_format(self, dataframe, table_name, primary_key=None, foreign_keys=None):
        markdown_data = dataframe.to_markdown(index=False)
        metadata = f"### Table: {table_name}\n"
        if primary_key:
            metadata += f"**Primary Key:** {primary_key}\n"
        if foreign_keys:
            metadata += f"**Foreign Keys:** {', '.join(foreign_keys)}\n"
        return metadata + "\n" + markdown_data

    def to_dfloader_format(self, dataframe, table_name, primary_key=None, foreign_keys=None):
        df_dict = dataframe.to_dict(orient='list')
        df_str = f"# Table: {table_name}\n"
        if primary_key:
            df_str += f"# Primary Key: {primary_key}\n"
        if foreign_keys:
            df_str += f"# Foreign Keys: {', '.join(foreign_keys)}\n"
        df_str += "pd.DataFrame({\n"
        for key, value in df_dict.items():
            df_str += f"    '{key}': {value},\n"
        df_str += "})"
        return df_str

    def get_embedding(self, text, model_name='bert'):
        if model_name == 'bert':
            tokenizer, model = self.bert_tokenizer, self.bert_model
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).detach().numpy()
        elif model_name == 't5':
            tokenizer, model = self.t5_tokenizer, self.t5_model
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = model.encoder(**inputs)
            return outputs.last_hidden_state.mean(dim=1).detach().numpy()
        elif model_name == 'gpt2':
            tokenizer, model = self.gpt2_tokenizer, self.gpt2_model
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def calculate_similarity(self, original_text, transformed_text, model_name='bert'):
        original_embedding = self.get_embedding(original_text, model_name=model_name)
        transformed_embedding = self.get_embedding(transformed_text, model_name=model_name)
        
        similarity = cosine_similarity(original_embedding, transformed_embedding)
        return similarity[0][0]

def main(customer_csv, book_csv, order_csv):
    customer_df = pd.read_csv(customer_csv)
    book_df = pd.read_csv(book_csv)
    order_df = pd.read_csv(order_csv)
    convertor = TableConvertor()
    
    formats = {
        "Linearized Data": convertor.to_linearized_data(customer_df, "Customer", primary_key="CustomerID") + 
                            convertor.to_linearized_data(book_df, "Book", primary_key="BookID") + 
                            convertor.to_linearized_data(order_df, "Order", primary_key="OrderID", foreign_keys=["CustomerID", "BookID"]),
        "JSON Format": convertor.to_json_format(customer_df, "Customer", primary_key="CustomerID") + 
                        convertor.to_json_format(book_df, "Book", primary_key="BookID") + 
                        convertor.to_json_format(order_df, "Order", primary_key="OrderID", foreign_keys=["CustomerID", "BookID"]),
        "Data-Matrix Format": convertor.to_data_matrix_format(customer_df, "Customer", primary_key="CustomerID") + 
                              convertor.to_data_matrix_format(book_df, "Book", primary_key="BookID") + 
                              convertor.to_data_matrix_format(order_df, "Order", primary_key="OrderID", foreign_keys=["CustomerID", "BookID"]),
        "Comma Separated Format": convertor.to_comma_separated_format(customer_df, "Customer", primary_key="CustomerID") + 
                                  convertor.to_comma_separated_format(book_df, "Book", primary_key="BookID") + 
                                  convertor.to_comma_separated_format(order_df, "Order", primary_key="OrderID", foreign_keys=["CustomerID", "BookID"]),
        "HTML Format": convertor.to_html_format(customer_df, "Customer", primary_key="CustomerID") + 
                       convertor.to_html_format(book_df, "Book", primary_key="BookID") + 
                       convertor.to_html_format(order_df, "Order", primary_key="OrderID", foreign_keys=["CustomerID", "BookID"]),
        "Markdown Format": convertor.to_markdown_format(customer_df, "Customer", primary_key="CustomerID") + 
                           convertor.to_markdown_format(book_df, "Book", primary_key="BookID") + 
                           convertor.to_markdown_format(order_df, "Order", primary_key="OrderID", foreign_keys=["CustomerID", "BookID"]),
        "DFLoader Format": convertor.to_dfloader_format(customer_df, "Customer", primary_key="CustomerID") + 
                           convertor.to_dfloader_format(book_df, "Book", primary_key="BookID") + 
                           convertor.to_dfloader_format(order_df, "Order", primary_key="OrderID", foreign_keys=["CustomerID", "BookID"])
    }

    original_text = f"Customer Table:\n{customer_df.to_string(index=False)}\n\n" \
                    f"Book Table:\n{book_df.to_string(index=False)}\n\n" \
                    f"Order Table:\n{order_df.to_string(index=False)}"
    
    print("Original Data:")
    print(original_text)
    similarity_scores = {'bert': {}, 't5': {}, 'gpt2': {}}
    for model_name in similarity_scores.keys():
        for format_name, transformed_data in formats.items():
            similarity_score = convertor.calculate_similarity(original_text, transformed_data, model_name=model_name)
            similarity_scores[model_name][format_name] = similarity_score
            if model_name == 'bert':  
                print(f"\n{format_name}:")
                print(transformed_data)
                print(f"\nSemantic Similarity ({model_name}): {similarity_score}\n")

    table_data = []
    headers = ["Format", "BERT Similarity", "T5 Similarity", "GPT2 Similarity"]


    all_formats = sorted(list(formats.keys()))

    for fmt in all_formats:
        bert_score = similarity_scores['bert'].get(fmt, None)
        t5_score = similarity_scores['t5'].get(fmt, None)
        gpt2_score = similarity_scores['gpt2'].get(fmt, None)
        table_data.append([fmt, f"{bert_score:.6f}", f"{t5_score:.6f}", f"{gpt2_score:.6f}"])


    print("\nSemantic Similarity Comparison (sorted by BERT similarity):")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--customer_csv", default="/home/shengkun/FW/wsk/FWProject/ER/Customer.csv", type=str, help="Path to the Customer CSV file")
    parser.add_argument("--book_csv", default="/home/shengkun/FW/wsk/FWProject/ER/Book.csv", type=str, help="Path to the Book CSV file")
    parser.add_argument("--order_csv", default="/home/shengkun/FW/wsk/FWProject/ER/Order_Updated.csv", type=str, help="Path to the Order CSV file")
    args = parser.parse_args()
    main(args.customer_csv, args.book_csv, args.order_csv)