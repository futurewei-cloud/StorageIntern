import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, exceptions
import dotenv
from typing import Optional
from pydantic import BaseModel, validator
import logging
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
from components.data_disambiguation import DataDisambiguation
from components.unstructured_data_extractor import DataExtractor, DataExtractorWithSchema
from driver.neo4j import Neo4jDatabase
import torch
from components.transform import create_neo4j_input
import fitz  # PyMuPDF
from typing import List 
from generated_text import generate_text_for_long_article



# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImportPayload(BaseModel):
    input: str
    neo4j_schema: Optional[str] = None
    api_key: Optional[str] = None

    @validator('input')
    def input_must_be_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('input must be a non-empty string')
        return v

# Use Llama-3 Model 
tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained("unsloth/llama-3-8b-Instruct-bnb-4bit")

# Set the pad token ID
model.config.pad_token_id = tokenizer.eos_token_id


#generate prompt for input to llama
def generate_text(prompt: str, max_length: int = 6000, max_new_tokens: int = 1000) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id 
        )
        new_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input text from the newly generated text
        new_text_without_prompt = new_text[len(prompt):]
    
    return new_text_without_prompt




def process_paragraph(payload: ImportPayload):
    """
    Takes an input payload and processes it to extract and disambiguate nodes and relationships.
    """
    # try:
    llm = generate_text

    # if not payload.neo4j_schema:
    extractor = DataExtractor(llm=llm)
    result = extractor.run(data=payload.input)
    # else:
    #     extractor = DataExtractorWithSchema(llm=llm)
    #     result = extractor.run(schema=payload.neo4j_schema, data=payload.input)

    logger.info(f"Extracted result: {result}")
    print("Extracted result: " + str(result))

    #整理全部的段落的node和relationships，聚集results再处理DataDisambiguation
    # disambiguation = DataDisambiguation(llm=llm)
    # disambiguation_result = disambiguation.run(result)

    # logger.info(f"Disambiguation result: {disambiguation_result}")
    # print("Disambiguation result: " + str(disambiguation_result))

    # return {"data": disambiguation_result}

    return {"data": result}

    # except Exception as e:
    #     logger.error(f"Error processing paragraph: {e}")
    #     raise

def read_pdf(file_path: str) -> str:
    document = fitz.open(file_path)
    text = ""

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()

    return text

# Example usage
if __name__ == "__main__":
    file_path = '/home/angel/universe/NaLLM/dataset/Report_-SHEIN-Business-Breakdown-_-Founding-Story.txt'
    # Open the file in read mode
    with open(file_path, 'r') as file:
    # Read the content of the file
        paragraph = file.read()

    #Spilt paragraphs into chunks 
    # chunks = generate_text_for_long_article(paragraph)
    
    # Create payload and result 
    # for chunk in chunks: 
    payload = ImportPayload(input=paragraph)
    result = process_paragraph(payload)

    #connect to neo4j
    URI = "neo4j+s://fe00359c.databases.neo4j.io"
    AUTH = ("neo4j", "PeOMLGSq_Iohs8rsZn53Bz4zplGDUAHLmvaG61fZXUM")
    node_statements, relationship_statements = create_neo4j_input(result["data"])
       
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        for node_statement in node_statements:
            records, summary, keys = driver.execute_query(
                node_statement,
                database_="neo4j",
            )
        for relationship_statement in relationship_statements:
            
            records, summary, keys = driver.execute_query(
                relationship_statement,
                database_="neo4j",
            )
            # Loop through results and do something with them
            for record in records:
                print(record.data())  # obtain record as dict


            # Summary information
            print("The query `{query}` returned {records_count} records in {time} ms.".format(
                query=summary.query, records_count=len(records),
                time=summary.result_available_after
            ))    