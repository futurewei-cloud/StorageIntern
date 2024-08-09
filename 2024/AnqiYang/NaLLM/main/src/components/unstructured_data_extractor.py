import re
import os
from typing import List
import json

from components.base_component import BaseComponent
from llm.basellm import BaseLLM
from utils.unstructured_data_utils import (
    nodesTextToListOfDict,
    relationshipTextToListOfDict,
)
from unsloth import FastLanguageModel
from generated_text import generate_text_for_long_article

max_seq_length = 6000
dtype = None
load_in_4bit = True


def generate_system_message_with_schema() -> str:
    return """
    You are a data scientist working for a company that is building a graph database. Your task is to extract information from data and convert it into a graph database.
    Provide a set of Nodes in the form [ENTITY, TYPE, PROPERTIES] and a set of relationships in the form [ENTITY1, RELATIONSHIP, ENTITY2, PROPERTIES]. 
    Pay attention to the type of the properties, if you can't find data for a property set it to null. Don't make anything up and don't add any extra data. If you can't find any data for a node or relationship don't add it.
    Only add nodes and relationships that are part of the schema. If you don't get any relationships in the schema only add nodes.Make the nodes and relationship format a Python-compatible string.Format numeric values followed by strings as strings.

    Example1:
    Schema: Nodes: [Person {age: integer, name: string}] 
    Alice is 25 years old and Bob is her roommate.
    Nodes: [["Alice", "Person", {"age": 25, "name": "Alice}], ["Bob", "Person", {"name": "Bob"}]],["Jasper", "Company", {"founded": 2015, "valuation": "1.5 billion", "users": "100K"}]]
    Relationships: [["Alice", "roommate", "Bob",{}]]
    """


def generate_system_message() -> str:
    return """
    You are a data scientist working for a company that is building a graph database. Your task is to extract information from data and convert it into a graph database.
    Provide a set of Nodes in the form [ENTITY_ID, TYPE, PROPERTIES] and a set of relationships in the form [ENTITY_ID_1, RELATIONSHIP, ENTITY_ID_2, PROPERTIES].Make the nodes and relationship format a Python-compatible string.
    It is important that the ENTITY_ID_1 and ENTITY_ID_2 exists as nodes with a matching ENTITY_ID. If you can't pair a relationship with a pair of nodes don't add it.
    When you find a node or relationship you want to add try to create a generic TYPE for it that  describes the entity you can also think of it as a label.
    The eneity may include person, company, event, etc. 

    Example:
    Data: Alice lawyer and is 25 years old and Bob is her roommate since 2001. Bob works as a journalist. Alice owns a the webpage www.alice.com and Bob owns the webpage www.bob.com.
    Nodes: [["alice", "Person", {"age": 25, "occupation": "lawyer", "name":"Alice"}], ["bob", "Person", {"occupation": "journalist", "name": "Bob"}], ["alice.com", "Webpage", {"url": "www.alice.com"}], ["bob.com", "Webpage", {"url": "www.bob.com"}]]
    Relationships: [["alice", "roommate", "bob", {"start": 2021}], ["alice", "owns", "alice.com", {}], ["bob", "owns", "bob.com", {}]]
    
    Data:'Company's history goes back to a small ecommerce shop. There are conflicting reports on SHEIN's origins.
    Founded in 2012, the company's history goes back to a small ecommerce shop. It was launched by Chris Xu (CEO), an entrepreneur, and his ex-colleague, Wang Xiaohu, named Nanjing Dianwei Information Technology (NDIT) in 2008.
    Lily Peng, a part-time consultant, and known parent who is described as a "hardworking SEO whiz," is also known about the entrepreneur. Some reports describe Xu foucused on technical parts while leaving business development, finance, and corporate functions to Xiaohu and Peng.
    Xu, who studied at George Washington University, is described by some sources as a Chinese-American.'
    Nodes: [["chris_xu", "Person", {"name": "Chris Xu"}], ["wang_xiaohu", "Person", {"name": "Wang Xiaohu"}], ["SHEIN", "Company", {"name": "SHEIN"}], ["lily_peng", "Person", {"name": "Lily Peng"}], ["tidn", "Company", {"name": "TIDN"}], ["nanjing_information_technology", "Company", {"name": "Nanjing Information Technology"}]]
    Relationships: [["chris_xu", "cofounded_with", "wang_xiaohu", {}], ["chris_xu", "founded", "SHEIN", {}], ["Chris Xu", "focused on", "technical parts ", {}], ["lily_peng", "focused on", "business and finance part", {}]]
    """




def generate_system_message_with_labels() -> str:
    return """
    You are a data scientist working for a company that is building a graph database. Your task is to extract information from data and convert it into a graph database.
    Provide a set of Nodesin the form [ENTITY_ID, TYPE, PROPERTIES] and a set of relationships in the form [ENTITY_ID_1, RELATIONSHIP, ENTITY_ID_2, PROPERTIES].
    It is important that the ENTITY_ID_1 and ENTITY_ID_2 exists as nodes with a matching ENTITY_ID. If you can't pair a relationship with a pair of nodes don't add it.
    When you find a node or relationship you want to add try to create a generic TYPE for it that  describes the entity you can also think of it as a label.
    You will be given a list of types that you should try to use when creating the TYPE for a node. If you can't find a type that fits the node you can create a new one.
    Format numeric values followed by strings as strings.Make the nodes and relationship format a Python-compatible string.
    Example:
    Data: Alice lawyer and is 25 years old and Bob is her roommate since 2001. Bob works as a journalist. Alice owns a the webpage www.alice.com and Bob owns the webpage www.bob.com.
    Types: ["Person", "Webpage"]
    Nodes: [["alice", "Person", {"age": 25, "occupation": "lawyer", "name":"Alice"}], ["bob", "Person", {"occupation": "journalist", "name": "Bob"}], ["alice.com", "Webpage", {"url": "www.alice.com"}], ["bob.com", "Webpage", {"url": "www.bob.com"}]]
    Relationships: [["alice", "roommate", "bob", {"start": 2021}], ["alice", "owns", "alice.com", {}], ["bob", "owns", "bob.com", {}],
    ["Jasper","is", "Company", {"founded": 2015, "valuation": “1.5 billion”, "users": "100K"}]]
    """


def generate_prompt(data) -> str:
    return f"""
Data: {data}"""


def generate_prompt_with_schema(data, schema) -> str:
    return f"""
Schema: {schema}
Data: {data}"""


def generate_prompt_with_labels(data, labels) -> str:
    return f"""
Data: {data}
Types: {labels}"""


def splitString(string, max_length) -> List[str]:
    return [string[i : i + max_length] for i in range(0, len(string), max_length)]


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

def num_tokens_from_string(string: str) -> int:
    num_tokens = len(tokenizer.encode(string))
    return num_tokens

def max_allowed_token_length() -> int:
        # TODO: list all models and their max tokens from api
        return 6000

def splitStringToFitTokenSpace(string: str, token_use_per_string: int) -> List[str]:
    allowed_tokens = max_allowed_token_length() - token_use_per_string
    chunked_data = splitString(string, 6000)
    combined_chunks = []
    current_chunk = ""
    for chunk in chunked_data:
        if num_tokens_from_string(current_chunk) + num_tokens_from_string(chunk) < allowed_tokens:
            current_chunk += chunk
        else:
            combined_chunks.append(current_chunk)
            current_chunk = chunk
    combined_chunks.append(current_chunk)
    return combined_chunks

# def getNodesAndRelationshipsFromResult(result):
#     # regex = "Nodes:\s+(.*?)\s?\s?Relationships:\s?\s?(.*)"
#     # internalRegex = "\[(.*?)\]"
#     # regex = r"Nodes:\s*\[(.*?)\]\s*Relationships:\s*\[(.*?)\]"
#     # internalRegex = r"\[(.*?)\]"
#     # 正则匹配模式
#     nodes_pattern = r'Nodes:\s*(\[\[.*?\]\])'
#     relationships_pattern = r'Relationships:\s*(\[\[.*?\]\])'


#     nodes = []
#     relationships = []
#     for row in result:
#          # Replace null with None
#         row = row.replace('null', 'None')
#         # 提取Nodes和Relationships部分
#         nodes_match = re.search(nodes_pattern, row, re.DOTALL)
#         relationships_match = re.search(relationships_pattern, row, re.DOTALL)
#         # 转换为Python对象
#         nodes.extend(eval(nodes_match.group(1)) if nodes_match else [])
#         relationships.extend(eval(relationships_match.group(1)) if relationships_match else [])
#         # parsing = re.match(regex, row, flags=re.S)
#         # if nodes == None:
#         #     continue
#         # rawNodes = str(parsing.group(1))
#         # rawRelationships = parsing.group(2)
#         # nodes.extend(re.findall(internalRegex, rawNodes))
#         # relationships.extend(re.findall(internalRegex, rawRelationships))
#     # Convert nodes and relationships to the desired format
#     formatted_nodes = [{"name": node[0], "label": node[1], "properties": node[2]} for node in nodes]
#     formatted_relationships = [{"start": relationship[0], "end": relationship[1], "type": relationship[2], "properties": relationship[3]} for relationship in relationships]

#     result = {
#         "nodes": formatted_nodes,
#         "relationships": formatted_relationships
#     }
    
#     return result

def clean_json_string(json_str):
    # Remove all newlines and extra spaces
    json_str = re.sub(r'\s+', ' ', json_str)
    # Ensure proper JSON format for empty objects
    json_str = json_str.replace('None', 'null')
    # Ensure there are no trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    return json_str

def getNodesAndRelationshipsFromResult(result):
    nodes_pattern = r'Nodes:\s*(\[\[.*?\]\])'
    relationships_pattern = r'Relationships:\s*(\[\[.*?\]\])'

    nodes = []
    relationships = []

    result = ' '.join(result)
    # Replace null with None
    result = result.replace('null', 'None')
    
    # Extract all Nodes and Relationships parts
    nodes_matches = re.findall(nodes_pattern, result, re.DOTALL)
    relationships_matches = re.findall(relationships_pattern, result, re.DOTALL)
    
    # Convert to Python objects
    for nodes_match in nodes_matches:
        nodes.extend(eval(nodes_match))
    
    for relationships_match in relationships_matches:
        # Fixing the nested structure issue manually
        relationships_match = relationships_match.replace('[[', '[').replace(']]', ']')
        relationships_match = re.sub(r'\]\s*,\s*\[', '],[', relationships_match)      
        # Split the match into individual relationship entries
        relationship_entries = relationships_match.split('],[')
        relationship_entries = [entry.strip('[]') for entry in relationship_entries]

        # Clean and convert to Python objects using JSON
        for entry in relationship_entries:
            clean_entry = clean_json_string(f'[{entry}]')
            relationships.append(json.loads(clean_entry))
          

    # Convert nodes and relationships to the desired format
    formatted_nodes = [{"name": node[0], "label": node[1], "properties": node[2]} for node in nodes]
    formatted_relationships = [{"start": relationship[0], "end": relationship[1], "type": relationship[2], "properties": relationship[3]} for relationship in relationships]

    result = {
        "nodes": formatted_nodes,
        "relationships": formatted_relationships
    }

    return result

    # result = dict()
    # result["nodes"] = []
    # result["relationships"] = []
    # result["nodes"].extend(nodes)
    # result["relationships"].extend(relationships)
    # return result


class 
DataExtractor(BaseComponent):

    def __init__(self,llm) -> None:
        self.llm = llm

    def process(self, chunk):
        # messages = [
        #     {"role": "system", "content": generate_system_message()},
        #     {"role": "user", "content": generate_prompt(chunk)},
        # ]
        # print(messages)
        messages= generate_system_message() + generate_prompt(chunk)
        output = self.llm(messages)
        return output

    def process_with_labels(self, chunk, labels):
        # messages = [
        #     {"role": "system", "content": generate_system_message_with_schema()},
        #     {"role": "user", "content": generate_prompt_with_labels(chunk, labels)},
        # ]
        messages = generate_system_message_with_schema()+ generate_prompt_with_labels(chunk, labels)
        print(messages)
        output = self.llm(messages)
        return output


    def addnode(self, chunkResult):
        nodes = chunkResult["nodes"]
        relationships = chunkResult["relationships"]

        # 提取 relationships 中的 start 和 type，并创建 extra_node
        extra_nodes = []
        updated_relationships = []

        for rel in relationships:
            # 更新 rel 中的 start, end 和 type
            rel["start"] = rel["start"].replace(".", "_").replace(" ", "_").replace("-", "_").replace("&", "_")
            rel["end"] = rel["end"].replace(".", "_").replace(" ", "_").replace("-", "_").replace("&", "_")
            rel["type"] = rel["type"].replace(".", "_").replace(" ", "_").replace("-", "_").replace("&", "_")
            rel["type"] = f"NUM_{rel['type']}" if rel["type"][0].isdigit() else rel["type"]
            
            updated_relationships.append(rel)
            
            extra_node_start = {
                "name": rel["start"],
                "label": "node",
                "properties": {'name': rel["start"]}
            }

            # 检查 extra_node 是否已经存在于 nodes 中
            if not any(node['name'] == extra_node_start['name'] for node in nodes):
                extra_nodes.append(extra_node_start)

            extra_node_type = {
                "name": rel["type"],
                "label": "node",
                "properties": {'name': rel["type"]}
            }

            # 检查 extra_node 是否已经存在于 nodes 中
            if not any(node['name'] == extra_node_type['name'] for node in nodes):
                extra_nodes.append(extra_node_type)

        # 将 extra_nodes 插入到 nodes 中
        nodes.extend(extra_nodes)

        # 更新 chunkResult 中的 nodes 和 relationships
        chunkResult["nodes"] = nodes
        chunkResult["relationships"] = updated_relationships

        # 更新原始节点名，添加 NUM_ 前缀
        for node in nodes:
            node["name"] = f"NUM_{node['name']}" if node["name"][0].isdigit() else node["name"]

        return chunkResult



    def run(self, data: str) -> List[str]:
        system_message = generate_system_message()
        prompt_string = generate_prompt("")
        token_usage_per_prompt = num_tokens_from_string(
            system_message + prompt_string
        )
        chunked_data = generate_text_for_long_article(data)
        # chunked_data = splitStringToFitTokenSpace(string=data, token_use_per_string=token_usage_per_prompt)

        results = []
        labels = set()
        print("Starting chunked processing")
        merged_result = {
        'nodes': [],
        'relationships': []
        }
    


        for chunk in chunked_data:
            proceededChunk = self.process(chunk)
            print("proceededChunk", proceededChunk)
            chunkResult = getNodesAndRelationshipsFromResult([proceededChunk])
            print("chunkResult", chunkResult)
           
            newLabels = [node['label'] for node in chunkResult["nodes"]]
            print("newLabels", newLabels)

            merged_result['nodes'].extend(chunkResult['nodes'])
            merged_result['relationships'].extend(chunkResult['relationships'])
            
            # results.append(chunkResult)
            labels.update(newLabels)
        merged_result = self.addnode(merged_result)

        return merged_result


class DataExtractorWithSchema(BaseComponent):
    def __init__(self,llm) -> None:
        self.llm = llm

    def run(self, data: str, schema: str) -> List[str]:
        system_message = generate_system_message_with_schema()
        prompt_string = (
            generate_system_message_with_schema()
            + generate_prompt_with_schema(schema=schema, data="")
        )
        token_usage_per_prompt = num_tokens_from_string(
            system_message + prompt_string
        )

        chunked_data = splitStringToFitTokenSpace(
            llm=self.llm, string=data, token_use_per_string=token_usage_per_prompt
        )
        result = []
        print("Starting chunked processing")

        for chunk in chunked_data:
            print("prompt", generate_prompt_with_schema(chunk, schema))
            messages = system_message + generate_prompt_with_schema(chunk, schema)
            
            output = self.llm(messages)
            result.append(output)
        return getNodesAndRelationshipsFromResult(result)
