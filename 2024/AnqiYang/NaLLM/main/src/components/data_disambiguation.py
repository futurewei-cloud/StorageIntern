import json
import re
from itertools import groupby
from components.base_component import BaseComponent
from utils.unstructured_data_utils import (
    nodesTextToListOfDict,
    relationshipTextToListOfDict,
)
import json
from collections import defaultdict

def generate_system_message_for_nodes() -> str:
    return """Your task is to identify if there are duplicated nodes and if so merge them into one nod. Only merge the nodes that refer to the same entity.
    You will be given different datasets of nodes and some of these nodes may be duplicated or refer to the same entity. 
    The datasets contains nodes in the form [ENTITY_ID, TYPE, PROPERTIES]. When you have completed your task please give me the 
    resulting nodes in the same format. Only return the nodes and relationships no other text. If there is no duplicated nodes return the original nodes.

    Here is an example of the text you will be given:

    text = NVIDIA Announces Financial Results for Fourth Quarter and Fiscal 2023. Quarterly revenue of $6.05 billion, down 21% from a year ago
    Fiscal-year revenue of $27.0 billion, flat from a year ago.
    Here is an example of the input you will be given:
    ["alice", "Person", {"age": 25, "occupation": "lawyer", "name":"Alice"}], ["bob", "Person", {"occupation": "journalist", "name": "Bob"}], ["alice.com", "Webpage", {"url": "www.alice.com"}], ["bob.com", "Webpage", {"url": "www.bob.com"}]
    ['nvidia', 'Company',  {'name': 'NVIDIA'}}, {'revenue', 'FinancialTerm', 'PROPERTIES': {'name': 'Revenue'}}, {'fiscal_year', 'TimePeriod', 'PROPERTIES': {'name': 'Fiscal Year'}}]
    format should be {"founded": 2015, "valuation": "1.5 billion", "users": "100K"}]
    """
# def merge_duplicate_nodes(nodes,relationships):
    
#     """
#     Identifies duplicate nodes and merges them if they refer to the same entity.
    
#     Args:
#     nodes (list): List of nodes in the form [ENTITY_ID, TYPE, PROPERTIES].
    
#     Returns:
#     list: List of merged nodes.
#     """
#     # Group nodes by ENTITY_ID
#     grouped_nodes = defaultdict(list)
#     for node in nodes:
#         grouped_nodes[node[0]].append(node)
    
#     # Merge nodes that refer to the same entity
#     merged_nodes = []
#     for entity_id, node_list in grouped_nodes.items():
#         if len(node_list) > 1:
#             # Merge properties
#             merged_properties = {}
#             for node in node_list:
#                 merged_properties.update(node[2])
#             # Take the first type found (assuming same type for duplicates)
#             merged_nodes.append([entity_id, node_list[0][1], merged_properties])
#         else:
#             merged_nodes.append(node_list[0])
    
#     return merged_nodes




# def process_relationships(relationships, valid_ids):
#     """
#     Processes the relationships to ensure they make sense, merges duplicates, and replaces invalid ENTITY_IDs.
    
#     Args:
#     relationships (list): List of relationships in the form [ENTITY_ID_1, RELATIONSHIP, ENTITY_ID_2, PROPERTIES].
#     valid_ids (set): Set of valid ENTITY_IDs.
    
#     Returns:
#     list: List of valid relationships.
#     """
    

def generate_system_message_for_relationships() -> str:
    return """
    Your task is to identify if a set of relationships make sense.
    If they do not make sense please remove them from the dataset.
    Some relationships may be duplicated or refer to the same entity. 
    Please merge relationships that refer to the same entity.
    The datasets contains relationships in the form [ENTITY_ID_1, RELATIONSHIP, ENTITY_ID_2, PROPERTIES].
    You will also be given a set of ENTITY_IDs that are valid.
    Some relationships may use ENTITY_IDs that are not in the valid set but refer to a entity in the valid set.
    If a relationships refer to a ENTITY_ID in the valid set please change the ID so it matches the valid ID.
    When you have completed your task please give me the valid relationships in the same format. Only return the relationships no other text.

    Here is an example of the input you will be given:

    text = NVIDIA Announces Financial Results for Fourth Quarter and Fiscal 2023. Quarterly revenue of $6.05 billion, down 21% from a year ago
    Fiscal-year revenue of $27.0 billion, flat from a year ago.

    Here is an example of the input you will be given:
    ["nvidia", "Announces", "Financial Results", {"time": Fourth Quarter and Fiscal 2023}],["nvidia", "went down", "revenue", {amount: 21% from a year ago}]
    
    """
def merge_duplicate_nodes(nodes):
    """
    Identifies duplicate nodes and merges them if they refer to the same entity.
    
    Args:
    nodes (list): List of nodes in the form [{'name': ENTITY_ID, 'label': TYPE, 'properties': PROPERTIES}].
    
    Returns:
    list: List of merged nodes.
    """
    # Group nodes by ENTITY_ID
    grouped_nodes = defaultdict(list)
    for node in nodes:
        grouped_nodes[node['name']].append(node)
    

    
    # Merge nodes that refer to the same entity
    merged_nodes = []
    for entity_id, node_list in grouped_nodes.items():
        if len(node_list) > 1:
            # Merge properties
            merged_properties = {}
            for node in node_list:
                merged_properties.update(node['properties'])
            # Take the first type found (assuming same type for duplicates)
            merged_nodes.append({'name': entity_id, 'label': node_list[0]['label'], 'properties': merged_properties})
        else:
            merged_nodes.append(node_list[0])
    
    return merged_nodes

def process_relationships(relationships, valid_ids):
    """
    Processes the relationships to ensure they make sense, merges duplicates, and replaces invalid ENTITY_IDs.
    
    Args:
    relationships (list): List of relationships in the form {'start': ..., 'end': ..., 'type': ..., 'properties': ...}.
    valid_ids (set): Set of valid ENTITY_IDs.
    
    Returns:
    list: List of valid relationships.
    """
    # Function to find the valid ENTITY_ID
    def find_valid_entity_id(entity_id):
        for valid_id in valid_ids:
            if entity_id == valid_id or entity_id.startswith(valid_id):
                return valid_id
        return None

    # Dictionary to store relationships grouped by (start, type, end)
    grouped_relationships = defaultdict(list)

    for relationship in relationships:
        start, end, rel_type, props = relationship['start'], relationship['end'], relationship['type'], relationship['properties']

        # Replace invalid ENTITY_IDs with valid ones
        valid_start = find_valid_entity_id(start)
        rel_type = find_valid_entity_id(rel_type)

        if valid_start and rel_type:
            key = (valid_start, rel_type, end)
            grouped_relationships[key].append(props)

    # Merge properties of duplicated relationships
    merged_relationships = []
    for (start, rel_type, end), props_list in grouped_relationships.items():
        merged_properties = {}
        for props in props_list:
            merged_properties.update(props)
        merged_relationships.append({'start': start, 'end': end, 'type': rel_type, 'properties': merged_properties})

    return merged_relationships



def generate_prompt(data) -> str:
    return f""" Here is the data:
{data}
"""


internalRegex = "\[(.*?)\]"


class DataDisambiguation(BaseComponent):
    def __init__(self, llm) -> None:
        self.llm = llm
        
    def run(self, data: dict) -> str:
        nodes = sorted(data["nodes"], key=lambda x: x["label"])
        relationships = data["relationships"]
        new_nodes = []
        new_relationships = []
        #relationships 中的node都应该在new_nodes中
        node_groups = groupby(nodes, lambda x: x["label"])
        for group in node_groups:
            disString = ""
            nodes_in_group = list(group[1])
            if len(nodes_in_group) == 1:
                new_nodes.extend(nodes_in_group)
                continue
            new_nodes.extend(merge_duplicate_nodes(nodes_in_group))
            # for node in nodes_in_group:
            #     disString += (
            #         '["'
            #         + node["name"]
            #         + '", "'
            #         + node["label"]
            #         + '", '
            #         + json.dumps(node["properties"])
            #         + "]\n"
            #     )

            # # messages = [
            # #     {"role": "system", "content": generate_system_message_for_nodes()},
            # #     {"role": "user", "content": generate_prompt(disString)},
            # # ]
            # messages = generate_system_message_for_nodes() + generate_prompt(disString)
            # rawNodes = self.llm(messages)

            # n = re.findall(internalRegex, rawNodes)

            # new_nodes.extend(merge_duplicate_nodes())

        # relationship_data = "Relationships:\n"
        # for relation in relationships:
        #     relationship_data += (
        #         '["'
        #         + relation["start"]
        #         + '", "'
        #         + relation["type"]
        #         + '", "'
        #         + relation["end"]
        #         + '", '
        #         + json.dumps(relation["properties"])
        #         + "]\n"
        #     )

        node_labels = [node["name"] for node in new_nodes]
        # relationship_data += "Valid Nodes:\n" + "\n".join(node_labels)

        # # messages = [
        # #     {
        # #         "role": "system",
        # #         "content": generate_system_message_for_relationships(),
        # #     },
        # #     {"role": "user", "content": generate_prompt(relationship_data)},
        # # ]
        # messages = generate_system_message_for_relationships() + generate_prompt(relationship_data)
        # rawRelationships = self.llm(messages)
        # rels = re.findall(internalRegex, rawRelationships)
        new_relationships.extend(process_relationships(relationships,node_labels))
        return {"nodes": new_nodes, "relationships": new_relationships}
