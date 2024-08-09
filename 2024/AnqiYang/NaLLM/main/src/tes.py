def addnode(self, chunkResult):
    nodes = chunkResult["nodes"]
    relationships = chunkResult["relationships"]

    # 提取 relationships 中的 start 和 type，并创建 extra_node
    extra_nodes = []
    updated_relationships = []

    for rel in relationships:
        # 更新 rel 中的 start, end 和 type
        rel["start"] = rel["start"].replace(".", "_").replace(" ", "_").replace("-", "_")
        rel["end"] = rel["end"].replace(".", "_").replace(" ", "_").replace("-", "_")
        rel["type"] = rel["type"].replace(".", "_").replace(" ", "_").replace("-", "_")
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



# Test cases
chunkResult = {
    'nodes': [
        {'name': 'nvidia', 'label': 'Company', 'properties': {'name': 'NVIDIA'}},
        {'name': 'jensen_huang', 'label': 'Person', 'properties': {'name': 'Jensen Huang'}},
        {'name': 'nvidia', 'label': 'Company', 'properties': {'name': 'NVIDIA'}},
        {'name': 'financial_results', 'label': 'node', 'properties': {'name': 'financial_results'}},
        {'name': 'cloud_service_providers', 'label': 'node', 'properties': {'name': 'cloud_service_providers'}},
        {'name': '1_15_billion', 'label': 'node', 'properties': {'name': '1.15 billion'}},
        {'name': '0_04', 'label': 'node', 'properties': {'name': '0.04'}},
        {'name': 'quarter-over-quarter_improvement', 'label': 'node', 'properties': {'name': 'quarter-over-quarter improvement'}},
        {'name': 'strong_fiscal_year_revenue', 'label': 'node', 'properties': {'name': 'strong fiscal year revenue'}},
        {'name': 'significant_advancements_in_AI', 'label': 'node', 'properties': {'name': 'significant advancements in AI'}},
        {'name': 'strategic_partnerships', 'label': 'node', 'properties': {'name': 'strategic partnerships'}}
    ],
    'relationships': [
        {'start': 'nvidia', 'end': 'announced', 'type': 'financial_results', 'properties': {}},
        {'start': 'jensen_huang', 'end': 'founded', 'type': 'nvidia', 'properties': {}},
        {'start': 'nvidia', 'end': 'partnered_with', 'type': 'cloud_service_providers', 'properties': {}},
        {'start': 'nvidia', 'end': 'returned_to_shareholders', 'type': '1.15 billion', 'properties': {}},
        {'start': 'nvidia', 'end': 'announced_dividend', 'type': '0.04', 'properties': {}},
        {'start': 'nvidia', 'end': 'showed', 'type': 'quarter-over-quarter improvement', 'properties': {}},
        {'start': 'nvidia', 'end': 'maintained', 'type': 'strong fiscal year revenue', 'properties': {}},
        {'start': 'nvidia', 'end': 'made', 'type': 'significant advancements in AI', 'properties': {}},
        {'start': 'nvidia', 'end': 'had', 'type': 'strategic partnerships', 'properties': {}}
    ]
}

result = addnode(None, chunkResult)
import pprint
pprint.pprint(result)
