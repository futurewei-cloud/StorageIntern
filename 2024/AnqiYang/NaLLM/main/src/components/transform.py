def create_neo4j_input(data):
    nodes = data['nodes']
    relationships = data['relationships']
    
    node_statements = []
    relationship_statements = []
    
    # Create node statements
    for node in nodes:
        node['properties']['name'] = node['name']
        label = node['label']
        properties = ', '.join([f"{key}: '{value}'" for key, value in node['properties'].items()])
        node_statement = f"MERGE (:{label} {{ {properties} }});"
        node_statements.append(node_statement)
    
    # Create relationship statements
    for relationship in relationships:
        start_node = relationship['start']
        end_node = relationship['type']
        rel_type = relationship['end'].upper()
        # relationship_statement = f"MATCH (a:{nodes[0]['label']} {{ name: '{nodes[0]['properties']['name']}' }}), (b:{nodes[1]['label']} {{ name: '{nodes[1]['properties']['name']}' }}) MERGE (a)-[:{rel_type}]->(b);"
        # relationship_statements.append(relationship_statement)
        relationship_statement = f"MATCH (a {{ name: '{start_node}' }}), (b {{ name: '{end_node}' }}) MERGE (a)-[:{rel_type}]->(b);"
        relationship_statements.append(relationship_statement)
    
    return node_statements, relationship_statements

