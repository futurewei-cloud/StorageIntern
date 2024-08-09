import networkx as nx

def extract_node_types(file_path):
    graph = nx.read_graphml(file_path)
    
    node_types = {}

    for node_id, data in graph.nodes(data=True):
        node_type = data.get('type', None)
        if node_type:
            node_types[node_id] = node_type.strip('"')

    return node_types

def main():
    file_path = '/home/angel/universe/graphrag/ragtest/output/20240715-201729/artifacts/clustered_graph.1.graphml'  # Replace with the actual path to your .graphml file

    node_types = extract_node_types(file_path)
    
    # Print the extracted node types
    if node_types:
        for node_id, node_type in node_types.items():
            print(f'Node ID: {node_id}, Node Type: {node_type}')
    else:
        print("No node types found.")

if __name__ == "__main__":
    main()
