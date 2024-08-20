import json

# 文件路径
input_file_path = '/home/ljc/representation-engineering/llama-recipes/src/llama_recipes/datasets/couterdata/counterfact.json'
output_file_path = '/home/ljc/representation-engineering/llama-recipes/src/llama_recipes/datasets/couterdata/counterfact_first_100.json'

# 读取 JSON 文件
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 截取前 100 个条目
subset_data = data[:1000]

# 保存到新的 JSON 文件
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(subset_data, f, ensure_ascii=False, indent=2)

print(f"前 100 个条目已保存到 {output_file_path}")
