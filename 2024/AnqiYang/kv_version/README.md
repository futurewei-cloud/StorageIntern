#  KV Cache Distance Visualization 

This project is for fine tuning the Bert Model, and using kv cache extracted from the fifth layer of attention head as distance. The kv cache method optimize the node location and relations distance by using kv score. 

## Repository Structure

- **dataset**: Input articles
- **test_trainer** Saved fine-tuned model
- **main** finetune_kv: code for fine tuning Bert Model, extract attention score, and convert to json file. 
- **visualizaion** visualization code:kv_graph


## Reference 
- Fine-tuned method and code: https://www.tensorflow.org/tfmodels/nlp/fine_tune_bert

- Key, Query Extraction: https://github.com/opensourceai/spark-plan/blob/master/article/2019/11/%E5%9B%BE%E8%A7%A3GPT-2(%E5%8F%AF%E8%A7%86%E5%8C%96Transformer%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B).md
