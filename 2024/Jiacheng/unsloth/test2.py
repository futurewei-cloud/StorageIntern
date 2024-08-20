

from transformers import pipeline

# Load the relation extraction pipeline
re_pipeline = pipeline("ner", model="Jean-Baptiste/camembert-ner")

# Process the document
doc = """In machine learning, a hyperparameter is a parameter, such as the learning rate or choice of optimizer, which specifies details of the learning process, hence the name hyperparameter. This is in contrast to parameters which determine the model itself.

Hyperparameters can be classified as model hyperparameters, that typically cannot be inferred while fitting the machine to the training set because the objective function is typically non-differentiable with respect to them. As a result, gradient based optimization methods cannot be applied directly. An example of a model hyperparameter is the topology and size of a neural network. Examples of algorithm hyperparameters are learning rate and batch size as well as mini-batch size. Batch size can refer to the full data sample where mini-batch size would be a smaller sample set."""
relations = re_pipeline(doc)

for relation in relations:
    print(relation)
