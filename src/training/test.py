'''
This file is for comparing models and thier performances
'''

from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments

# put model names here
test_model_names = []

# put actual models here w.r.t. test_model_names
test_models = []

args = TensorFlowBenchmarkArguments(
    models=test_model_names, 
    batch_sizes=[8], 
    sequence_lengths=[8, 32, 128, 512],
    training=False,
    inference=True,
)

benchmark = TensorFlowBenchmark(args, configs=test_models)
benchmark.run()