import os
import pprint
import inspect
import warnings
from transformers import pipeline

warnings.filterwarnings("ignore", category=UserWarning)

# check pipeline configs
print(pipeline.__doc__)

gpt2_pipeline = pipeline(task="text-generation", model="openai-community/gpt2")

# see all model knobs
print(gpt2_pipeline.model.generation_config)

pprint.pprint(gpt2_pipeline("As far as I am concerned, I will"))

# add parameters to model call
results = gpt2_pipeline(
    "As far as I am concerned, I will", max_new_tokens=10, num_return_sequences=3
)

pprint.pprint(results)

# loop over results extract generated texts
for result in results:
    print(result["generated_text"])


# lets try downloading a dataset and running a pipeline on it
from datasets import load_dataset

data = load_dataset("IVN-RIN/BioBERT_Italian", split="train")

print(data)
print(data.column_names)
data["text"][0:2]
