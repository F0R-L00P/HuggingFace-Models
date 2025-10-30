import os
import pprint
import inspect
import warnings

warnings.filterwarnings("ignore")
import evaluate
import lighteval
from transformers import pipeline

# build pipe, call model
generator = pipeline(
    task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)

input_text = """
classify the sentiment of this sentence as either Positive or Negative.

Example:
Text: "I'm feeling great today!" Sentiment: Positive
Text: "The garden looks beautiful" Sentiment:
"""

result = generator(input_text, max_length=100)

print(result[0]["label"])

###################################################################
####################### EVALUATE ##################################
###################################################################
# load metric -> classic sklearn
accuracy = evaluate.load("accuracy")
# details of teh metric
print(accuracy.description)
# view feature attributes
# reference is ground truth
print(accuracy.features)

classifier = pipeline
