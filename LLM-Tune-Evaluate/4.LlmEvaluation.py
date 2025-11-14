import os
import pprint
import inspect
import warnings

import numpy as np

import torch
import evaluate
import lighteval
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

warnings.filterwarnings("ignore")

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
################# TEXT CLASSIFICATION EVALUATION ##################
###################################################################
# load metric -> classic sklearn
accuracy = evaluate.load("accuracy")
# details of teh metric
print(accuracy.description)
# view feature attributes
# reference is ground truth
print(accuracy.features)

# initialize model to test evaluate library
classifier = pipeline(
    task="text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

# Example input texts
evaluation_text = [
    "This movie was horrible!",
    "Absolutely loved the movie!",
    "The plot was predictable and boring.",
    "Fantastic performances and a great soundtrack.",
    "I wouldn’t watch it again, even for free.",
    "The dialogue felt real and the pacing was perfect.",
    "Terrible ending that ruined an otherwise okay film.",
    "A solid film with some surprisingly emotional moments.",
    "One of the best thrillers I’ve seen this year!",
    "Mediocre at best—nothing memorable about it.",
]


predictions = classifier(evaluation_text)

predicted_labels = []
for i in range(len(predictions)):
    if predictions[i]["label"] == "POSITIVE":
        predicted_labels.append(int(1))
    else:
        predicted_labels.append(int(0))

predicted_labels
reference = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]

accuracy.compute(references=reference, predictions=predicted_labels)
evaluate.load("f1").compute(references=reference, predictions=predicted_labels)

###################################################################
################# TEXT GENERATION PREPLEXITY EVALUATION ###########
###################################################################
##TEXT GENERATION
# prompt
input_text = "Latest research findins in Anarctica shows"

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

# tokenizer encoding
text2id = tokenizer.encode(input_text, return_tensors="pt")
# model token generation
with torch.no_grad():
    generated_ids = model.generate(text2id)
# tokenizer decoding
generated_text = tokenizer.decode(generated_ids[0], skip_skeptical_tokens=True)

print(generated_text)

# lets use preplexity to evalute generated text
perplex = evaluate.load("perplexity", module_type="metric")
score = perplex.compute(predictions=generated_text, model_id="distilbert/distilgpt2")

# NOTE: interpretation should be based on some BASELINE results
# output is perplexity score per token, provided as list
print(np.mean(score["perplexities"]))
print(score["mean_perplexity"])  # holds the key no need for np.mean(lol)

###################################################################
################# TEXT GENERATION BLEU EVALUATION #################
###################################################################
# TEXT GENERATION-SUMMARIZATION-TRANSLATION-Q&A
# Measure translation quality against human reference
# prediction LLM output
# reference Human reference
bleu = evaluate.load("bleu")

reference = [
    [
        "Latest research findings in Antarctica show significant ice loss due to climate change.",
        "latest research findings in Anarctica show that the ice sheet is melting faster than previously thought",
    ]
]

# score between 0-1 indicating how close the generated text is to the reference
# a value close to1, means high similarity
results = bleu.compute(predictions=[generated_text], references=reference)

###################################################################
################# TEXT SUMMARIZATION ROUGE EVALUATION #############
###################################################################
# TEXT SUMMARIZATION-Q&A
rouge = evaluate.load("rouge")
# similarities, between generated summaries, and reference summaries
pipe = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn",
    tokenizer="facebook/bart-large-cnn",
)

target_text = """
For centuries, astronomers have discussed the possibility of inhabited worlds — from Herschel’s 18th-century observations suggesting Mars may host life, to the systematic search for technosignatures that began in the 1960s using radio telescopes. Searching for artifacts in the solar system has received relatively little formal scientific interest and has faced significant technical and social challenges. Automated surveys and new observational techniques developed over the past decade now enable astronomers to survey parts of the sky for anomalous objects. We briefly describe four methods for detecting extraterrestrial artifacts and probes within the Solar System and then focus on demonstrating one of these. The first makes use of pre-Sputnik
images to search for flashes from glinting objects. 
"""
reference = [
    """"
Astronomers have long speculated about life beyond Earth, but direct searches for extraterrestrial artifacts in our Solar System have historically received little attention. Recent advances in automated surveys and observation technologies now make such searches feasible. Researchers outline four possible methods for detecting alien probes or artifacts and demonstrate one approach that examines pre-Sputnik astronomical images for reflective flashes from glinting objects.
"""
]
model_summary = pipe(target_text)[0]["summary_text"]

results_bleu = rouge.compute(predictions=[model_summary], references=reference)
# NOTE: ROUGE score 1 (unigram), 2 (bigram), L (long overlapping sebsequences)
# score between 0-1, highest score indicating highest similarity
print(results_bleu)

###################################################################
################ TEXT SUMMARIZATION METEOR EVALUATION #############
###################################################################
# this metric, tries to account meaning, stem matches, synonym, paraphrase
# acore also between 0-1
meteor = evaluate.load("meteor")
results_meteor = meteor.compute(predictions=[model_summary], references=reference)

print(results_meteor)


###################################################################
################ TEXT SQ&A EXTRACTIVE EVALUATION ##################
###################################################################
# Extractive match Q&A | Generative Q&A
# exact match is generally used with F-1 score, rather than isolation
# 1 if LLM's output matches its reference "exactly"
exact_match = evaluate.load("exact_match")
pred = ["the cat sat on the mat.", "like comparing oranges and apples."]
ref = ["the cat sat on the mat.", "like comparing apples and oranges."]
match_results = exact_match.compute(references=ref, predictions=pred)
print(match_results)

###################################################################
#################### TEXT TOXICITY EVALUATION #####################
###################################################################
toxicity_evaluation = evaluate.load("toxicity")
# NOTE: maximum -> returns maximum toxicity score across inputs
# NOTE: ration -> returns the % predictions with toxicity above 0.5

string1 = ["everyone in the team adores him", "He is a pure talent"]
string2 = ["nobody in the team likes him", "He is a useless tool"]

tox1 = toxicity_evaluation.compute(predictions=string1, aggregation="maximum")
tox2 = toxicity_evaluation.compute(predictions=string2, aggregation="ration")

print(tox1)
print(tox2)

###################################################################
#################### TEXT TREGARD EVALUATION ######################
###################################################################
# BIAS EVALUATION TOWARDS A GROUP OR LAGUAGE
# assuming an LLM generated text about a group
regard = evaluate.load("regard")

group1 = ["abc are described as loyal employees", "abc are career driven people"]
group2 = ["xyz cause a lot of team conflicts", "xyz are verbally violent"]

polarity_reuslts1 = regard.compute(data=group1)
polarity_reuslts2 = regard.compute(data=group2)

for i in polarity_reuslts1["regard"]:
    print(i)


for j in polarity_reuslts2["regard"]:
    print(j)
