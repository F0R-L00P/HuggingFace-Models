import os
import pprint
import inspect
import warnings
from transformers import pipeline

warnings.filterwarnings("ignore")

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

# --------------------------------------------------------------
# data loading pipeline-----------------------------------------
# --------------------------------------------------------------
# lets try downloading a dataset and running a pipeline on it
from datasets import load_dataset

data = load_dataset("IVN-RIN/BioBERT_Italian", split="train")

print(data)
print(data.column_names)
data["text"][0:2]
# --------------------------------------------------------------
# create a text classification pipeline ------------------------
# --------------------------------------------------------------
sentence_classification = pipeline(
    task="text-classification", model="distilbert-base-uncased-finetuned-sst-2-english"
)

print(sentence_classification("I love using transformers library."))
print(sentence_classification("I hate so much waiting in traffic jams."))
# --------------------------------------------------------------
# grammer checker pipeline -------------------------------------
# --------------------------------------------------------------
# grammar checking pipeline
grammar_checker = pipeline(
    task="text-classification", model="abdulmatinomotoso/English_grammar_checker"
)

sentence1 = ["He eat pizza every day."]
sentence2 = ["She doesn't like that kind of music."]

result1 = grammar_checker(sentence1)
result2 = grammar_checker(sentence2)
# label 0, is incorrect grammar
# label 1, is correct grammar
print(result1)
print(result2)

# --------------------------------------------------------------
# Quesion Natural Language Inference QNLI ----------------------
# --------------------------------------------------------------
# QNLI pipeline
qnli_classifier = pipeline(
    task="text-classification", model="cross-encoder/qnli-electra-base"
)

qnli_classifier("Where is Seattle located?", "Seattle is located in Washington State")

# --------------------------------------------------------------
# dynamic category assignment ----------------------------------
# --------------------------------------------------------------
# q: i want to know more about your pricing plans
# categories: sales (True), Marketing (False), Support (False)
# content moderation in recommender systems

content_classifier = pipeline(
    task="zero-shot-classification", model="facebook/bart-large-mnli"
)

text = " i would like to request a demo of your product in our organization."
candidate_labels = ["sales", "marketing", "support"]

result = content_classifier(text, candidate_labels)
pprint.pprint(result)

# --------------------------------------------------------------
# summarization pipeline --------------------------------------
# --------------------------------------------------------------
# extractive summarization
summarizer = pipeline(task="summarization", model="nyamuda/extractive-summarization")

text = """
In mathematics, the Fourier transform (FT) is an integral transform that takes a function as input, and outputs another function that describes the extent to which various frequencies are present in the original function. The output of the transform is a complex-valued function of frequency. The term Fourier transform refers to both the mathematical operation and to this complex-valued function. When a distinction needs to be made, the output of the operation is sometimes called the frequency domain representation of the original function.[note 1] The Fourier transform is analogous to decomposing the sound of a musical chord into the intensities of its constituent pitches.


The Fourier transform relates the time domain, in red, with a function in the domain of the frequency, in blue. The component frequencies, extended for the whole frequency spectrum, are shown as peaks in the domain of the frequency.
Functions that are localized in the time domain have Fourier transforms that are spread out across the frequency domain and vice versa, a phenomenon known as the uncertainty principle. The critical case for this principle is the Gaussian function, of substantial importance in probability theory and statistics as well as in the study of physical phenomena exhibiting normal distribution (e.g., diffusion). The Fourier transform of a Gaussian function is another Gaussian function. Joseph Fourier introduced sine and cosine transforms (which correspond to the imaginary and real components of the modern Fourier transform) in his study of heat transfer, where Gaussian functions appear as solutions of the heat equation."""

summary_text = summarizer(text)
pprint.pprint(summary_text)

# example abstractive summarization
abstractive_summarizer = pipeline(
    task="summarization", model="sshleifer/distilbart-cnn-12-6"
)
text = """Data Science is the art and scinence of Data?"""
# model could make stuff up! can fabricate content not presented in the original text
abstractive_summary = abstractive_summarizer(text)
pprint.pprint(abstractive_summary)
