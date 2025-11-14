# huggingface-cli login
# token,
# hf auth whoami
import warnings

warnings.filterwarnings("ignore")

from huggingface_hub import HfApi

# set api
api = HfApi()

# use token automatically
models = api.list_models(filter="text-to-image")
print(f"Task: text-to-image, Models: {len(list(models))}")

# ----------------------------------------------------------
# -------------------Text preprocessing---------------------
# ----------------------------------------------------------
# normalize text (i.e. removing special characters, lowercasing, etc.)
# pre-tokenization (i.e. splitting text into tokens/words) also add CLS, and SEP tokens
# ID conversion (i.e. converting tokens/words into numerical IDs using a vocabulary)
# padding (i.e. adding special PAD tokens to ensure uniform input length) or zero-padding
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
text = "DO you need more Cookies?"
# visualize normalization
print(tokenizer.backend_tokenizer.normalizer.normalize_str(text))
# entire preprocessing pipeline
# 101 is CLS, 102 is SEP
tokenizer_output = tokenizer(text, return_tensors="pt", padding=True)
print(tokenizer_output)

# ----------------------------------------------------------
# -------------------Image preprocessing--------------------
# ----------------------------------------------------------
# normalizing pixel intensity, scaling to mean and std of the dataset
# resizing images, matching the input layer of model
from datasets import load_dataset
from transformers import BlipProcessor, BlipForConditionalGeneration

checkpoint = "Salesforce/blip-image-captioning-base"
model = BlipForConditionalGeneration.from_pretrained(checkpoint)
processor = BlipProcessor.from_pretrained(checkpoint)

# process encoding image -> transform to text encoding -> decode text
# pass image to processor
image = load_dataset("nlphuji/flickr30k", split="test")
image[0]["image"]