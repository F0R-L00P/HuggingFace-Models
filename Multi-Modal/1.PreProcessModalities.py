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
from IPython.display import display
from transformers import BlipProcessor, BlipForConditionalGeneration

checkpoint = "Salesforce/blip-image-captioning-base"
model = BlipForConditionalGeneration.from_pretrained(checkpoint)
processor = BlipProcessor.from_pretrained(checkpoint)

# process encoding image -> transform to text encoding -> decode text
# pass image to processor
data_files = "hf://datasets/nlphuji/flickr30k@refs/convert/parquet/TEST/test/*.parquet"

image = load_dataset(
    "parquet",
    data_files=data_files,
    split="train",
)
for i in range(10):
    display(image[i]["image"])

# process images using model
inputs = processor(images=image[0:10]["image"], return_tensors="pt")
# generate token ID
output = model.generate(**inputs)
# decode to obtain captions
captions = processor.batch_decode(output, skip_special_tokens=True)
for caption in captions:
    print(caption)

# ----------------------------------------------------------
# -------------------Audio preprocessing--------------------
# ----------------------------------------------------------
# sequence padding, resampling, feature extraction (i.e. MFCCs, spectrograms)
from datasets import load_dataset, Audio
from transformers import AutoProcessor

# Load dataset (decode=False to avoid torchcodec dependency)
dataset = load_dataset("geronimobasso/drone-audio-detection-samples")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000, decode=False))

# Manually decode audio bytes
import io
import soundfile as sf

audio_bytes = dataset["train"][0]["audio"]["bytes"]
with io.BytesIO(audio_bytes) as f:
    waveform, sr = sf.read(f, dtype="float32")

# Preprocess for Whisper
processor = AutoProcessor.from_pretrained("openai/whisper-small")
audio_preprocessed = processor(waveform, sampling_rate=sr, return_tensors="pt")

print(audio_preprocessed)
# -----------------------------------------------------------
# -------------------------pipeline--------------------------
# -----------------------------------------------------------
