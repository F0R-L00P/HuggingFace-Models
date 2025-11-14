# Use a pipeline as a high-level helper
import warnings

warnings.filterwarnings("ignore")
from transformers import pipeline
from datasets import load_dataset

pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

data_files = "hf://datasets/nlphuji/flickr30k@refs/convert/parquet/TEST/test/*.parquet"

image = load_dataset(
    "parquet",
    data_files=data_files,
    split="train",
)

for i in range(10):
    result = pipe(image[i]["image"])
    print(result)

# text-to-audio
import scipy.io.wavfile as wavfile

pipe = pipeline("text-to-audio", model="facebook/musicgen-small", framework="pt")
generate_kwargs = {"temperature": 0.8, "max_new_tokens": 1000}
output = pipe(
    "A calming piano melody with soft strings in the background.",
    generate_kwargs=generate_kwargs,
)

# Save and play audio
from IPython.display import Audio, display

sampling_rate = output["sampling_rate"]
audio_data = output["audio"][0]  # Get the audio array
display(Audio(audio_data, rate=sampling_rate))


# -----------------------------------------------------------
# -------------------------Evaluation------------------------
# -----------------------------------------------------------
from evaluate import evaluator

task_evaluator = evaluator("image-classification")
metrics_dict = {"percision": "percision", "recall": "recall", "f1": "f1"}
label_map = pipe.model.config.label2id
