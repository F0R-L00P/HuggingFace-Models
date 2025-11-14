# Use a pipeline as a high-level helper
import warnings

warnings.filterwarnings("ignore")
from transformers import pipeline
from datasets import load_dataset

# -----------------------------------------------------------
# ------------------------Image-to-Text-----------------------
# -----------------------------------------------------------
pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

data_files = "hf://datasets/nlphuji/flickr30k@refs/convert/parquet/TEST/test/*.parquet"

image = load_dataset(
    "parquet",
    data_files=data_files,
    split="train",
)

for i in range(10):
    # using pipeline to make the conversion
    result = pipe(image[i]["image"])
    print(result)

# -----------------------------------------------------------
# ------------------------Text-to-Audio-----------------------
# -----------------------------------------------------------
import scipy.io.wavfile as wavfile

music1 = "A calming piano melody with soft strings in the background."
music2 = "An upbeat pop song with energetic drums and catchy vocals."
music3 = "An electronic dance track with pulsating bass and vibrant synths."

pipe = pipeline("text-to-audio", model="facebook/musicgen-small", framework="pt")
generate_kwargs = {"temperature": 0.8, "max_new_tokens": 1000}
output = pipe(music3, generate_kwargs=generate_kwargs)

# Save and play audio
from IPython.display import Audio, display

sampling_rate = output["sampling_rate"]
audio_data = output["audio"][0]  # Get the audio array
display(Audio(audio_data, rate=sampling_rate))


# -----------------------------------------------------------
# -------------------------Evaluation------------------------
# -----------------------------------------------------------
# Minimal zero-shot image classification example
zs_pipe = pipeline(
    "zero-shot-image-classification", model="openai/clip-vit-base-patch32"
)
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png"
zs_result = zs_pipe(image_url, candidate_labels=["animals", "humans", "landscape"])
print(zs_result)
print(f"Top label: {zs_result[0]['label']} (score={zs_result[0]['score']:.4f})")

# -----------------------------------------------------------
# ---------------Proper Evaluation (labeled data)------------
# -----------------------------------------------------------
# Use a classifier fine-tuned on a labeled dataset (beans) and compute accuracy.
from evaluate import evaluator

clf_data = load_dataset("beans", split="validation")
clf_pipe = pipeline("image-classification", model="nateraw/vit-base-beans")
task_eval = evaluator("image-classification")

# Map string labels from pipeline outputs to dataset int ids
label_map = clf_pipe.model.config.label2id  # e.g., {"angular_leaf_spot":0, ...}

eval_results = task_eval.compute(
    model_or_pipeline=clf_pipe,
    data=clf_data,
    metric="accuracy",
    input_column="image",
    label_column="labels",
    label_mapping=label_map,
)
print("Evaluation results:", eval_results)
