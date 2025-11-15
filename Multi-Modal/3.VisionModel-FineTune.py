# Use a pipeline as a high-level helper
import warnings

warnings.filterwarnings("ignore")
from transformers import pipeline
from datasets import load_dataset

# -----------------------------------------------------------
# --------------------Image-Classification-------------------
# -----------------------------------------------------------
data_files = "hf://datasets/nlphuji/flickr30k@refs/convert/parquet/TEST/test/*.parquet"

dataset = load_dataset(
    "parquet",
    data_files=data_files,
    split="train",
)

image = dataset[134]["image"]
image.show()
# 224x224 image auto resizing for MobileNet
pipe = pipeline(task="image-classification", model="google/mobilenet_v2_1.0_224")
result = pipe(image)
print(result)

# -----------------------------------------------------------
# ------------------------Object Detection-------------------
# -----------------------------------------------------------
image = dataset[52]["image"]
image.show()

pipe = pipeline(task="object-detection", model="facebook/detr-resnet-50")

# define a threshold to filter results
result = pipe(image, threshold=0.95)
print(result)

# model found 5 people with greater thn 95% confidence
for object in result:
    box = object["box"]
    label = object["label"]
    score = object["score"]
    print(
        f"Detected {label} with confidence {score:.4f} at "
        f"({box['xmin']}, {box['ymin']}) - ({box['xmax']}, {box['ymax']})"
    )

# visualize using patches modules
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ax = plt.gca()
colors = ["r", "g", "b", "y", "c", "m", "k"]
plt.imshow(image)
for i, object in enumerate(result):
    box = object["box"]
    label = object["label"]
    score = object["score"]
    rect = patches.Rectangle(
        (box["xmin"], box["ymin"]),
        box["xmax"] - box["xmin"],
        box["ymax"] - box["ymin"],
        linewidth=2,
        edgecolor=colors[i % len(colors)],
        facecolor="none",
    )
    ax.add_patch(rect)
    plt.text(
        box["xmin"],
        box["ymin"] - 10,
        f"{label}: {score:.2f}",
        color=colors[i % len(colors)],
        fontsize=12,
        weight="bold",
    )
plt.show()

# -----------------------------------------------------------
# ---------------------Image Segmentation-------------------
# ----------------------------------------------------------
# image labels as 1 or 0, for background or foreground
# lets apply background removal
pipe = pipeline(task="image-segmentation", model="facebook/detr-resnet-50-panoptic")
result = pipe(image)
# Visualize segmentation correctly (result is a dict or list, not an image)
if isinstance(result, dict):
    seg = result.get("segmentation") or result.get("map") or result.get("mask")
    if seg is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(image)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Segmentation")
        plt.imshow(seg)
        plt.axis("off")
        plt.show()
    else:
        print("Segmentation result does not contain a displayable mask.")
elif isinstance(result, list):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    for r in result:
        m = r.get("mask") or r.get("segmentation")
        if m is not None:
            plt.imshow(m, alpha=0.5)
    plt.show()
else:
    print("Unexpected segmentation output type:", type(result))

# -----------------------------------------------------------
# ----------------FINE-TUNING VISION MODEL-------------------
# -----------------------------------------------------------
from datasets import load_dataset

# ~1,200 images total
dataset = load_dataset("beans")
print(dataset)
print(dataset["train"][0])

# view a single image
image = dataset["train"][0]["image"]
image.show()

# view labels
labels = dataset["train"].features["labels"].names
print(labels)

# generate label mapping
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

print(label2id)
print(id2label)

from transformers import AutoImageProcessor, AutoModelForImageClassification

checkpoint = "google/mobilenet_v2_1.0_224"
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    # final classification Linear layer is reinitialized with output size of 3
    ignore_mismatched_sizes=True,
)

# new data must be preprocessed as original input data and converted to tensors
# Data PreProcessing
import torch
from transformers import AutoImageProcessor
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

image_processor = AutoImageProcessor.from_pretrained(checkpoint)

height = width = 224

# normalization pixl intensity
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

transform = Compose([Resize((height, width)), normalize, ToTensor()])


# define a transforms function for conversion of the new data
def transforms(examples):
    # 1) make sure images are RGB PIL
    rgb_images = []
    for img in examples["image"]:
        rgb_images.append(img.convert("RGB"))

    # 2) image_processor do resize + normalize
    encoded = image_processor(images=rgb_images, return_tensors="pt")

    pixel_values_batch = encoded["pixel_values"]  # shape: (batch, 3, H, W)

    # 3) convert batch tensor into a list of tensors (one per example)
    pixel_values_list = []
    for i in range(pixel_values_batch.shape[0]):
        pixel_values_list.append(pixel_values_batch[i])

    examples["pixel_values"] = pixel_values_list
    return examples

# with transforms attached the transforms to the dataset
dataset = dataset.with_transform(transforms)
print(dataset)

# view
plt.imshow(dataset["train"][10]["pixel_values"].permute(1, 2, 0))
plt.show()

data_train = dataset["train"]
data_val = dataset["validation"]
data_test = dataset["test"]

# train model
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits: (batch, num_labels)
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir="./dataset-beans",
    per_device_train_batch_size=16,
    learning_rate=6e-4,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    push_to_hub=False,
    remove_unused_columns=False,
)


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    labels = torch.tensor([e["labels"] for e in examples])
    return {"pixel_values": pixel_values, "labels": labels}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_train,
    eval_dataset=data_val,
    data_collator=collate_fn,
)

# predict prior to training
predictions = trainer.predict(data_test)
predictions.metrics['test_accuracy']