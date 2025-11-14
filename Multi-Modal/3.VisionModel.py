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