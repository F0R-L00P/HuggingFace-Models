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

# Background removal: build an alpha mask from segments and apply to the image
try:
    import numpy as np
    from PIL import Image

    if isinstance(result, dict) and ("segments_info" in result):
        seg_info = result["segments_info"]
        seg_map_img = result.get("map") or result.get("segmentation")
        if seg_map_img is not None:
            seg_map = np.array(seg_map_img)
            # Keep all "thing" classes (foreground objects). If key missing, default True.
            ids_to_keep = [s.get("id") for s in seg_info if s.get("is_thing", True)]
            if ids_to_keep:
                alpha = np.isin(seg_map, ids_to_keep).astype(np.uint8) * 255
                rgba = image.convert("RGBA")
                rgba.putalpha(Image.fromarray(alpha, mode="L"))
                plt.figure(figsize=(6, 6))
                plt.title("Background removed")
                plt.imshow(rgba)
                plt.axis("off")
                plt.show()
            else:
                print("No foreground segments found to keep for background removal.")
        else:
            print("Segmentation map not available to build background mask.")
    elif isinstance(result, list):
        # If instance masks are returned, union them as foreground
        masks = []
        for r in result:
            m = r.get("mask") or r.get("segmentation")
            if m is not None:
                masks.append(np.array(m).astype(bool))
        if masks:
            union = np.any(np.stack(masks, axis=0), axis=0).astype(np.uint8) * 255
            rgba = image.convert("RGBA")
            rgba.putalpha(Image.fromarray(union, mode="L"))
            plt.figure(figsize=(6, 6))
            plt.title("Background removed")
            plt.imshow(rgba)
            plt.axis("off")
            plt.show()
        else:
            print("No instance masks available for background removal.")
    else:
        print("Segmentation output does not support background removal path.")
except Exception as e:
    print("Background removal failed:", e)

# Robust background removal fallback using RMBG (returns a clean alpha matte)
try:
    import numpy as np
    from PIL import Image

    rmbg_pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4")
    rmbg_res = rmbg_pipe(image)

    matte = None
    if isinstance(rmbg_res, list) and len(rmbg_res) > 0:
        matte = rmbg_res[0].get("mask") or rmbg_res[0].get("segmentation")
    elif isinstance(rmbg_res, dict):
        matte = rmbg_res.get("mask") or rmbg_res.get("segmentation")

    if matte is not None:
        alpha = np.array(matte.resize(image.size)).astype(np.uint8)
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        rgba = image.convert("RGBA")
        rgba.putalpha(Image.fromarray(alpha, mode="L"))
        plt.figure(figsize=(6, 6))
        plt.title("Background removed (RMBG)")
        plt.imshow(rgba)
        plt.axis("off")
        plt.show()
    else:
        print("RMBG did not return a usable mask for background removal.")
except Exception as e:
    print("RMBG background removal failed:", e)
