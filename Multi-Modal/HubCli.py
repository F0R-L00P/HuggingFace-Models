# huggingface-cli login
# token,
# hf auth whoami

from huggingface_hub import HfApi

# set api
api = HfApi()

# use token automatically
models = api.list_models(filter="text-to-image")
print(f"Task: text-to-image, Models: {len(list(models))}")
