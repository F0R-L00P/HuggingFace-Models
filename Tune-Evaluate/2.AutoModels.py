import pprint
import warnings

warnings.filterwarnings("ignore")
# more ocntrol over the model and its behvaior
from transformers import pipeline
from transformers import AutoModelForSequenceClassification

# download a pretrain text classification
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

# to prepare the text use a tokenizer associated with the model
from transformers import AutoTokenizer

# get tokenizer, paired with the model
# cleans and splits text into tokens
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# NOTE: Each model, may produce different output from the same input
# tokenize input
inputs = tokenizer.tokenize(
    "LLMs, help human to understand the world better, with some added noise!"
)
print(inputs)


# -----------------------------------------------------------------
# building a custom pipeline --------------------------------------
# -----------------------------------------------------------------
# setup the model and the tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
# create a text classification pipeline
my_pipeline = pipeline(task="text-classification", model=model, tokenizer=tokenizer)
