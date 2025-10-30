# TASK -> finetune a pretrained model with imdb data
import warnings

warnings.filterwarnings("ignore")
from datasets import load_dataset

# Dynamic padding: each batch is padded only to the longest example in that batch
from transformers import DataCollatorWithPadding
from transformers import AutoModel, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification

import torch, platform, sys

print("CUDA available:", torch.cuda.is_available())


# load in teh data
train_data = load_dataset("imdb", split="train")
# train_data = data.shard(num_shards=4, index=0)

test_data = load_dataset("imdb", split="test")
# test_data = data.shard(num_shards=4, index=0)

# load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- use GPU if available ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model device:", next(model.parameters()).device)


# tokenize row by row
def tokenize_function(text_data):
    return tokenizer(
        text_data["text"],
        # padding=True,
        truncation=True,
    )


# tokenize in batches
# tokenized_in_batches = train_data.map(tokenize_function, batched=True)

# tokenized train data
# tokenized_training_data = train_data.map(tokenize_function, batched=False)
# tokenized_test_data = test_data.map(tokenize_function, batched=False)
tokenized_train = train_data.map(
    tokenize_function, batched=True, remove_columns=["text"]
)
# I only need token ids, attention_mask etc, not the raw text
tokenized_test = test_data.map(tokenize_function, batched=True, remove_columns=["text"])

# Lets setup model tuning arguments
training_arguments = TrainingArguments(
    output_dir="./FineTuned",
    eval_strategy="epoch",  # obtain evaluation metrics after EACH epoch
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
)

# now pass object to trainer class instance
trainer = Trainer(
    model=model,  # model wanting finetuned
    args=training_arguments,  # pass argumets
    train_dataset=tokenized_train,  # train data
    eval_dataset=tokenized_test,  # test data
    data_collator=data_collator,  # dynamic padding
    tokenizer=tokenizer,  # tokenizer
)

# instantiate training method loop
trainer.train()

##########################################################
################## test FineTuned Model ##################
##########################################################
model.eval()

new_data = ["This movie was horrible!", "Absolutely loved the movie!"]

new_input = tokenizer(
    new_data, return_tensors="pt", padding=True, truncation=True, max_length=256
)

# Move inputs to same device as model
for k in new_input:
    new_input[k] = new_input[k].to(device)

with torch.no_grad():
    outputs = model(**new_input)

predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()

label_map = {0: "NEGATIVE", 1: "POSITIVE"}

for i, predicted_label in enumerate(predicted_labels):
    sentiment = label_map[predicted_label]
    print(f"\nInput Text {i + 1}: {new_data[i]}")
    print(f"Predicted Label: {sentiment}")

# SVAE FINETUNE MODEL AND TOKENIZER
model.save_pretrained("imdb_expert_model")
tokenizer.save_pretrained("imdb_expert_tokenizer")
