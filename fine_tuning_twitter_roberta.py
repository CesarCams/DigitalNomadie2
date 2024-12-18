import random

# 1. Load Positive Tweets
with open("twitter-datasets/train_pos_full.txt", "r", encoding="utf-8") as file:
    positive_tweets = file.readlines()
    #positive_tweets = positive_tweets[:len(positive_tweets)//10]  # Select only half of the positive tweets

# 2. Load Negative Tweets
with open("twitter-datasets/train_neg_full.txt", "r", encoding="utf-8") as file:
    negative_tweets = file.readlines()
    #negative_tweets = negative_tweets[:len(negative_tweets)]  # Select only half of the negative tweets

# 3. Create the texts and labels lists
texts = positive_tweets + negative_tweets
labels = [1] * len(positive_tweets) + [0] * len(negative_tweets)

# 4. Shuffle the dataset
data = list(zip(texts, labels))  # Combine texts and labels into tuples
random.shuffle(data)  # Shuffle the dataset


# 5. Unzip the shuffled data back into texts and labels
texts, labels = zip(*data)

# Convert back to list if you need to work with them as lists
texts = list(texts)
labels = list(labels)

#texts = texts[:len(texts)//100]
#labels = labels[:len(labels)//100]


from transformers import AutoTokenizer

# Load a pretrained tokenizer
# model_name = "distilbert-base-uncased"  # You can replace this with another model
# tokenizer = AutoTokenizer.from_pretrained(model_name)

from sentence_transformers import SentenceTransformer
#model_name = 'sentence-transformers/all-MiniLM-L6-v2'
# Load the all-MiniLM-L6-v2 model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
#device = "mps"
#model = SentenceTransformer(model_name)
from transformers import AutoModelForSequenceClassification
#model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2,ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#model.to(device)

# Tokenize the data
tokenized_data = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

import torch

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# Create dataset and dataloaders
dataset = TweetDataset(texts, labels, tokenizer)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

from transformers import AutoModelForSequenceClassification

# Load model with classification head
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2,ignore_mismatched_sizes=True)
#model.to(device)

from transformers import TrainingArguments, Trainer
from datasets import Dataset

# Prepare the dataset for Trainer
hf_dataset = Dataset.from_dict({"text": texts, "label": labels})
tokenized_dataset = hf_dataset.map(
    lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=128),
    batched=True
)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

import accelerate

# Train the model
# Initialize the Trainer
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to the MPS device

print("Using device:", device)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=10000,
    #no_cuda=True,  # Disable CUDA to avoid conflicts with MPS
    #use_mps_device=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Use a separate dataset for evaluation in real use
)

trainer.train()

trainer.save_model("./fine_tuned_twitter_roberta_bis")
tokenizer.save_pretrained("./fine_tuned_twitter_roberta_bis")

eval_results = trainer.evaluate()
print(eval_results)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_twitter_roberta_bis")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_twitter_roberta_bis")

data_path = "data/twitter-datasets/"
test_path = f"{data_path}test_data.txt"


with open(test_path, "r") as f:
    test_tweets = [line.strip() for line in f]

from tqdm import tqdm
labels = []

# Analyze sentiment
for tweet in tqdm(test_tweets):
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    negative_prob, positive_prob = probs[0].tolist()
    if positive_prob > negative_prob:
        label = 1  # Positive
    else:
        label = -1  # Negative
    labels.append(label)

ids = np.arange(1, len(labels) + 1)

from helpers import create_csv_submission

create_csv_submission(ids, labels, "data/submission_fully_fine_tuned_twitter_roberta_bis.csv")
