from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from helpers import create_csv_submission
from tqdm import tqdm

model_name = "fine_tuned_bertweet-base" #Path to the model used for the submission

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

data_path = "data/twitter-datasets/"
test_path = f"{data_path}test_data.txt"


with open(test_path, "r") as f:
    test_tweets = [line.strip() for line in f]

labels = []


for tweet in tqdm(test_tweets):
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    negative_prob, positive_prob = probs[0].tolist()
    if positive_prob > negative_prob:
        label = 1  
    else:
        label = -1 
    labels.append(label)

ids = np.arange(1, len(labels) + 1)

#Creates AI crowd submission 
create_csv_submission(ids, labels, f"data/submission_{model_name.split("/")[-1]}.csv")
