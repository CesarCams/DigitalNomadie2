from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from tqdm import tqdm
 
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

model.to("cuda")
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

    negative_prob, neutral_prob, positive_prob = probs[0].tolist()
    print("negative : ",negative_prob)
    print("positive : ",positive_prob)
    print("neutral : ",neutral_prob)
    if positive_prob > negative_prob:
        label = 1  # Positive
    else:
        label = -1  # Negative
    labels.append(label)

ids = np.arange(1, len(labels) + 1)
from helpers import create_csv_submission

create_csv_submission(ids, labels, "connection/submission_not_fine_tuned_twitter_roberta.csv")
