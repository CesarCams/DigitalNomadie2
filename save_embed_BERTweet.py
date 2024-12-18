import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
#embedding_matrix = np.load("embeddings_transfo.npy")
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

#model_name = 'sentence-transformers/all-MiniLM-L6-v2'
# Load the all-MiniLM-L6-v2 model
device = "cuda"
model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
#model = SentenceTransformer(model_name)
model.to(device)

#tokenizer.model_max_length = 64
#tokenizer.truncation = True

import pandas as pd

# Define file paths
data_path = "data/twitter-datasets/"
train_neg_path = f"{data_path}train_neg_full.txt"
train_pos_path = f"{data_path}train_pos_full.txt"
test_path = f"{data_path}test_data.txt"

# Load negative tweets and assign a label of -1
with open(train_neg_path, "r") as f:
    neg_tweets = [(line.strip(), -1) for line in f]

print(len(neg_tweets))

# Load positive tweets and assign a label of +1
with open(train_pos_path, "r") as f:
    pos_tweets = [(line.strip(), 1) for line in f]

print(len(pos_tweets))

with open(test_path, "r") as f:
    test_tweets = [(line.strip()[2:], -1) for line in f]

print(len(test_tweets))
# Combine the positive and negative tweets into a single list
texts = neg_tweets + pos_tweets

labels = [1] * len(pos_tweets) + [-1] * len(neg_tweets)

# 4. Shuffle the dataset
import random
data = list(zip(texts, labels))  # Combine texts and labels into tuples
random.shuffle(data)  # Shuffle the dataset

# 5. Unzip the shuffled data back into texts and labels
texts, labels = zip(*data)

# Convert back to list if you need to work with them as lists
texts = list(texts)
labels = list(labels)

#tweets_with_labels = texts[:len(texts)//1000]
#labels = labels[:len(labels)//1000]
tweets_with_labels = texts
# Optional: Shuffle the dataset (important for training)


# Convert to a DataFrame for easy manipulation and viewing
df = pd.DataFrame({'tweet': tweets_with_labels, 'label': labels})
df["tweet"] = df["tweet"].apply(lambda x: x[0])
print(df.head())


df["word_count"] = df["tweet"].apply(lambda x: len(x.split()))
#df_test = pd.DataFrame(test_tweets, columns=["tweet", "label"])
#df_test["word_count"] = df_test["tweet"].apply(lambda x: len(x.split()))
# Display the first few rows of the DataFrame
print(df.head())
#print(df_test.head())

import torch

batch_size = 32  # Choose an appropriate batch size based on your available memory
all_embeddings = []

for start_idx in tqdm(range(0, len(df), batch_size)):
    batch_tweets =list(df["tweet"].values[start_idx:start_idx + batch_size])
    inputs = tokenizer(batch_tweets, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
    	outputs = model(**inputs)
    batch_embeddings = outputs.last_hidden_state[:, 0, :]
    all_embeddings.append(batch_embeddings.cpu())

    # Clear cache after each batch to avoid memory fragmentation
    torch.cuda.empty_cache()
   # all_embeddings.append(batch_embeddings)

# The embeddings are in `outputs.last_hidden_state`
# Each token has a 768-dimensional embedding (for BERTweet-base)
# If you need a sentence-level embedding, take the [CLS] token (index 0)
#sentence_embeddings = outputs.last_hidden_state[:, 0, :]
# all_embeddings.append(batch_embeddings)

# Concatenate the results
features_matrix_train = torch.cat(all_embeddings, dim=0).to(device)

#features_matrix_train = model.encode(df["tweet"].values, convert_to_tensor=True)
word_count_tensor = torch.tensor(df["word_count"].values, dtype=torch.float32).unsqueeze(1).to(device)  # Shape: (num_samples, 1)
features_all_train = torch.cat((features_matrix_train, word_count_tensor), dim=1)  # Shape: (num_samples, embedding_dim + 1)

features = features_all_train.cpu()
labels = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

import torch

#features_matrix_test = model.encode(df_test["tweet"].values, convert_to_tensor=True)
#word_count_tensor_test = torch.tensor(df_test["word_count"].values, dtype=torch.float32).unsqueeze(1).to(device)  # Shape: (num_samples, 1)
#features_all_test = torch.cat((features_matrix_test, word_count_tensor_test), dim=1)  # Shape: (num_samples, embedding_dim + 1)

np.save("features_all_train_BERTweet.npy", X_train.cpu().detach().numpy())
np.save("features_all_test_BERTweet.npy", X_test.cpu().detach().numpy())

np.save("labels_train_BERTweet.npy", y_train)
np.save("labels_test_BERTweet.npy", y_test)
