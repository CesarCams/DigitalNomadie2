import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import torch
from sklearn.linear_model import LogisticRegression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #To run on SCITAS
#device = "mps" if torch.backends.mps.is_available() else "cpu" #Uncomment to run on GPU for Mac

print("Using device:", device)

#Chose a model to encode the tweets ahead of training the classifier : 

#model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_name = "vinai/bertweet-base" #This one gave the best results
#model_name = "distilbert-base-uncased"
#model_name = "cardiffnlp/twitter-roberta-base-sentiment"

if "sentence-transformers" in model_name:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
else: 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)

data_path = "data/twitter-datasets/"
train_neg_path = f"{data_path}train_neg_full.txt"
train_pos_path = f"{data_path}train_pos_full.txt"
test_path = f"{data_path}test_data.txt"

with open(train_neg_path, "r") as f:
    neg_tweets = [(line.strip(), -1) for line in f]

#print(len(neg_tweets))

with open(train_pos_path, "r") as f:
    pos_tweets = [(line.strip(), 1) for line in f]

#print(len(pos_tweets))

with open(test_path, "r") as f:
    test_tweets = [(line.strip()[2:], -1) for line in f]


texts = neg_tweets + pos_tweets

labels = [1] * len(pos_tweets) + [-1] * len(neg_tweets)

data = list(zip(texts, labels))  
random.shuffle(data)  

texts, labels = zip(*data)

texts = list(texts)
labels = list(labels)

texts = texts[:len(texts)//1000]
labels = labels[:len(labels)//1000]

df = pd.DataFrame({'tweet': texts, 'label': labels})
df["tweet"] = df["tweet"].apply(lambda x: x[0])

#Get the tweet length to add as an extra feature
df["word_count"] = df["tweet"].apply(lambda x: len(x.split()))


batch_size = 32  #Tradeoff between computation time and memory usage
all_embeddings = []

for start_idx in tqdm(range(0, len(df), batch_size)):
    batch_tweets =list(df["tweet"].values[start_idx:start_idx + batch_size])
    if "sentence-transformers" in model_name:
        batch_embeddings = model.encode(batch_tweets, convert_to_tensor=True)
    else:
        inputs = tokenizer(batch_tweets, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :]
    all_embeddings.append(batch_embeddings.cpu())

    #Clear cache for memory management    
    torch.cuda.empty_cache()
   
features_matrix_train = torch.cat(all_embeddings, dim=0).to(device)

word_count_tensor = torch.tensor(df["word_count"].values, dtype=torch.float32).unsqueeze(1).to(device)  

#Add the tweet length as an extra feature
features_all_train = torch.cat((features_matrix_train, word_count_tensor), dim=1)  
features = features_all_train.cpu()
labels = df["label"].values

#Save the features and labels for training and testing of the different classifiers 
model_id = model_name.split("/")[-1]
filename = "features_all_{}.npy".format(model_id)
filename_labels = "labels_all_{}.npy".format(model_id)

np.save(filename, features.cpu().detach().numpy())
np.save(filename_labels, labels)

