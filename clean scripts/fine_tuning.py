import random
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from transformers import TrainingArguments, Trainer
from datasets import Dataset

with open("twitter-datasets/train_pos_full.txt", "r", encoding="utf-8") as file:
    positive_tweets = file.readlines()

with open("twitter-datasets/train_neg_full.txt", "r", encoding="utf-8") as file:
    negative_tweets = file.readlines()

texts = positive_tweets + negative_tweets
labels = [1] * len(positive_tweets) + [0] * len(negative_tweets)

data = list(zip(texts, labels))  
random.shuffle(data)  

texts, labels = zip(*data)

texts = list(texts)
labels = list(labels)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #To run on SCITAS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #Uncomment to run on GPU for Mac

print("Using device:", device)

#Chose a model to fine-tune, we fine-tuned the fopllowing ones : 

#model_name = 'sentence-transformers/all-MiniLM-L6-v2'
#model_name = "vinai/bertweet-base"
#model_name = "distilbert-base-uncased"
model_name = "cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

tokenized_data = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

hf_dataset = Dataset.from_dict({"text": texts, "label": labels})
tokenized_dataset = hf_dataset.map(
    lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=128),
    batched=True
)

tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_size = int(0.8 * len(tokenized_dataset))
eval_size = len(tokenized_dataset) - train_size

train_dataset, eval_dataset = torch.utils.data.random_split(tokenized_dataset, [train_size, eval_size])

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=10000,
    #use_mps_device=True #Uncomment to use mps device
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  
)

trainer.train()

trainer.save_model(f"./fine_tuned_{model_name}")
tokenizer.save_pretrained(f"./fine_tuned_{model_name}")

eval_results = trainer.evaluate()
print(eval_results)