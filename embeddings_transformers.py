from sentence_transformers import SentenceTransformer
from transformers import RobertaModel, RobertaTokenizer
import pickle
import numpy as np
import torch
from tqdm import tqdm

def main():
    # Load the
    #  vocabulary from vocab.pkl
    with open("vocab_cut.txt", "r") as f:
        vocabulary = [line.strip() for line in f]
    # Load SBERT model
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)

    # Generate embeddings
    embeddings = []
    for word in tqdm(vocabulary):
        # Tokenize the word
        inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
        # Forward pass through the model to get the embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Get the embeddings from the last hidden state (average across all tokens)
            last_hidden_state = outputs.last_hidden_state
            word_embedding = last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(word_embedding)

    # Convert the list of embeddings to a numpy array
    embeddings = np.array(embeddings)

    # Save the embeddings to a file
    np.save("embeddings_twitter_roberta.npy", embeddings)


    # Generate embeddings
    #embeddings = model.encode(vocabulary)
    #print("Embeddings generated:", embeddings)

    np.save("embeddings_transfo_twitter.npy", embeddings)

if __name__ == "__main__":
    main()
    