from sentence_transformers import SentenceTransformer
import pickle
import numpy as np


def main():
    # Load the
    #  vocabulary from vocab.pkl
    with open("vocab_cut.txt", "r") as f:
        vocabulary = [line.strip() for line in f]
    # Load SBERT model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Vocabulary


    # Generate embeddings
    embeddings = model.encode(vocabulary)
    #print("Embeddings generated:", embeddings)

    np.save("embeddings_transfo.npy", embeddings)

if __name__ == "__main__":
    main()
    