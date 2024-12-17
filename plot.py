
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.tokenize import word_tokenize
import nltk

# Download nltk 'punkt' tokenizer if not already downloaded
nltk.download('punkt_tab')

# Function to load tweets
def load_tweets(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tweets = file.readlines()
    return tweets

# Function to load and clean test_data tweets
def load_and_clean_test_tweets(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tweets = file.readlines()
    # Remove leading numerical IDs (e.g., "1," or "2,")
    cleaned_tweets = [re.sub(r"^\d+,\s*", "", tweet).strip() for tweet in tweets]
    return cleaned_tweets

def count_metric(tweet, vocab, count_words=True):
    """
    Count words or characters in a tweet using the vocabulary filter.

    Args:
        tweet (str): The tweet text.
        vocab (set): A set of valid words from the vocabulary.
        count_words (bool): If True, count words; otherwise, count characters.

    Returns:
        int: Word count or character count.
    """
    if count_words:
        return len(tokenize_with_vocab(tweet, vocab))
    else:
        return len(tweet)  # Count characters directly

# Custom tokenizer based on vocab_full.txt
def load_vocab(vocab_path):
    """
    Load the dictionary (vocabulary) file as a set of valid words.

    Args:
        vocab_path (str): Path to the vocabulary file.

    Returns:
        set: A set of valid words in the vocabulary.
    """
    with open(vocab_path, 'r', encoding='utf-8') as file:
        return set(line.strip() for line in file)

def tokenize_with_vocab(tweet, vocab):
    """
    Tokenize a tweet and retain only words present in the vocabulary.

    Args:
        tweet (str): The tweet text to tokenize.
        vocab (set): A set of valid words from the vocabulary.

    Returns:
        list: A list of valid tokens.
    """
    return [word for word in tweet.split() if word in vocab]

# Main plotting function
def tweet_length_plotter(data_paths, clean_paths, vocab_path, count_words=True, normalize=True, plot_type='hist'):
    """
    Function to plot tweet lengths (words or characters) for given datasets,
    using a vocabulary-based tokenizer.

    Args:
        data_paths (list): List of file paths for raw datasets.
        clean_paths (list): List of file paths for datasets to clean.
        vocab_path (str): Path to the vocabulary file.
        count_words (bool): If True, count words; otherwise, count characters.
        normalize (bool): If True, normalize histograms.
        plot_type (str): 'hist' for histograms, 'kde' for Kernel Density Plots.
    """
    # Load vocabulary
    vocab = load_vocab(vocab_path)

    # Load datasets
    datasets = [load_tweets(path) for path in data_paths]
    cleaned_datasets = [load_and_clean_test_tweets(path) for path in clean_paths]
    
    # Combine all datasets
    all_datasets = datasets + cleaned_datasets
    dataset_labels = ['Train Negative Tweets', 'Train Positive Tweets', 'Test Tweets (Cleaned)']
    
    # Calculate lengths using the vocabulary filter
    dataset_lengths = [
        [count_metric(tweet, vocab, count_words) for tweet in dataset] 
        for dataset in all_datasets
    ]
    
    # Plot setup
    plt.figure(figsize=(12, 6))
    metric = 'Words' if count_words else 'Characters'

    # Choose plot type
    if plot_type == 'hist':
        for lengths, label, color in zip(dataset_lengths, dataset_labels, ['red', 'green', 'blue']):
            plt.hist(lengths, bins=50, alpha=0.5, label=label, color=color, density=normalize)
        plt.ylabel('Density' if normalize else 'Number of Tweets',fontsize=18)
    elif plot_type == 'kde':
        for lengths, label, color in zip(dataset_lengths, dataset_labels, ['red', 'green', 'blue']):
            sns.kdeplot(lengths, shade=True, label=label, color=color)
        plt.ylabel('Density',fontsize=18)
    else:
        raise ValueError("Invalid plot_type. Use 'hist' or 'kde'.")
    
    # Add plot details
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(f'Number of {metric} per Tweet',fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def count_tweets_above_word_threshold(data_paths, clean_paths, word_threshold):
    """
    Function to count the number of tweets with more than a specified number of words.

    Args:
        data_paths (list): List of file paths for raw datasets.
        clean_paths (list): List of file paths for cleaned datasets.
        word_threshold (int): Number of words to use as the threshold.

    Returns:
        int: Total number of tweets with words greater than the given threshold.
    """
    # Load datasets
    datasets = [load_tweets(path) for path in data_paths]
    cleaned_datasets = [load_and_clean_test_tweets(path) for path in clean_paths]
    
    # Combine all datasets
    all_datasets = datasets + cleaned_datasets
    
    # Count tweets exceeding the word threshold
    total_count = sum(
        1 for dataset in all_datasets for tweet in dataset 
        if len(tweet.split()) > word_threshold
    )
    
    return total_count

def ratio_tweets_above_word_threshold(data_paths, clean_paths, word_threshold):
    """
    Function to calculate the ratio of tweets with more than a specified number of words.

    Args:
        data_paths (list): List of file paths for raw datasets.
        clean_paths (list): List of file paths for cleaned datasets.
        word_threshold (int): Number of words to use as the threshold.

    Returns:
        float: Ratio of tweets with words greater than the given threshold.
    """
    # Load datasets
    datasets = [load_tweets(path) for path in data_paths]
    cleaned_datasets = [load_and_clean_test_tweets(path) for path in clean_paths]
    
    # Combine all datasets
    all_datasets = datasets + cleaned_datasets
    
    # Flatten all tweets into one list
    all_tweets = [tweet for dataset in all_datasets for tweet in dataset]
    
    # Count tweets exceeding the word threshold
    count_above_threshold = sum(1 for tweet in all_tweets if len(tweet.split()) > word_threshold)
    
    # Calculate ratio
    total_tweets = len(all_tweets)
    ratio = count_above_threshold / total_tweets if total_tweets > 0 else 0
    
    return ratio

from collections import Counter

def top_100_words_from_datasets(data_paths, clean_paths, vocab_path, min_word_length=1):
    """
    Function to return the 100 most frequent words in the datasets based on a given vocabulary,
    considering a minimum word length.

    Args:
        data_paths (list): List of file paths for raw datasets.
        clean_paths (list): List of file paths for cleaned datasets.
        vocab_path (str): Path to the vocabulary file (one word per line).
        min_word_length (int): Minimum length of words to consider.

    Returns:
        list: A list of tuples containing the 100 most frequent words and their counts.
    """
    # Load vocabulary
    with open(vocab_path, 'r', encoding='utf-8') as file:
        vocabulary = set(line.strip() for line in file)  # Read and clean words in vocab
    
    # Load datasets
    datasets = [load_tweets(path) for path in data_paths]
    cleaned_datasets = [load_and_clean_test_tweets(path) for path in clean_paths]
    
    # Combine all datasets
    all_datasets = datasets + cleaned_datasets
    
    # Flatten all tweets into one list of words
    word_list = []
    for dataset in all_datasets:
        for tweet in dataset:
            # Tokenize tweet into words, filter by vocabulary and word length
            words = tweet.split()
            valid_words = [
                word for word in words 
                if word in vocabulary and len(word) >= min_word_length
            ]
            word_list.extend(valid_words)
    
    # Count word occurrences
    word_counts = Counter(word_list)
    
    # Get the 100 most common words
    top_100_words = word_counts.most_common(100)
    
    return top_100_words


