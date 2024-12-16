
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

# Function to calculate word or character counts
def count_metric(tweet, count_words=True):
    if count_words:
        return len(word_tokenize(tweet))  # Count words
    else:
        return len(tweet)  # Count characters

# Main plotting function
def tweet_length_plotter(data_paths, clean_paths, count_words=True, normalize=True, plot_type='hist'):
    """
    Function to plot tweet lengths (words or characters) for given datasets.

    Args:
        data_paths (list): List of file paths for raw datasets.
        clean_paths (list): List of file paths for datasets to clean.
        count_words (bool): If True, count words; otherwise, count characters.
        normalize (bool): If True, normalize histograms.
        plot_type (str): 'hist' for histograms, 'kde' for Kernel Density Plots.
    """
    # Load datasets
    datasets = [load_tweets(path) for path in data_paths]
    cleaned_datasets = [load_and_clean_test_tweets(path) for path in clean_paths]
    
    # Combine all datasets
    all_datasets = datasets + cleaned_datasets
    dataset_labels = ['Train Negative Tweets', 'Train Positive Tweets', 'Test Tweets (Cleaned)']
    
    # Calculate lengths
    dataset_lengths = [
        [count_metric(tweet, count_words) for tweet in dataset] 
        for dataset in all_datasets
    ]
    
    # Plot setup
    plt.figure(figsize=(12, 6))
    metric = 'Words' if count_words else 'Characters'

    # Choose plot type
    if plot_type == 'hist':
        for lengths, label, color in zip(dataset_lengths, dataset_labels, ['red', 'green', 'blue']):
            plt.hist(lengths, bins=50, alpha=0.5, label=label, color=color, density=normalize)
        plt.ylabel('Density' if normalize else 'Number of Tweets')
    elif plot_type == 'kde':
        for lengths, label, color in zip(dataset_lengths, dataset_labels, ['red', 'green', 'blue']):
            sns.kdeplot(lengths, shade=True, label=label, color=color)
        plt.ylabel('Density')
    else:
        raise ValueError("Invalid plot_type. Use 'hist' or 'kde'.")
    
    # Add plot details
    plt.xlabel(f'Number of {metric} per Tweet')
    # plt.title(f'{plot_type.upper()} Plot of Tweet {metric} Counts' + (' (Normalized)' if normalize else ''))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
