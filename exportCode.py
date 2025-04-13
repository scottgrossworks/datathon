

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# This formula calculates the number of punctuation marks (.,!, ?, ;) in the text in cell 
def punctuation_count(text):
    punctuation_marks = [".", ",", "!", "?", ";"]
    count = sum(text.count(p) for p in punctuation_marks)
    return count



# This formula measures the verbosity of the response by dividing the total length of the response by the number of words.
def verbosity_score(text):
    total_length = len(text)
    word_count = len(text.split())
    if word_count == 0:
        return 0
    return total_length / word_count



# This formula calculates the number of words in a given response in cell E2 by counting the spaces between words.
def word_count(text):
    words = text.split()
    return len(words)


# This formula calculates the difference in length between the responses (presumably E2 and F2).
def length_difference(text_a, text_b):
    return abs(len(text_a) - len(text_b))




# Example functions for embeddings, you should have your actual embedding calculation here
def embedding_similarity(response_a, response_b):
    # Example: Compute cosine similarity between response embeddings
    # embeddings_a = model.encode(response_a)
    # embeddings_b = model.encode(response_b)
    # similarity = cosine_similarity(embeddings_a, embeddings_b)
    # For now, assuming this is a placeholder for actual model embedding calculation
    similarity = np.random.rand()  # Placeholder
    return similarity



# Example of calculating originality score
def originality_score(response):
    # Placeholder for calculating originality score (e.g., comparing with other responses)
    score = np.random.rand()  # Placeholder for actual originality calculation
    return score




# Integrate all features into MVAT
def mvat_function(response_a, response_b):
    # Punctuation count
    punctuation_a = punctuation_count(response_a)
    punctuation_b = punctuation_count(response_b)
    
    # Verbosity score
    verbosity_a = verbosity_score(response_a)
    verbosity_b = verbosity_score(response_b)
    
    # Word count
    word_count_a = word_count(response_a)
    word_count_b = word_count(response_b)
    
    # Length difference
    length_diff = length_difference(response_a, response_b)
    
    # Embedding similarity (semantic similarity between responses)
    embedding_similarity_score = embedding_similarity(response_a, response_b)
    
    # Originality score (how original is the response)
    originality_a = originality_score(response_a)
    originality_b = originality_score(response_b)
    
    # Combine all features into a single vector to represent both responses
    features_a = [punctuation_a, verbosity_a, word_count_a, len(response_a), embedding_similarity_score, originality_a]
    features_b = [punctuation_b, verbosity_b, word_count_b, len(response_b), embedding_similarity_score, originality_b]
    
    # Feature differences: compare the responses based on all factors
    feature_diff = [a - b for a, b in zip(features_a, features_b)] + [length_diff]
    
    return feature_diff


