# functions.py
# This file will include all the necessary functions for feature extraction, embedding similarity, 
# and originality scoring. It will also contain the mvat_function that integrates all the features.
#
# scottgross.works
# 
#
#
# Import necessary libraries
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# This formula calculates the number of punctuation marks (.,!, ?, ;) in the text
def punctuation_count(text):
    punctuation_marks = [".", ",", "!", "?", ";"]
    count = sum(text.count(p) for p in punctuation_marks)
    return count

# This formula measures the verbosity of the response by dividing the total length by the number of words
def verbosity_score(text):
    total_length = len(text)
    word_count = len(text.split())
    if word_count == 0:
        return 0
    return total_length / word_count

# This formula calculates the number of words in the response
def word_count(text):
    words = text.split()
    return len(words)

# This formula calculates the difference in length between the responses (text_a and text_b)
def length_difference(text_a, text_b):
    return abs(len(text_a) - len(text_b))

# Example function for calculating embedding similarity
def embedding_similarity(response_a, response_b, tokenizer, model):
    # Tokenize and encode the responses
    inputs_a = tokenizer(response_a, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs_b = tokenizer(response_b, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Get the embeddings from DistilBERT model (you can use a different model if necessary)
    with torch.no_grad():
        embedding_a = model(**inputs_a).last_hidden_state.mean(dim=1)  # Take mean of the token embeddings
        embedding_b = model(**inputs_b).last_hidden_state.mean(dim=1)

    # Calculate cosine similarity between the embeddings
    similarity = cosine_similarity(embedding_a, embedding_b)
    return similarity[0][0]

# Example function for calculating originality score
def originality_score(response, all_responses, tokenizer, model):
    # Compute the similarity of the response with all other responses and calculate a uniqueness score
    similarities = []
    for other_response in all_responses:
        similarity = embedding_similarity(response, other_response, tokenizer, model)
        similarities.append(similarity)
    
    # Calculate an originality score (lower similarity means higher originality)
    originality_score = 1 - np.mean(similarities)  # Higher originality when mean similarity is low
    return originality_score

# Integrating all features into the MVAT function
def mvat_function(response_a, response_b, tokenizer, model, all_responses):
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
    embedding_similarity_score = embedding_similarity(response_a, response_b, tokenizer, model)
    
    # Originality score (how original is the response)
    originality_a = originality_score(response_a, all_responses, tokenizer, model)
    originality_b = originality_score(response_b, all_responses, tokenizer, model)
    
    # Combine all features into a single vector to represent both responses
    features_a = [punctuation_a, verbosity_a, word_count_a, len(response_a), embedding_similarity_score, originality_a]
    features_b = [punctuation_b, verbosity_b, word_count_b, len(response_b), embedding_similarity_score, originality_b]
    
    # Feature differences: compare the responses based on all factors
    feature_diff = [a - b for a, b in zip(features_a, features_b)] + [length_diff]
    
    return feature_diff

