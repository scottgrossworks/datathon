# train_llm.py
#
# This file will handle all aspects related to training the LLM (Local Language Model), 
# including preparing the dataset, training the model, and running the training pipeline.
#
# scottgross.works
#


# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from functions import mvat_function  # Import the function from functions.py

# Example code to test MVAT with a dataset
def prepare_dataset():
    # Example dataset of responses (to be replaced with actual data)
    responses_a = ["Response A1", "Response A2", "Response A3"]
    responses_b = ["Response B1", "Response B2", "Response B3"]
    labels = [1, 0, 1]  # Placeholder labels: 0 for B wins, 1 for A wins
    
    # Tokenizer and model initialization (use a pre-trained DistilBERT)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    # Prepare data (compute features)
    all_responses = responses_a + responses_b
    features = []
    for a, b in zip(responses_a, responses_b):
        feature_diff = mvat_function(a, b, tokenizer, model, all_responses)
        features.append(feature_diff)
    
    return features, labels

# Main execution to train a model using the extracted features
def train_model():
    # Prepare the dataset
    features, labels = prepare_dataset()
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Initialize Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    classifier.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = classifier.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy}")

# Ensure that the training process only runs when this script is executed directly
if __name__ == "__main__":
    train_model()
