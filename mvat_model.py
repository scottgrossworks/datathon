#
# mvat_model.py
#
# run the MVAT test and generate results
#
# This main() function provides a clean entry point and can be run directly as a standalone script.
#
# scottgross.works
#


import torch
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load test data from CSV."""
    return pd.read_csv(file_path)

def train_random_forest_model(X_train, y_train):
    """Train Random Forest model with training data."""
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy}')
    return accuracy

def plot_results(accuracy):
    """Generate and display results plot for PPT."""
    plt.bar([1, 2], [accuracy, 1 - accuracy], tick_label=['Accuracy', 'Error'])
    plt.title('MVAT Model Performance')
    plt.ylabel('Percentage')
    plt.show()

def main():
    # Step 1: Load the data
    test_df = load_data('test_500.csv')
    
    # Step 2: Prepare your feature set and target labels
    X_test = test_df[['syntactic', 'semantic', 'heuristic', 'llm_features']].values  # Adjust according to actual columns
    y_test = test_df['LABEL'].values

    # Step 3: Train or load your model (if already trained, load it)
    rf_model = RandomForestClassifier()  # Replace with the actual trained model if needed

    # Step 4: Evaluate the model
    accuracy = evaluate_model(rf_model, X_test, y_test)

    # Step 5: Plot and show the results
    plot_results(accuracy)

if __name__ == '__main__':
    main()
