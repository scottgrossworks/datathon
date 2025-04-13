#
# eval_code.py
# Full Evaluation Code (with Model Loading)
#
# scottgross.works
#
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
import numpy as np

# Load the test dataset
test_df = load_dataset('csv', data_files={'test': 'test_500.csv'}, split='test')

# Initialize the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('./finetuned_model')

# Define a custom collate function to handle padding, truncation, and batching correctly
def collate_fn(batch):
    # Tokenize all the inputs
    input_encodings = tokenizer(
        [item['prompt'] for item in batch],
        [item['response_a'] for item in batch],
        [item['response_b'] for item in batch],
        padding=True,               # Automatically pad the sequences to the longest one in the batch
        truncation=True,            # Handle truncation of long sequences
        max_length=512,             # Set the maximum length for tokenization
        return_tensors='pt',        # Return tensors
        return_attention_mask=True,
        padding_side="right",       # Ensure padding is done on the right side for uniformity
        truncation_strategy="longest_first"  # Using the longest_first truncation strategy
    )
    
    # Add labels from the batch to the encodings
    labels = torch.tensor([item['LABEL'] for item in batch])
    input_encodings['labels'] = labels
    
    return input_encodings

# Create a DataLoader to load the dataset in batches
test_dataloader = DataLoader(test_df, batch_size=8, collate_fn=collate_fn)

# Define the evaluation function
def evaluate(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    total_eval_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for batch in dataloader:
        with torch.no_grad():
            # Move batch to the GPU (if available)
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            # Perform the forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_eval_loss += loss.item()

            # Convert logits to predictions
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    avg_loss = total_eval_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

# Run the evaluation
eval_loss, eval_accuracy = evaluate(model, test_dataloader)

# Print the results
print(f"Evaluation Loss: {eval_loss:.4f}")
print(f"Evaluation Accuracy: {eval_accuracy:.4f}")
