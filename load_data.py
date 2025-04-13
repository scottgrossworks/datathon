import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments

# Step 1: Load the test data
test_df = pd.read_csv('test_500.csv')  # Ensure path is correct

# Custom Dataset class for tokenization
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data.iloc[idx]['prompt']
        response_a = self.data.iloc[idx]['response_a']
        response_b = self.data.iloc[idx]['response_b']

        # Tokenize input and output with padding and truncation handling
        encoding = self.tokenizer(
            prompt,
            response_a,
            response_b,
            padding='max_length',  # This was max_length but we will change it to 'longest' below
            truncation=True,  # This is applied to all sequences
            max_length=self.max_length,
            return_tensors='pt',  # Return PyTorch tensors
            return_attention_mask=True
        )

        # Instead of using torch.tensor(), directly squeeze tensor
        item = {key: val.squeeze() for key, val in encoding.items()}  # Squeeze to remove extra dimensions
        item['labels'] = torch.tensor(self.data.iloc[idx]['LABEL'])
        return item

# Dynamic padding and truncation strategy
def collate_fn(batch):
    # dynamically padding the sequences in the batch
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Using pad_sequence to pad the sequences in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# Example: Loading data into a DataLoader
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
test_dataset = CustomDataset(test_df, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)

# Step 6: Set up evaluation arguments
eval_args = TrainingArguments(
    output_dir='./results',            
    per_device_eval_batch_size=8,      # Batch size for evaluation
    do_eval=True,                      # Perform evaluation
)

# Step 7: Load the fine-tuned model
model = DistilBertForSequenceClassification.from_pretrained('./finetuned_model')

# Step 8: Initialize Trainer
trainer = Trainer(
    model=model,                       
    args=eval_args,                    
    eval_dataset=test_dataset,         # Use the test dataset
    data_collator=collate_fn           # Use the custom collate function
)

# Step 9: Run evaluation
eval_results = trainer.evaluate()

# Step 10: Print evaluation results
print(eval_results)
