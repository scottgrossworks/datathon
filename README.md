#
# README.txt
# 
# scottgross.works
#
#
# Project Title: Response Evaluation with MVAT

## Overview
This project evaluates responses using a Multi-Vector Assessment Tree (MVAT) approach. It aggregates multiple features into a hierarchical structure to provide a holistic assessment of response quality. The project includes model training, evaluation, and results visualization in PowerPoint format.

## Project Files

### `train_llm.py`
This script is responsible for training the local Large Language Model (LLM). It defines the training process and model parameters, and logs training progress.

### `mvat_model.py`
Implements the MVAT approach which scores responses by combining features like syntactic, semantic, heuristic, and LLM-based scores. This model makes the final decision based on the aggregated feature scores.

### `load_data.py`
Handles the loading and preprocessing of the dataset (`test_500.csv`). It uses tokenization, padding, and batching to prepare the data for training and evaluation.

### `gen_ppt.py`
Generates a PowerPoint presentation summarizing the results from model evaluation, including evaluation metrics and MVAT scores.

### `functions.py`
Provides various helper functions for feature extraction including punctuation count, verbosity score, word count, and more. These features are then used in the MVAT scoring system.

### `extracted_code.py`
Contains code for feature extraction used in MVAT scoring, such as tokenization, feature aggregation, and response comparisons.

### `exportCode.py`
Calculates features such as punctuation count, verbosity score, word count, and embedding similarity. These features are then used in the MVAT function to compare responses.

### `eval_code.py`
Evaluates the model on the dataset, calculates the evaluation loss, and measures accuracy. It also includes functions for processing the data and handling batching.

## How the Files Tie Together:
- **Data Processing**: `load_data.py` handles loading and tokenizing the dataset.
- **Model Training**: `train_llm.py` is used to train the model using the preprocessed data.
- **MVAT Scoring**: The actual MVAT algorithm is implemented in `mvat_model.py` and uses features from `functions.py`, `exportCode.py`, and `extracted_code.py`.
- **Evaluation**: `eval_code.py` is responsible for evaluating the trained model's performance on the test dataset.
- **PowerPoint Generation**: `gen_ppt.py` generates the PowerPoint slides based on the evaluation results and the MVAT approach.




Dependencies:
torch

transformers

sklearn

datasets

pandas

numpy

matplotlib



## How to Run the Project:
1. **Step 1**: Preprocess data using `load_data.py`:
$python load_data.py

Step 2: Train the model with train_llm.py:
$python train_llm.py

Step 3: Evaluate the model using eval_code.py:
$ python eval_code.py

Step 4: Generate the PowerPoint slides with gen_ppt.py:
$ python gen_ppt.py



License:
This project is licensed under the MIT License - see the LICENSE file for details.
