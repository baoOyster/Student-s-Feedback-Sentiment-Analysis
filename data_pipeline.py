# -*- coding: utf-8 -*-
"""
Sentiment Analysis RNN/LSTM — Data Pipeline

Downloads the dataset, cleans text, builds vocabulary, and prepares
PyTorch DataLoaders.

Run this file directly to execute the data preparation pipeline::

    python data_pipeline.py
"""

import os
import re
import json
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    DATA_DIR, TRAIN_URL, VAL_URL, TEST_URL, 
    MIN_FREQ, MAX_LEN, BATCH_SIZE
)


# ---------------------------------------------------------------------------
# 1. Dataset download
# ---------------------------------------------------------------------------

def download_datasets():
    """Download Parquet files and return train, val, test DataFrames."""
    print("Downloading dataset directly via Parquet...")
    
    train_df = pd.read_parquet(TRAIN_URL)
    val_df = pd.read_parquet(VAL_URL)
    test_df = pd.read_parquet(TEST_URL)
    
    train_df = train_df[['sentence', 'sentiment']]
    val_df = val_df[['sentence', 'sentiment']]
    test_df = test_df[['sentence', 'sentiment']]
    
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 2. Text preprocessing & Vocabulary Building
# ---------------------------------------------------------------------------

def clean_text(text):
    """Clean Vietnamese text."""
    text = text.lower()
    # Remove punctuation, keep only letters/numbers and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_vocabulary(train_df):
    """Build or load vocabulary dictionary from training data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    vocab_path = os.path.join(DATA_DIR, "word_to_idx.json")
    
    if os.path.exists(vocab_path):
        print(f"Loading cached vocabulary from {vocab_path}...")
        with open(vocab_path, "r", encoding="utf-8") as f:
            word_to_idx = json.load(f)
    else:
        print("Building Vocabulary...")
        word_counts = Counter()
        
        for sentence in train_df['clean_sentence']:
            word_counts.update(sentence.split())
            
        valid_words = [word for word, count in word_counts.items() if count >= MIN_FREQ]
        
        word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, word in enumerate(valid_words, start=2):
            word_to_idx[word] = idx
            
        print(f"Total unique words in Vocabulary: {len(word_to_idx)}")
        
        # Cache the vocabulary
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(word_to_idx, f, ensure_ascii=False, indent=2)
            
    return word_to_idx


def text_to_sequence(text, word_to_idx):
    """Convert a cleaned string into a list of vocabulary indices."""
    words = text.split()
    return [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]


def pad_sequence(seq, max_len):
    """Pad or truncate a sequence to the specified max_len."""
    # Assuming <PAD> index is always 0
    pad_token = 0 
    if len(seq) < max_len:
        # Too short, add 0s to the end
        return seq + [pad_token] * (max_len - len(seq))
    else:
        # Too long, chop off the end
        return seq[:max_len]


# ---------------------------------------------------------------------------
# 3. PyTorch Dataset & DataLoaders
# ---------------------------------------------------------------------------

class FeedbackDataset(Dataset):
    """PyTorch Dataset for Student Feedback."""
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences.tolist(), dtype=torch.long)
        self.labels = torch.tensor(labels.tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def prepare_datasets():
    """Run the full pipeline: download, clean, vocab, pad, DataLoader.
    
    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader, word_to_idx)
    """
    train_df, val_df, test_df = download_datasets()
    
    print("\n--- Training Data Sample ---")
    print(train_df.head())
    print(f"\nTraining samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Testing samples: {len(test_df)}")
    
    # 1. Clean Text
    train_df['clean_sentence'] = train_df['sentence'].apply(clean_text)
    val_df['clean_sentence'] = val_df['sentence'].apply(clean_text)
    test_df['clean_sentence'] = test_df['sentence'].apply(clean_text)
    
    # 2. Build/Load Vocab
    word_to_idx = build_vocabulary(train_df)
    
    # 3. Convert to Sequences
    train_df['sequence'] = train_df['clean_sentence'].apply(lambda x: text_to_sequence(x, word_to_idx))
    val_df['sequence'] = val_df['clean_sentence'].apply(lambda x: text_to_sequence(x, word_to_idx))
    test_df['sequence'] = test_df['clean_sentence'].apply(lambda x: text_to_sequence(x, word_to_idx))
    
    # 4. Pad Sequences
    train_df['padded'] = train_df['sequence'].apply(lambda x: pad_sequence(x, MAX_LEN))
    val_df['padded'] = val_df['sequence'].apply(lambda x: pad_sequence(x, MAX_LEN))
    test_df['padded'] = test_df['sequence'].apply(lambda x: pad_sequence(x, MAX_LEN))
    
    # 5. Create PyTorch Datasets
    train_dataset = FeedbackDataset(train_df['padded'], train_df['sentiment'])
    val_dataset = FeedbackDataset(val_df['padded'], val_df['sentiment'])
    test_dataset = FeedbackDataset(test_df['padded'], test_df['sentiment'])
    
    # 6. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    y_train = train_df['sentiment'].values

    return train_loader, val_loader, test_loader, word_to_idx, y_train


# ---------------------------------------------------------------------------
# 4. Main - run full pipeline when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_loader, val_loader, test_loader, word_to_idx, y_train = prepare_datasets()
    
    print("\n--- Testing the DataLoader ---")
    sample_inputs, sample_labels = next(iter(train_loader))
    print(f"Input batch shape: {sample_inputs.shape} --> (Batch Size, Sequence Length)")
    print(f"Labels batch shape: {sample_labels.shape} --> (Batch Size)")
    print(f"\nFirst sentence in batch (padded numbers):\n{sample_inputs[0]}")
    print(f"Label for that sentence: {sample_labels[0]}")
