# -*- coding: utf-8 -*-
"""
Sentiment Analysis RNN/LSTM — Train Vanilla RNN

Usage::

    python train_rnn.py
"""

import torch
import torch.nn as nn

from config import EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, SENTIMENT_MAP, DEVICE
from data_pipeline import prepare_datasets
from utils import train_model, evaluate_model, predict_sentiment, plot_training_history

# ---------------------------------------------------------------------------
# 1. Model Definition
# ---------------------------------------------------------------------------

class VanillaRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(VanillaRNNClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        out, hidden = self.rnn(embedded)
        final_hidden = hidden.squeeze(0)
        return self.fc(final_hidden)


# ---------------------------------------------------------------------------
# 2. Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    MODEL_NAME = "vanilla_rnn"
    
    # --- Data ---
    train_loader, val_loader, test_loader, word_to_idx = prepare_datasets()
    
    # --- Build ---
    VOCAB_SIZE = len(word_to_idx)
    model = VanillaRNNClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model = model.to(DEVICE)
    print(model)
    
    # --- Train ---
    history = train_model(model, train_loader, val_loader, MODEL_NAME)
    
    # --- Results & Evaluation ---
    plot_training_history(history, MODEL_NAME)
    
    class_names = [SENTIMENT_MAP[0], SENTIMENT_MAP[1], SENTIMENT_MAP[2]]
    evaluate_model(model, test_loader, class_names, MODEL_NAME)
    
    # --- Interactive Testing ---
    print("\nTesting custom sentences...\n")
    
    # Positive Example
    predict_sentiment(model, "Cô giáo dạy rất nhiệt tình và vui vẻ, slide bài giảng chi tiết.", word_to_idx)
    
    # Negative Example
    predict_sentiment(model, "Môn này quá nhàm chán, giảng viên toàn đi trễ.", word_to_idx)
    
    # Neutral / Mixed Example
    predict_sentiment(model, "Em sẽ phải học lại môn này vào kỳ sau.", word_to_idx)
