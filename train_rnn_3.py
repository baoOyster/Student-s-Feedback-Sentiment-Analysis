# -*- coding: utf-8 -*-
"""
Sentiment Analysis RNN/LSTM — Train Custom Deep Res RNN
with batch normalization and gradient clipping

Usage::

    python train_rnn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from config import OUTPUT_DIM, SENTIMENT_MAP, DEVICE, BATCH_SIZE, MAX_LEN
from data_pipeline import prepare_datasets
from utils import train_model, evaluate_model, predict_sentiment, plot_training_history

# ---------------------------------------------------------------------------
# 1. Model Definition
# ---------------------------------------------------------------------------

class CustomDeepResRNN_3(nn.Module):
    def __init__(self, vocab_size, num_classes=3):
        super(CustomDeepResRNN_3, self).__init__()
        
        # --- INITIAL LAYERS ---
        # Instead of rescaling pixels, we embed input token IDs to 100-dim vectors
        self.embedding = nn.Embedding(vocab_size, 100, padding_idx=0)
        
        # RNN equivalents to your initial Conv2D layers
        self.rnn1 = nn.RNN(100, 32, batch_first=True)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.rnn2 = nn.RNN(32, 32, batch_first=True)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.rnn3 = nn.RNN(32, 24, batch_first=True)
        # Replacing MaxPooling with downsampling is rare in text, 
        # so we pass the full sequence representation through to match your shape flow.
        
        # --- RESIDUAL BLOCK 1 ---
        self.b2_rnn1 = nn.RNN(24, 40, batch_first=True)
        self.b2_bn1 = nn.BatchNorm1d(40)
        
        self.b2_rnn2 = nn.RNN(40, 40, batch_first=True)
        self.b2_bn2 = nn.BatchNorm1d(40)
        
        # Conv1d with kernel 1 projects the sequence channels from 24 to 40 for the shortcut
        self.skip1 = nn.Conv1d(24, 40, kernel_size=1)
        
        # --- RESIDUAL BLOCK 2 ---
        self.b3_rnn1 = nn.RNN(40, 80, batch_first=True)
        self.b3_bn1 = nn.BatchNorm1d(80)
        
        self.b3_rnn2 = nn.RNN(80, 80, batch_first=True)
        self.b3_bn2 = nn.BatchNorm1d(80)
        
        # Connects to shortcut_1 (24 channels) and scales it up to 80 channels
        self.skip2 = nn.Conv1d(24, 80, kernel_size=1)
        
        # --- FINAL RNN LAYERS ---
        self.rnn4 = nn.RNN(80, 112, batch_first=True)
        self.bn4 = nn.BatchNorm1d(112)
        
        self.rnn5 = nn.RNN(112, 112, batch_first=True)
        self.bn5 = nn.BatchNorm1d(112)
        
        # --- DENSE CLASSIFIER HEAD (Identical to your CNN) ---
        self.fc1 = nn.Linear(112, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.4)
        
        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: [Batch, Sequence_Length]
        x = self.embedding(x) # [Batch, Sequence_Length, 100]
        
        # --- Initial Sequential Processing ---
        x, _ = self.rnn1(x)
        # PyTorch BatchNorm1d expects [Batch, Features, Sequence_Length]
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(x)).permute(0, 2, 1)
        
        x, _ = self.rnn2(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn2(x)).permute(0, 2, 1)
        
        x, _ = self.rnn3(x) # Output shape: [Batch, Sequence_Length, 24]
        
        # Cache shortcuts (Permute to [Batch, 24, Sequence_Length] for Conv1D compatibility)
        shortcut = x.permute(0, 2, 1)
        shortcut_1 = x.permute(0, 2, 1)
        
        # --- Residual Block 1 ---
        x_b2, _ = self.b2_rnn1(x)
        x_b2 = x_b2.permute(0, 2, 1)
        x_b2 = F.relu(self.b2_bn1(x_b2)).permute(0, 2, 1)
        
        x_b2, _ = self.b2_rnn2(x_b2)
        x_b2 = x_b2.permute(0, 2, 1)
        x_b2 = self.b2_bn2(x_b2)
        
        s1 = self.skip1(shortcut) # Project shortcut to 40 channels
        x = F.relu(x_b2 + s1) # Element-wise Addition
        x = x.permute(0, 2, 1) # Return to [Batch, Sequence_Length, 40]
        
        # --- Residual Block 2 ---
        x_b3, _ = self.b3_rnn1(x)
        x_b3 = x_b3.permute(0, 2, 1)
        x_b3 = F.relu(self.b3_bn1(x_b3)).permute(0, 2, 1)
        
        x_b3, _ = self.b3_rnn2(x_b3)
        x_b3 = x_b3.permute(0, 2, 1)
        x_b3 = self.b3_bn2(x_b3)
        
        s2 = self.skip2(shortcut_1) # Project shortcut_1 to 80 channels
        x = F.relu(x_b3 + s2)
        x = x.permute(0, 2, 1) # Return to [Batch, Sequence_Length, 80]
        
        # --- Final Processing Layers ---
        x, _ = self.rnn4(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn4(x)).permute(0, 2, 1)
        
        x, _ = self.rnn5(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn5(x)) # Leaves x at [Batch, 112, Sequence_Length]
        
        # --- Global Max Pooling ---
        # Performs GlobalMaxPooling1D over the entire sequence duration dimension
        x = F.adaptive_max_pool1d(x, 1) 
        x = torch.flatten(x, 1) # Shape: [Batch, 112]
        
        # --- Fully Connected Classifier Head ---
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.drop2(x)
        
        return self.output(x)


# ---------------------------------------------------------------------------
# 2. Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    MODEL_NAME = "custom_deep_res_rnn_3"
    
    # --- Data ---
    train_loader, val_loader, test_loader, word_to_idx, y_train = prepare_datasets()
    
    # --- Calculate Class Weights ---
    print("\n--- Calculating Class Weights ---")
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

    for i, weight in enumerate(cw):
        print(f"Class {i} Weight: {weight:.4f}")

    # Convert to PyTorch tensor
    class_weights_tensor = torch.tensor(cw, dtype=torch.float, device=DEVICE)

    # --- Build ---
    VOCAB_SIZE = len(word_to_idx)
    model = CustomDeepResRNN_3(vocab_size=VOCAB_SIZE, num_classes=OUTPUT_DIM)
    model = model.to(DEVICE)
    
    # View the explicit pipeline matrix
    print("\n--- Model Summary ---")
    # Setting device for summary correctly
    summary(model, input_size=(BATCH_SIZE, MAX_LEN), dtypes=[torch.long], device=DEVICE)
    
    # --- Train ---
    history = train_model(
        model = model, 
        train_loader = train_loader, 
        val_loader = val_loader, 
        model_name = MODEL_NAME,
        class_weights = class_weights_tensor,
        grad_clip = 1.0
    )
    
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
