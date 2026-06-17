# -*- coding: utf-8 -*-
"""
Sentiment Analysis RNN/LSTM — Shared Configuration
"""

import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR        = os.path.join(BASE_DIR, "logs")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
DATA_DIR       = os.path.join(BASE_DIR, "data")       # cached train/val/test splits & vocab

# ---------------------------------------------------------------------------
# Data URLs
# ---------------------------------------------------------------------------

TRAIN_URL = "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
VAL_URL   = "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"
TEST_URL  = "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"

# ---------------------------------------------------------------------------
# Hyperparameters & constants
# ---------------------------------------------------------------------------

MIN_FREQ      = 2
MAX_LEN       = 25
BATCH_SIZE    = 32
EMBEDDING_DIM = 100
HIDDEN_DIM    = 128
OUTPUT_DIM    = 3 # 3 classes: Negative (0), Neutral (1), Positive (2)
EPOCHS        = 5
LEARNING_RATE = 0.001

SENTIMENT_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# ---------------------------------------------------------------------------
# Device Configuration
# ---------------------------------------------------------------------------

def get_device():
    try:
        # TPU check
        import torch_xla
        return torch_xla.device()
    except ImportError:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEVICE = get_device()
