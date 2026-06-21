# Sentiment Analysis: Deep Residual RNN

This project analyzes student feedback using deep learning models built with PyTorch. It predicts sentiment across three classes: **Negative (0)**, **Neutral (1)**, and **Positive (2)**.

> [!NOTE]
> **Architecture Deep Dive:** For a detailed look at the evolution of our models, including performance metrics across different normalization and recurrent unit strategies (RNN vs LSTM), please see [ABOUT.md](./ABOUT.md).

## Project Structure

```text
week5_6_student_feedback_analysis_RNN_LSTM/
├── config.py              # Shared paths, hyperparameters (EPOCHS=100, PATIENCE=10)
├── data_pipeline.py       # Data downloading, text cleaning, vocab building
├── train_rnn.py           # CustomDeepResRNN definition & training
├── requirements.txt       # Project dependencies
├── utils/                 
│   ├── training.py        # Training loop with Checkpointing & Early Stopping
│   ├── evaluation.py      # Classification reports, confusion matrices
│   └── plotting.py        # Training history plots
```

## Architecture: CustomDeepResRNN

The core model in `train_rnn.py` has been updated to a `CustomDeepResRNN`. It moves beyond vanilla RNNs by incorporating deep sequence processing, Batch Normalization between time-steps, and Convolutional Shortcut connections (Residual Blocks).

```mermaid
flowchart TD
    In["Input: Token IDs"] --> Emb["Embedding (100-dim)"]
    
    subgraph "Initial Sequential Layers"
    Emb --> R1["RNN 1 (32 units) + BN + ReLU"]
    R1 --> R2["RNN 2 (32 units) + BN + ReLU"]
    R2 --> R3["RNN 3 (24 units)"]
    end
    
    R3 --> B2_R1
    R3 --> S1["Shortcut (24 channels)"]
    R3 --> S2["Shortcut_1 (24 channels)"]
    
    subgraph "Residual Block 1"
    B2_R1["RNN (40 units) + BN + ReLU"] --> B2_R2["RNN (40 units) + BN"]
    end
    
    B2_R2 --> Add1((+))
    S1 -- "Conv1d (24→40)" --> Add1
    Add1 -- "ReLU" --> B3_R1
    
    subgraph "Residual Block 2"
    B3_R1["RNN (80 units) + BN + ReLU"] --> B3_R2["RNN (80 units) + BN"]
    end
    
    B3_R2 --> Add2((+))
    S2 -- "Conv1d (24→80)" --> Add2
    Add2 -- "ReLU" --> R4
    
    subgraph "Final RNN Layers"
    R4["RNN 4 (112 units) + BN + ReLU"] --> R5["RNN 5 (112 units) + BN + ReLU"]
    end
    
    R5 --> Pool["Global 1D Max Pooling"]
    
    subgraph "Classifier Head"
    Pool --> FC1["Linear (512) + BN + ReLU + Dropout(0.5)"]
    FC1 --> FC2["Linear (128) + BN + ReLU + Dropout(0.4)"]
    FC2 --> Out["Linear (3 classes)"]
    end
```

### Key Features
* **100 Epoch Training with Early Stopping:** `config.py` is set to train up to 100 epochs, but `utils/training.py` implements an early stopping mechanism that safely halts training and restores best weights if validation accuracy doesn't improve for 10 consecutive epochs (`PATIENCE=10`).
* **Residual Connections:** Time-distributed 1D convolutions project the dimensionality of sequence shortcuts to match block outputs, preventing vanishing gradients in deep RNN stacks.
* **Global Max Pooling:** Consolidates the final sequence features efficiently before passing to a heavy dense classifier.

## Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Deep Res RNN pipeline**:
   This single script automatically prepares the data pipeline, trains the model with early stopping, plots validation curves, and performs interactive testing.
   ```bash
   python train_rnn.py
   ```

## Outputs

*   **`checkpoints/custom_deep_res_rnn/best_model.pt`**: The optimal weights retrieved via Early Stopping.
*   **`logs/custom_deep_res_rnn_training_log.csv`**: Historical epoch metrics.
*   **`results/custom_deep_res_rnn/...`**: Training plots, confusion matrix, and classification report.
