# Sentiment Analysis: RNN & LSTM

This project analyzes student feedback using recurrent neural networks (Vanilla RNN and LSTM) built with PyTorch. It predicts sentiment across three classes: **Negative (0)**, **Neutral (1)**, and **Positive (2)**.

The architecture follows a modular, scalable pattern allowing new models to be added with minimal code duplication.

## Project Structure

```text
week5_6_student_feedback_analysis_RNN_LSTM/
├── config.py              # Shared paths, hyperparameters, and device config
├── data_pipeline.py       # Data downloading, cleaning, vocab building, DataLoaders
├── train_rnn.py           # Model definition & training execution for Vanilla RNN
├── train_lstm.py          # Model definition & training execution for LSTM
├── sentiment_analysis_rnn_yanjiabao.py  # Original monolothic Colab reference file
├── requirements.txt       # Project dependencies
├── utils/                 # Reusable utility modules
│   ├── training.py        # Generic PyTorch training loop with checkpointing
│   ├── evaluation.py      # Classification reports, confusion matrices, inference
│   └── plotting.py        # Training history plots
├── data/                  # Downloaded splits & cached vocabulary
├── checkpoints/           # Saved model weights
├── logs/                  # Training CSV logs
└── results/               # Evaluation plots and reports
```

## Setup & Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. *(Optional)* Run the data pipeline standalone to download data and build the vocabulary cache:
   ```bash
   python data_pipeline.py
   ```

## Training

Each model has its own dedicated training script. The script handles building the model, training, evaluating on the test set, saving plots, and running interactive predictions.

**To train the Vanilla RNN:**
```bash
python train_rnn.py
```

**To train the LSTM:**
```bash
python train_lstm.py
```

## Outputs

After training, the following artifacts are automatically generated for each model:

*   **`checkpoints/<model_name>/best_model.pt`**: The model weights with the highest validation accuracy.
*   **`logs/<model_name>_training_log.csv`**: Epoch-by-epoch loss and accuracy records.
*   **`results/<model_name>/training_history.png`**: Plots of the training/validation curves.
*   **`results/<model_name>/confusion_matrix.png`**: Heatmap of test set performance.
*   **`results/<model_name>/classification_report.txt`**: Precision, recall, and f1-score.
