# -*- coding: utf-8 -*-
"""
Sentiment Analysis RNN/LSTM — Plotting Utilities
"""

import os
import matplotlib.pyplot as plt

from config import RESULTS_DIR


def plot_training_history(history, model_name):
    """Plot training & validation accuracy/loss and save to disk.
    
    The figure is saved to `results/<model_name>/training_history.png`.
    
    Parameters
    ----------
    history : dict
        A dictionary containing 'loss', 'accuracy', 'val_loss', 'val_accuracy'
    model_name : str
        Short identifier used to create the output sub-directory.
    """
    results_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(results_dir, exist_ok=True)
    
    train_loss = history['loss']
    val_loss   = history['val_loss']
    train_acc  = history['accuracy']
    val_acc    = history['val_accuracy']
    epochs     = range(1, len(train_loss) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # — Loss —
    ax1.plot(epochs, train_loss, "b-", marker="o", label="Training Loss")
    ax1.plot(epochs, val_loss,   "r-", marker="s", label="Validation Loss")
    ax1.set_title(f"{model_name} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)
    
    # — Accuracy —
    ax2.plot(epochs, train_acc, "b-", marker="o", label="Training Accuracy")
    ax2.plot(epochs, val_acc,   "r-", marker="s", label="Validation Accuracy")
    ax2.set_title(f"{model_name} — Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, "training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close()
