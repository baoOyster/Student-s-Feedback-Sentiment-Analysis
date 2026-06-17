# -*- coding: utf-8 -*-
"""
Sentiment Analysis RNN/LSTM — Evaluation Utilities
"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config import RESULTS_DIR, SENTIMENT_MAP, MAX_LEN, DEVICE
from data_pipeline import clean_text, text_to_sequence, pad_sequence


def evaluate_model(model, test_loader, class_names, model_name):
    """Evaluate a trained PyTorch model on the test set and persist results.
    
    Outputs saved to `results/<model_name>/`:
    - `confusion_matrix.png`
    - `classification_report.txt`
    """
    results_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(results_dir, exist_ok=True)
    
    model.eval()
    y_true_list = []
    y_pred_list = []
    
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            predictions = model(inputs)
            
            _, predicted_class = torch.max(predictions, 1)
            
            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(predicted_class.cpu().numpy())
            
            correct_test += (predicted_class == labels).sum().item()
            total_test += labels.size(0)
            
    test_acc = correct_test / total_test
    print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")
    print("-" * 50)
    
    # --- Classification report ---
    report = classification_report(y_true_list, y_pred_list, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n\n")
        f.write(report)
    print(f"Saved → {report_path}")
    
    # --- Confusion matrix ---
    cm = confusion_matrix(y_true_list, y_pred_list)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(f"{model_name} — Confusion Matrix on Test Set")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {cm_path}")
    plt.close() # Close to prevent display issues in some environments
    
    
def predict_sentiment(model, custom_text, word_to_idx):
    """Predict the sentiment of a custom string."""
    model.eval()
    
    with torch.no_grad():
        cleaned = clean_text(custom_text)
        sequence = text_to_sequence(cleaned, word_to_idx)
        padded = pad_sequence(sequence, MAX_LEN)
        
        input_tensor = torch.tensor([padded], dtype=torch.long).to(DEVICE)
        output = model(input_tensor)
        
        probabilities = F.softmax(output, dim=1).squeeze()
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item() * 100
        
        print(f"Review: {custom_text}")
        print(f"Prediction: {SENTIMENT_MAP[predicted_class]}")
        print(f"Confidence proportion: {confidence:.2f}%\n")
