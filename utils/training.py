# -*- coding: utf-8 -*-
"""
Sentiment Analysis RNN/LSTM — Training Utility
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from config import CHECKPOINT_DIR, LOG_DIR, EPOCHS, PATIENCE, LEARNING_RATE, DEVICE

def train_model(model, train_loader, val_loader, model_name, class_weights=None, grad_clip=None, learning_rate=LEARNING_RATE):
    """Generic PyTorch training loop for RNN/LSTM models.
    
    Logs to CSV and saves the best model checkpoint.
    
    Returns
    -------
    dict
        A history dictionary with keys: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
    """
    ckpt_dir = os.path.join(CHECKPOINT_DIR, model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    log_file = os.path.join(LOG_DIR, f"{model_name}_training_log.csv")
    best_model_path = os.path.join(ckpt_dir, "best_model.pt")
    
    if class_weights is not None:
        class_weights = class_weights.to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    print(f"Starting Training Loop for {model_name} on {DEVICE}...\n")
    
    # Initialize CSV Log
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])
    
    print(f"Training for {EPOCHS} epochs with patience {PATIENCE}...\n")

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            try:
                # Update weights for TPU if present
                import torch_xla.core.xla_model as xm
                xm.optimizer_step(optimizer)
            except ImportError:
                optimizer.step()
                
            total_train_loss += loss.item()
            _, predicted_class = torch.max(predictions, 1)
            correct_train += (predicted_class == labels).sum().item()
            total_train += labels.size(0)
            
        train_acc = correct_train / total_train
        train_loss = total_train_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                predictions = model(inputs)
                loss = criterion(predictions, labels)
                
                total_val_loss += loss.item()
                _, predicted_class = torch.max(predictions, 1)
                correct_val += (predicted_class == labels).sum().item()
                total_val += labels.size(0)
                
        val_acc = correct_val / total_val
        val_loss = total_val_loss / len(val_loader)
        
        # Save history
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Log to CSV
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])
            
        # Checkpoint saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            print(f"--> Validation accuracy improved. Saving model to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            print(f"--> No improvement in validation accuracy for {epochs_no_improve} epochs.")
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                if os.path.exists(best_model_path):
                    print("Restoring best model weights...")
                    model.load_state_dict(torch.load(best_model_path, weights_only=True))
                break
            
    print("\nTraining Complete!")
    return history
