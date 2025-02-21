import torch
from torch import optim
import torch.nn as nn
import argparse
from create_data import create_data_loaders, create_data_loaders_transformers
from models.CNN import PromoterCNN, FocalLoss
from models.Transformer import DNABERT_TransformerClassifier
import numpy as np
import time
from tools import find_optimal_threshold
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef

# CNN Training and Evaluation

def train(model, train_loader, device, num_epochs=20, patience=3):
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_loss = train_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Checkpointing
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'transformer_model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_model.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
        # Early stopping
        if early_stopping_counter >= patience:
            print(f'Early stopping triggered after epoch {epoch+1}')
            break
    
    # Load best model before returning
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

def evaluate(model, test_loader, y_test, device, threshold=0.5):
    model.eval()
    y_pred_probs = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_pred_probs.extend(model(X_batch).cpu().numpy())

    y_pred_probs = np.array(y_pred_probs).ravel()
    
    optimal_threshold = find_optimal_threshold(y_test.numpy(), y_pred_probs)

    y_pred = (y_pred_probs >= optimal_threshold).astype(int)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test.numpy(), y_pred).ravel()

    # Compute classification metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  
    accuracy = accuracy_score(y_test.numpy(), y_pred)  
    mcc = matthews_corrcoef(y_test.numpy(), y_pred) 

    # Print and return metrics
    print(f"Sensitivity (Sn): {sensitivity:.4f}")
    print(f"Specificity (Sp): {specificity:.4f}")
    print(f"Accuracy (Acc): {accuracy:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

# Transformer Training and Evaluation

def train_transformer(model, train_loader, device, num_epochs=20, patience=3):
    model.train()  # Set model to training mode
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()
    best_loss = float('inf')
    early_stopping_counter = 0

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        total_train_loss = 0
        correct = 0
        total = 0

        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model({"input_ids": input_ids, "attention_mask": attention_mask})
            
            # Ensure shapes match
            outputs = outputs.squeeze()
            labels = labels.float()
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track statistics
            total_train_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Print batch progress
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f}')

        # Calculate epoch statistics
        avg_train_loss = total_train_loss / len(train_loader)
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Average Loss: {avg_train_loss:.4f} | "
              f"Accuracy: {accuracy:.4f}")

        # Save checkpoint if loss improved
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            early_stopping_counter = 0
            print(f"Loss improved to {best_loss:.4f}. Saving model...")
            
            # Save the model
            torch.save({
                'epoch': epoch,
                'transformer_model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_transformer_model.pth')
            print(f"Model saved at epoch {epoch+1}")
        else:
            early_stopping_counter += 1
            print(f"Loss did not improve. Counter: {early_stopping_counter}/{patience}")
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    print("Training completed!")
    return model

def evaluate_transformer(model, test_loader, test_labels, device):
    checkpoint = torch.load('best_transformer_model.pth')
    model.load_state_dict(checkpoint['transformer_model_state_dict'])
    model.to(device)
    model.eval()

    y_pred_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model({"input_ids": input_ids, "attention_mask": attention_mask})
            y_pred_probs.extend(outputs.cpu().numpy())

    y_pred_probs = np.array(y_pred_probs).ravel()
    print(y_pred_probs)
    # Remove .numpy() since test_labels is already a numpy array
    optimal_threshold = find_optimal_threshold(test_labels, y_pred_probs)
    y_pred = (y_pred_probs >= optimal_threshold).astype(int)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()
    print(y_pred)
    print(test_labels)
    # Compute classification metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = accuracy_score(test_labels, y_pred)
    mcc = matthews_corrcoef(test_labels, y_pred)

    # Print evaluation metrics
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Sensitivity (Sn): {sensitivity:.4f}")
    print(f"Specificity (Sp): {specificity:.4f}")
    print(f"Accuracy (Acc): {accuracy:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CNN', choices=['CNN', 'transformer'],
                      help='Choose model type: CNN or transformer')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--num_epochs', type=int, default=10, 
                      help='Number of epochs for training')
    parser.add_argument('--subset_size', type=int, default=None,
                      help='Use subset of data for quick testing (e.g., 100 samples)')
    parser.add_argument('--prom_path', type=str, default='promo_dataset/Ecoli_prom.fa', 
                      help='Path to promoter dataset')
    parser.add_argument('--non_prom_path', type=str, default='non_promo_dataset/Ecoli_non_prom.fa', 
                      help='Path to non-promoter dataset')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.model == 'transformer':
        train_loader, test_loader, test_labels = create_data_loaders_transformers(
            batch_size=2,
            subset_size=args.subset_size
        )
        model = DNABERT_TransformerClassifier().to(device)
        if args.train:
            start_time = time.time()
            train_transformer(model, train_loader=train_loader, device=device, num_epochs=args.num_epochs)
            train_time = time.time() - start_time
            print(f"Training completed in {train_time:.2f} seconds")

        elif args.eval:
            start_time = time.time()
            evaluate_transformer(model, test_loader, test_labels, device)
            eval_time = time.time() - start_time
            print(f"Evaluation completed in {eval_time:.2f} seconds")
    else:
        train_loader, test_loader, y_test, seq_length = create_data_loaders(
            batch_size=2,
            prom_path=args.prom_path,
            non_prom_path=args.non_prom_path
        )
        model = PromoterCNN(seq_length).to(device)
        
        if args.train:
            start_time = time.time()
            train(model, train_loader, device, num_epochs=args.num_epochs)
            train_time = time.time() - start_time
            print(f"Training completed in {train_time:.2f} seconds")
        
        if args.eval:
            start_time = time.time()
            checkpoint = torch.load('best_model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            evaluate(model, test_loader, y_test, device)
            eval_time = time.time() - start_time
            print(f"Evaluation completed in {eval_time:.2f} seconds")

if __name__ == "__main__":
    main()
