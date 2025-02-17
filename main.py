import torch
from torch import optim
import argparse
from create_data import create_data_loaders
from models.CNN import PromoterCNN, FocalLoss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import time

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
                'model_state_dict': model.state_dict(),
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

def evaluate(model, test_loader, y_test, device):
    model.eval()
    y_pred_probs = []
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_pred_probs.extend(model(X_batch).cpu().numpy())
    
    y_pred_probs = np.array(y_pred_probs).ravel()
    
    fpr, tpr, _ = roc_curve(y_test.numpy(), y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--prom_path', type=str, default='dataset/Ecoli_prom.fa', help='Path to promoter dataset')
    parser.add_argument('--non_prom_path', type=str, default='dataset/Ecoli_non_prom.fa', help='Path to non-promoter dataset')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, test_loader, y_test, seq_length = create_data_loaders(
        batch_size=2,
        prom_path=args.prom_path,
        non_prom_path=args.non_prom_path
    )
    model = PromoterCNN(seq_length).to(device)
    
    if args.train:
        start_time = time.time()
        train(model, train_loader, device)
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
