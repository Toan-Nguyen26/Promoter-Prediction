from torch.utils.data import DataLoader
from promoter_class import PromoterDataset, PromoterTransformerDataset
from data_preprocess import create_train_test_data, create_train_test_data_transformer
from transformers import AutoTokenizer
import torch
import numpy as np

def create_data_loaders(batch_size=32, prom_path="dataset/Ecoli_prom.fa", non_prom_path="dataset/Ecoli_non_prom.fa"):
    # Create train-test split
    X_train, X_test, y_train, y_test, seq_length = create_train_test_data(prom_path, non_prom_path)
    
    # Create datasets
    train_dataset = PromoterDataset(X_train, y_train)
    test_dataset = PromoterDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, y_test, seq_length

# Load DNA-BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

def collate_fn(batch):
    sequences, labels = zip(*batch)  # Unpack batch into sequences and labels
    
    # Tokenize and pad sequences to the longest in the batch
    tokens = tokenizer(
        list(sequences), 
        padding='longest',  # Pad to longest in the batch
        truncation=False,  
        return_tensors="pt"
    )
    
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return {
        'input_ids': tokens['input_ids'],
        'attention_mask': tokens['attention_mask'],
        'labels': labels
    }

def create_data_loaders_transformers(batch_size=32, subset_size=None):
    def collate_fn(batch):
        sequences, labels = zip(*batch)
        tokens = tokenizer(
            list(sequences), 
            padding='longest',
            truncation=False,  
            return_tensors="pt"
        )
        
        labels = torch.tensor(labels, dtype=torch.float32)
        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
            'labels': labels
        }

    # Get split data
    train_sequences, test_sequences, train_labels, test_labels = create_train_test_data_transformer()
    
    # Convert test_labels to numpy array if it isn't already
    test_labels_np = np.array(test_labels)
    
    # Apply subset if specified
    if subset_size is not None:
        print(f"\nUsing subset of data:")
        print(f"Original train size: {len(train_sequences)}")
        train_sequences = train_sequences[:subset_size]
        train_labels = train_labels[:subset_size]
        test_sequences = test_sequences[:subset_size//5]
        test_labels = test_labels[:subset_size//5]
        test_labels_np = test_labels_np[:subset_size//5]
        print(f"Subset train size: {len(train_sequences)}")
        print(f"Subset test size: {len(test_sequences)}")

    train_dataset = PromoterTransformerDataset(train_sequences, train_labels)
    test_dataset = PromoterTransformerDataset(test_sequences, test_labels)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Return test_labels_np for evaluation
    return train_loader, test_loader, test_labels_np

# def test_transformer_data_loading(batch_size=2):
#     print("\nTesting transformer data loading...")
#     train_loader, test_loader = create_data_loaders_transformers(batch_size=batch_size)
    
#     # Test a few batches from the training loader
#     print("\nChecking training data:")
#     for batch_idx, batch in enumerate(train_loader):
#         if batch_idx >= 3:  # Only check first 3 batches
#             break
            
#         print(f"\nBatch {batch_idx + 1}:")
#         print(f"Batch size: {len(batch['labels'])}")
        
#         # Print details for each item in the batch
#         for i in range(len(batch['labels'])):
#             print(f"\nItem {i + 1}:")
#             print(f"Label: {batch['labels'][i]}")
#             print(f"Input shape: {batch['input_ids'][i].shape}")
#             # Decode the tokens back to text to verify
#             decoded = tokenizer.decode(batch['input_ids'][i])
#             print(f"Decoded sequence (first 50 chars): {decoded[:50]}...")

# def main():
#     print("Testing data preprocessing functions...")
    
#     # Test transformer data loading
#     test_transformer_data_loading()

# if __name__ == "__main__":
#     main()