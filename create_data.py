from torch.utils.data import DataLoader
from promoter_class import PromoterDataset, PromoterTransformerDataset
from data_preprocess import create_train_test_data, create_train_test_data_transformer
from transformers import AutoTokenizer
import torch

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

def create_data_loaders_transformers(batch_size=32):
    def tokenize_sequences(sequences):
        return tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

    train_sequences, test_sequences, train_labels, test_labels = create_train_test_data_transformer()
    # Tokenize train and test sequences
    train_tokens = tokenize_sequences(train_sequences)
    test_tokens = tokenize_sequences(test_sequences)

    # Convert labels to tensors
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)
    train_dataset = PromoterTransformerDataset(train_tokens, train_labels)
    test_dataset = PromoterTransformerDataset(test_tokens, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader