from torch.utils.data import DataLoader
from promoter_class import PromoterDataset
from data_preprocess import create_train_test_data

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