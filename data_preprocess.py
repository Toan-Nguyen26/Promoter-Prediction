import torch
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os


# One-hot encoding function
def one_hot_encode(seq, seq_length):
    mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    encoded = np.zeros((seq_length, 4), dtype=np.float32)
    
    for i, base in enumerate(seq):
        if base in mapping:
            encoded[i] = mapping[base]
    return encoded

# Load promoter and non-promoter sequences
def load_fasta(file, seq_length):
    sequences = []
    for record in SeqIO.parse(file, "fasta"):
        seq = str(record.seq).upper()
        if len(seq) >= seq_length:
            sequences.append(one_hot_encode(seq[:seq_length], seq_length))
    return np.array(sequences)

def create_train_test_data(prom_path="dataset/Ecoli_prom.fa", non_prom_path="dataset/Ecoli_non_prom.fa"):
    # Dynamically get sequence length from first sequence
    with open(prom_path) as f:
        first_seq = next(SeqIO.parse(f, "fasta"))
        seq_length = len(str(first_seq.seq))

    # Load data with determined sequence length
    X_promoter = load_fasta(prom_path, seq_length)
    X_non_promoter = load_fasta(non_prom_path, seq_length)

    print(f"\nDataset sizes:")
    print(f"Promoter sequences: {len(X_promoter)}")
    print(f"Non-promoter sequences: {len(X_non_promoter)}")

    # Labels (1 = promoter, 0 = non-promoter)
    y_promoter = np.ones(len(X_promoter))
    y_non_promoter = np.zeros(len(X_non_promoter))

    # Combine data
    X = np.vstack((X_promoter, X_non_promoter))
    y = np.hstack((y_promoter, y_non_promoter))

    # Shuffle and split
    X, y = shuffle(X, y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Convert to PyTorch tensors
    X_train, X_test, y_train, y_test = (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    return X_train, X_test, y_train, y_test, seq_length

# Function to load and split sequences
def create_train_test_data_transformer():
    PROMO_DIR = "promo_dataset/"
    NON_PROMO_DIR = "non_promo_dataset/"

    def load_sequences_from_folder(folder):
        sequences = []
        for file in os.listdir(folder):
            if file.endswith(".fa") or file.endswith(".fasta"):
                filepath = os.path.join(folder, file)
                for record in SeqIO.parse(filepath, "fasta"):
                    sequences.append(str(record.seq).upper())  # Convert to uppercase
        return sequences

    promo_sequences = load_sequences_from_folder(PROMO_DIR)
    non_promo_sequences = load_sequences_from_folder(NON_PROMO_DIR)

    promo_labels = [1] * len(promo_sequences)
    non_promo_labels = [0] * len(non_promo_sequences)

    all_sequences = promo_sequences + non_promo_sequences
    all_labels = promo_labels + non_promo_labels

    print(f"Total sequences: {len(all_sequences)} (Promoters: {len(promo_sequences)}, Non-promoters: {len(non_promo_sequences)})")
    
    # Split dataset into train and test
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(
        all_sequences, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )
    
    return train_sequences, test_sequences, train_labels, test_labels

def main():
    print("Testing data preprocessing functions...")
    
    # Test transformer data creation
    print("\nTesting create_train_test_data_transformer:")
    train_sequences, test_sequences, train_labels, test_labels = create_train_test_data_transformer()
    
    print(f"\nDataset Split Results:")
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")
    print(f"Train labels: {len(train_labels)}")
    print(f"Test labels: {len(test_labels)}")
    
    # Print sample data
    print(f"\nSample sequence from training set:")
    print(train_sequences[0][:100] + "...") # Print first 100 chars
    print(f"Sample label: {train_labels[0]}")
    
    # # Test regular data creation
    # print("\nTesting create_train_test_data:")
    # X_train, X_test, y_train, y_test, seq_length = create_train_test_data()
    
    # print(f"\nSequence length: {seq_length}")
    # print(f"X_train shape: {X_train.shape}")
    # print(f"X_test shape: {X_test.shape}")

if __name__ == "__main__":
    main()