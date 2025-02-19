import torch
import torch.nn as nn
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

# tokenizer = BertTokenizer.from_pretrained("jzhihan1996/DNABERT-2-117M", do_lower_case=False)
# dna_bert = BertModel.from_pretrained("zhihan1996/DNABERT-2-117M")

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
dna_bert = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dna_bert.to(device)

# Function to extract DNA-BERT embeddings
def get_dna_bert_embedding(sequence):
    tokens = tokenizer(sequence, padding="max_length", max_length=81, truncation=True, return_tensors="pt")
    tokens = {key: val.to(device) for key, val in tokens.items()}

    with torch.no_grad():
        embeddings = dna_bert(**tokens).last_hidden_state  # Extract embeddings
    return embeddings  # Shape: (batch, seq_len, 768)


class DNABERT_TransformerClassifier(nn.Module):
    def __init__(self, num_encoder_layers=2, embed_dim=768, num_heads=8, ff_hidden=512):
        super(DNABERT_TransformerClassifier, self).__init__()
        
        # Load DNA-BERT
        self.dna_bert = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M")
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, ff_hidden, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.fc = nn.Linear(embed_dim, 1) 

    def forward(self, x):
        with torch.no_grad():
            x = self.dna_bert(**x).last_hidden_state  
        
        x = self.transformer_encoder(x)  #
        x = x.mean(dim=1) 
        return torch.sigmoid(self.fc(x))  

