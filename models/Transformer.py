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
    def __init__(self):
        super().__init__()
        self.dna_bert = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M")
        self.classifier = nn.Linear(768, 1)  # DNABERT-2 has 768 hidden dimensions
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Unpack the input dictionary
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        
        # Get BERT outputs
        outputs = self.dna_bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation (first token)
        cls_output = outputs[0][:, 0, :]  # Shape: (batch_size, hidden_size)
        
        # Apply dropout and classification layer
        x = self.dropout(cls_output)
        x = self.classifier(x)
        
        # Apply sigmoid for binary classification
        return torch.sigmoid(x)

