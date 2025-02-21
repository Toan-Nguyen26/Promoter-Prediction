# **Promoter Sequence Classifier**

A deep learning framework for classifying bacterial promoter sequences using Convolutional Neural Networks (CNN). This project implements a **binary classifier** to distinguish between **promoter** and **non-promoter** DNA sequences, such as those in *E. coli*.

---

## **Overview**

This framework offers two approaches for **sequence-based classification**:

### **CNN Model**
- **One-hot encoding** for DNA sequence representation.
- **Convolutional layers** to extract important sequence motifs.

### **Transformer Model (DNA-BERT)**
- Uses pre-trained **DNA-BERT** for sequence embeddings
- Uses **Transformer Decoder** to learn DNA sequences patterns

---

## **Dataset**

The dataset used in this project is from [CNNPromoterData](https://github.com/solovictor/CNNPromoterData/tree/master), which was originally used in the [paper](https://arxiv.org/abs/1610.00121).

---

## **Installation & Dependencies**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-repo/promoter-sequence-classifier.git
cd promoter-sequence-classifier
```

### **2. Install Required Packages**
```bash
pip install -r requirements.txt
```

---
## **Command Line Arguments**

### **General Arguments**
```
--model          Choose model type: 'CNN' or 'transformer' (default: 'CNN')
--train          Flag to train the model
--eval           Flag to evaluate the model
--num_epochs     Number of training epochs (default: 10)
--subset_size    Number of samples to use for quick testing (optional only for Transformer model)
```

### **Data Arguments**
```
--prom_path      Path to promoter sequence dataset (default: 'promo_dataset/Ecoli_prom.fa')
--non_prom_path  Path to non-promoter sequence dataset (default: 'non_promo_dataset/Ecoli_non_prom.fa')
(These arguements are only for CNN model)
```

## **Training & Evaluation**

### **Training the Model**
To train the model, use:
```bash
# For CNN model
python main.py --train --model CNN\
    --num_epochs 10
    --prom_path promoter_dataset_path \
    --non_prom_path nonpromoter_dataset_path \

python main.py --train --model transformer\
    --num_epochs 10
```

The trained models will be saved as:
- CNN: **`best_model.pth`**
- Transformer: **`best_transformer_model.pth`**

---

### **Evaluating the Model**
To evaluate a trained model, run:
```bash
# For CNN model
python main.py --eval --model CNN

# For Transformer model
python main.py --eval --model transformer
```

## **Model Details**

### **CNN Architecture**
- Input: One-hot encoded DNA sequences
- Multiple convolutional layers
- Max pooling and dropout
- Dense layers for classification

### **Transformer Architecture**
- Based on DNA-BERT (117M parameters)
- Specialized DNA sequence tokenization
- Pre-trained on large DNA corpus
- Fine-tuned classification head

## **Performance Metrics**
Both models are evaluated using:
- Sensitivity (Sn)
- Specificity (Sp)
- Accuracy (Acc)
- Matthews Correlation Coefficient (MCC)