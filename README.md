# **Promoter Sequence Classifier**

A deep learning model for classifying bacterial promoter sequences using Convolutional Neural Networks (CNN). This project implements a **binary classifier** to distinguish between **promoter** and **non-promoter** DNA sequences, such as those in *E. coli*.

---

## **Overview**

This model is designed specifically for **sequence-based classification** using CNNs with the following features:

- **One-hot encoding** for DNA sequence representation.
- **Convolutional layers** to extract important sequence motifs.
- **Batch normalization & dropout** for improved generalization.
- **Focal Loss** to handle class imbalance effectively.
- **Efficient training and evaluation scripts** for easy use.

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

## **Training & Evaluation**

### **Training the Model**
To train the model, use:
```bash
python main.py --train \
    --prom_path promoter_dataset_path \
    --non_prom_path nonpromoter_dataset_path \
```

The trained model will be saved as **`saved_model.pth`** by default.

---

### **Evaluating the Model**
To evaluate a trained model, run:
```bash
python main.py --eval \
    --model_path saved_model.pth \
    --test_prom promoter_dataset_path \
    --test_non_prom nonpromoter_dataset_path
```

## **Example Workflow**
### **Step 1: Train the Model**
```bash
python main.py --train \
    --prom_path dataset/human_non_tata.fa \
    --non_prom_path dataset/human_nonprom_big.fa \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001
```

### **Step 2: Save the Model**
After training, the model is saved as `saved_model.pth`.

### **Step 3: Evaluate the Model**
```bash
python main.py --eval \
    --model_path saved_model.pth \
    --test_prom dataset/human_non_tata.fa \
    --test_non_prom dataset/human_nonprom_big.fa
```
