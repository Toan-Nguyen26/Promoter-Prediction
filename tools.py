import numpy as np
from sklearn.metrics import roc_curve

def find_optimal_threshold(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    J_scores = tpr - fpr 
    optimal_idx = np.argmax(J_scores)  
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal Threshold (Youden’s J): {optimal_threshold:.4f}")
    return optimal_threshold
