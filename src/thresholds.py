"""Shared by agent.py and train_provisioning_model.py so both models pick
operating thresholds the same honest way: swept on a VALIDATION split
carved from TRAIN, never touching test."""
import numpy as np
from sklearn.metrics import precision_score, f1_score


def pick_thresholds(y_val, proba_val, hold_precision_target=0.90):
    best_hold = 0.9
    for t in np.arange(0.99, 0.3, -0.01):
        preds = (proba_val >= t).astype(int)
        if preds.sum() == 0:
            continue
        prec = precision_score(y_val, preds, zero_division=0)
        if prec >= hold_precision_target:
            best_hold = round(float(t), 2)
            break
    best_f1, best_t = -1, 0.5
    for t in np.arange(0.1, best_hold, 0.02):
        preds = (proba_val >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, round(float(t), 2)
    return best_hold, best_t
