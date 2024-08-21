'''Contains predefined metrics sets, implemented as dictionaries with metric
names as keys, and the corresponding function to calculate them as values. Note 
that all of these functions assume an input tuple: `(y_true, y_pred)`.'''

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_absolute_error, mean_squared_error
)


classification_metrics = {
    'Accuracy': accuracy_score,
    'Precision (pcrna)': lambda y_t, y_p: precision_score(y_t,y_p,pos_label=1),
    'Recall (pcrna)': lambda y_t, y_p: recall_score(y_t,y_p,pos_label=1),
    'Precision (ncrna)': lambda y_t, y_p: precision_score(y_t,y_p,pos_label=0),
    'Recall (ncrna)': lambda y_t, y_p: recall_score(y_t,y_p,pos_label=0),
    'F1 (macro)': lambda y_t, y_p: f1_score(y_t, y_p, average='macro')
}
'''Default lncRNA classification metrics: accuracy, precision and recall (for 
pcRNA and lncRNA), and F1 (macro-averaged over both classes).'''


mtm_metrics = {
    'Accuracy': accuracy_score,
    'Precision (macro)': lambda y_t, y_p: precision_score(
        y_t, y_p, average='macro', zero_division=np.nan),
    'Recall (macro)': lambda y_t, y_p: recall_score(
        y_t, y_p, average='macro', zero_division=np.nan),
    'F1 (macro)': lambda y_t, y_p: f1_score(
        y_t, y_p, average='macro', zero_division=np.nan),
    'Counts': lambda y_t, y_p: (
        np.unique(y_t.numpy(), return_counts=True),
        np.unique(y_p.numpy(), return_counts=True)
    ) 
}
'''Default MTM evaluation metrics: accuracy, precision, recal, and F1 (macro-
averaged), as well as counts per token.'''


mmm_metrics = {
    'Accuracy': accuracy_score
}
'''Default MMM evaluation metrics: accuracy.'''


def regression_distribution(y_true, y_pred):
    '''Calculates the distribution of values in y_pred'''
    return (
        [np.quantile(y_true, p) for p in [0.01,0.05,0.25,0.5,0.75,0.95,0.99]],
        [np.quantile(y_pred, p) for p in [0.01,0.05,0.25,0.5,0.75,0.95,0.99]],
    )

regression_metrics = {
    'RMSE': lambda y_t, y_p: np.sqrt(mean_squared_error(y_t, y_p)),
    'MAE': lambda y_t, y_p: mean_absolute_error(y_t, y_p),
    'Distribution': lambda y_t, y_p: regression_distribution(y_t, y_p)
}
'''Default regression metrics ((root) mean absolute/squared error)'''


def frame_agreement(y_true, y_pred):
    '''Fraction of times in which `y_true` and `y_pred` agree in terms of 
    reading frame'''
    return ((np.mod(np.round(y_true), 3) == np.mod(np.round(y_pred), 3)).sum() / 
             np.size(y_true))

def frame_consistency(y_true, y_pred):
    '''Fraction of times in which `y_pred` agrees with itself in terms of 
    reading frame'''
    return ((np.mod(np.round(y_pred[:,0]), 3) == 
             np.mod(np.round(y_pred[:,1]), 3)).sum() / 
            len(y_pred))

orf_prediction_metrics = {
    'RMSE (ORF (start))': 
        lambda y_t, y_p: np.sqrt(mean_squared_error(y_t[:,0], y_p[:,0])),
    'RMSE (ORF (end))': 
        lambda y_t, y_p: np.sqrt(mean_squared_error(y_t[:,1], y_p[:,1])),
    'MAE (ORF (start))': 
        lambda y_t, y_p: mean_absolute_error(y_t[:,0], y_p[:,0]),
    'MAE (ORF (end))': 
        lambda y_t, y_p: mean_absolute_error(y_t[:,1], y_p[:,1]),
    'Frame agreement': frame_agreement,
    'Frame consistency': frame_consistency,
    'Distribution (ORF (start))': 
        lambda y_t, y_p: regression_distribution(y_t[:,0], y_p[:,0]),
    'Distribution (ORF (end))': 
        lambda y_t, y_p: regression_distribution(y_t[:,1], y_p[:,1]),
}
'''Metrics designed for ORF prediction. Includes the RMSE/MAE separately for 
start and end coordinates of the ORF, as well as frame agreement and 
consistency.'''