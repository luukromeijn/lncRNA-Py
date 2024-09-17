'''Functions for evaluating RNA models.'''

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score)
import pandas as pd


# TODO unittest?
def lncRNA_classification_report(y_true, y_pred, method_name, testset_name,
                                 filepath=None):
    '''Calculates lncRNA classification accuracy, precision, recall, and F1.
    Returns `pd.DataFrame`. If provided, saves output to .csv file to specified
    `filepath`.'''

    metrics = {'Precision': precision_score, 
               'Recall': recall_score, 
               'F1': f1_score}
    
    scores = {'Method': method_name, 'Dataset': testset_name, 
              'Accuracy': accuracy_score(y_true, y_pred)}
    for metric_name in metrics:
        for calc_for in ['pcRNA', 'ncRNA', 'weighted', 'macro']:
            karg = {
                {'pcRNA':'pos_label', 'ncRNA':'pos_label', 'weighted':'average',
                 'macro':'average'}[calc_for]: calc_for
            }
            metric_func = metrics[metric_name]
            name = f'{metric_name} ({calc_for})'
            scores[name] = [metric_func(y_true, y_pred, **karg)]

    scores = pd.DataFrame(scores)
    if filepath is not None:
        scores.to_csv(filepath)
    return scores