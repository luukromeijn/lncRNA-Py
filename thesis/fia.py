'''Feature Importance Analysis'''

import pandas as pd 
from rhythmnblues.selection.importance_analysis import (
    plot_feature_importance, plot_feature_selection_results, 
    sorted_feature_importance
)
import matplotlib.pyplot as plt
from rhythmnblues.features import *
import numpy as np

imp_dir = 'results/importances'
results_dir = 'results/importances'

trainsets = [
    'GENCODE', 
    'RefSeq', 
    'NONCODE', 
    'CPAT (train)'
]
method_names = ['t-test', 'Regression', 'mDS', 'Random forest', 'RFE']

name_map = {
    'gencode_train': 'GENCODE', 
    'refseq_train': 'RefSeq',
    'noncode-refseq_train': 'NONCODE',
    'cpat_train_train': 'CPAT (train)',
}

methods = ['noselection', 'ttest', 'regression', 'mds', 'randomforest', 'rfe']
importances, results = [], []
for method in methods:
    if method != 'noselection':
        importances.append(pd.read_csv(f'{imp_dir}/importances.{method}.csv', 
                                       index_col=0))
    results.append(pd.read_csv(f'{imp_dir}/results.{method}.csv', index_col=0))

importances = pd.concat(importances)
results = pd.concat(results)

results = results.replace(name_map)
importances = importances.replace(name_map)

plot_feature_selection_results(results, ['Selection method', 'Trainset'],
                               filepath=f'{results_dir}/performance.png',
                               figsize=(12.8, 4.8))

for method in method_names:

    sorted, metric = sorted_feature_importance(importances, method=method)
    sorted.to_csv(f'{results_dir}/{method}.sorted.{metric}.csv')

    plot_feature_importance(importances, method=method, 
                            filepath=f'{results_dir}/{method}.all.png')
    plot_feature_importance(importances, k=20, method=method, 
                            filepath=f'{results_dir}/{method}.top20.png')
    plt.close('all')

for trainset in trainsets:

    sorted, metric = sorted_feature_importance(importances, trainset=trainset)
    sorted.to_csv(f'{results_dir}/{trainset}.sorted.{metric}.csv')

    plot_feature_importance(importances, trainset=trainset, 
                            filepath=f'{results_dir}/{trainset}.all.png')
    plot_feature_importance(importances, k=20, trainset=trainset, 
                            filepath=f'{results_dir}/{trainset}.top20.png')
    plt.close('all')

sorted, metric = sorted_feature_importance(importances)

sorted.to_csv(f'{results_dir}/all.sorted.{metric}.csv')
plot_feature_importance(importances, filepath=f'{results_dir}/all.all.png')
plot_feature_importance(importances, k=20, 
                        filepath=f'{results_dir}/all.top20.png')

# Analyze frequency counts{
groups = {
    '1-mers': KmerFreqs(k=1, apply_to='sequence', stride=1).name,
    '1-mers (ORF)': KmerFreqs(k=1, apply_to='ORF', stride=1).name,
    '2-mers': KmerFreqs(k=2, apply_to='sequence', stride=1).name,
    '2-mers (1-gapped)': KmerFreqs(k=2, apply_to='sequence', stride=1, 
                                   gap_length=1, gap_pos=1).name,
    '2-mers (2-gapped)': KmerFreqs(k=2, apply_to='sequence', stride=1, 
                                   gap_length=2, gap_pos=1).name,
    '2-mers (ORF)': KmerFreqs(k=2, apply_to='ORF', stride=1).name,
    '3-mers': KmerFreqs(k=3, apply_to='sequence', stride=1).name,
    '3-mers (PLEK)': KmerFreqs(k=3, apply_to='sequence', stride=1, 
                               PLEK=True).name,
    '3-mers (ORF)': KmerFreqs(k=3, apply_to='ORF', stride=3).name,
    '6-mers': KmerFreqs(k=6, apply_to='sequence', stride=1).name,
    '6-mers (ORF)': KmerFreqs(k=6, apply_to='ORF', stride=3).name,
}

for group in groups:

    data = sorted['avg'].T[groups[group]]
    mean = round(data.mean())
    median = round(data.median())
    dmax = round(data.max())
    dmin = round(data.min())
    best = np.array(groups[group])[data.argsort().values.tolist()[:3]]
    print(f'{group} & {mean} & {median} & {dmin} & {dmax} & {best[0]} & ' +
          f'{best[1]} & {best[2]}')