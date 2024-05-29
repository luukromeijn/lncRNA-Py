'''Contains `feature_importance_analysis` function, as well as several 
accompanying plotting functions that are tailored to this output format.'''
# NOTE currently excluded in unittests...

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from rhythmnblues.data import Data
from rhythmnblues.selection import (
    NoSelection, TTestSelection, RegressionSelection, ForestSelection, 
    RFESelection, MDSSelection
)


def feature_importance_analysis(
        trainsets, testsets, k, tables_folder, methods=[NoSelection, 
        TTestSelection, RegressionSelection, ForestSelection, RFESelection, 
        MDSSelection], excluded_features=['id', 'label', 'sequence', 
        'ORF protein', 'SSE'], test=False
    ):
    '''Runs a feature importance analysis, according to the following steps:
    - For every trainset in `trainsets`:
        - For every method in `methods`:
            - Assess feature importance.
            - Select `k` most important features.
            - Fit a random forest to the trainset using the selected features.
            - Evaluate the F1-score of the random forest on all `testsets`.
    
    Arguments
    ---------
    `trainsets`: `list[str]`
        Name of trainsets to be used for the importance analysis. For every name
        in the list, a hdf (.h5) file with feature data is assumed to be present
        in the directory specified by the `tables_folder` argument.
    `testsets`: `list[str]`
        Name of testsets to report performance on. Like with `trainsets`, every
        testset is assumed as hdf (.h5) file with this name in `tables_folder`.
    `k`: `int`
        Number of features to select.
    `methods`: `list[type]`
        Type of feature selection methods to apply. Should be classes from 
        `rhythmnblues.selection`.
    `excluded_features`: `list[str]`
        List of features to exclude from the importance analysis. 
    `test`: `bool`
        If True, performs analysis on 1000 random training samples.

    Returns
    -------
    `importances`: `pd.DataFrame`
        Reports the importance of all features for every combination of trainset
        and selection method.
    `results`: `pd.DataFrame`
        Reports the performance (macro-averaged F1-score) of all features for
        every combination of trainset, selection method, and testset.'''

    # Initialization
    print("Running feature importance analysis...")
    features = None
    results, importances = [], []

    for j, trainset_name in enumerate(trainsets): # Loop through trainsets

        # Load trainset
        print("Trainset:", trainset_name)
        trainset = Data(hdf_filepath=f'{tables_folder}/{trainset_name}.h5')
        if test:
            trainset = trainset.sample(N=1000) 
        if features is None: # First trainset determines features to consider
            features = trainset.all_features(except_columns=excluded_features)

        for i, method in enumerate(methods): # Loop through selection methods

            # Calculate importance & select features
            method = method(k) 
            print("Method:", method.name)
            sel_features, importance = method.select_features(trainset,features)
            importances.append([method.name, method.metric_name, trainset_name]
                               + list(importance))

            # Fit baseline algorithm
            baseline = make_pipeline(
                SimpleImputer(missing_values=np.nan), 
                StandardScaler(),
                RandomForestClassifier(class_weight='balanced')
            )
            baseline.fit(trainset.df[sel_features], trainset.df['label'])

            # Evaluate on test sets
            print("Evaluating performance...")
            for m, testset_name in enumerate(testsets):
                testset=Data(hdf_filepath=f'{tables_folder}/{testset_name}.h5')
                pred = baseline.predict(testset.df[sel_features])
                f1 = f1_score(testset.df['label'], pred, average='macro')
                results.append([method.name, trainset_name, testset_name, f1, 
                                sel_features])
                
    results = pd.DataFrame(results, columns=['Selection method', 'Trainset', 
                                             'Testset', 'F1', 'Features'])
    importances = pd.DataFrame(importances, columns=(['Selection method', 
                             'Importance metric', 'Trainset'] + list(features)))

    return importances, results


def plot_feature_importance(importances, k=None, method=None, trainset=None, 
                            filepath=None, figsize=None):
    '''Creates a feature importance plot, given the `importances` dataframe from
    `feature_importance_analysis`.
    
    Arguments
    ---------
    `importances`: `pd.DataFrame`
        Output from `feature_importance_analysis`.
    `k`: `int`
        Top number of features to plot, if specified (default is None).
    `method`: `str`
        Method to consider, if specified (default is None). If not specified 
        and data contains multiple metrics, will convert importances to rank.
    `trainset`: `str`
        Trainset name to consider, if specified (default is None). If not 
        specified, averages over all trainsets.
    `filepath`: `str`
        If specified, saves figure to this filepath (default is None).
    `figsize`: `tuple[int]`
        Matplotlib figure size (default is None).'''

    importances, metric = sorted_feature_importance(importances,method,trainset)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    width = 1
    if k is not None:
        importances = importances.head(k)
        width = 0.8
    ax.bar(np.arange(len(importances)), importances['avg'], width=width)
    if k is not None:
        ax.errorbar(np.arange(len(importances)), importances['avg'], 
                    yerr=importances['std'], fmt='none', ecolor='black')
        ax.set_xticks(np.arange(k), importances.index, rotation=90)
    else:
        ax.set_xlim(left=0)
    ax.set_ylabel(metric)
    suffix = method if method else ''
    suffix = suffix + ', ' if method and trainset else suffix
    suffix = suffix + trainset if trainset else suffix
    suffix = f' ({suffix})' if len(suffix) > 0 else suffix
    ax.set_title('Feature importance'+ suffix)
    fig.tight_layout()

    if filepath is not None:
        fig.savefig(filepath)
    return fig


def plot_feature_selection_results(results, groupby, filepath=None, 
                                   figsize=None):
    '''Plots the performance of different feature selection methods, based on 
    output from `feature_importance_analysis`.
    
    Arguments
    ---------
    `results`: `pd.DataFrame`
        Results output from `feature_importance_analysis`.
    `groupby`: `str` | `list[str]` | `tuple[str]`
        How to group the data. If of type `str`, will average over this column.
        If of type `list` or `tuple`, will apply a nested grouping where the 
        second element refers to the inner group. 
    `filepath`: `str`
        If specified, saves figure to this filepath (default is None).
    `figsize`: `tuple[int]`
        Matplotlib figure size (default is None).'''
    
    # Set inner and outer groups
    groupby = [groupby] if type(groupby) == str else groupby
    outer_groups = results[groupby[0]].unique()
    if len(groupby) == 1:
        inner_groups = [None] # Dummy in case of no inner groups
    elif len(groupby) == 2: 
        inner_groups = results[groupby[1]].unique()
    else: 
        raise ValueError("Minimum of 1 and maximum of 2 groups.")
    
    fig, ax = plt.subplots()

    multiplier = 0
    width = 1/(len(inner_groups) + 1)
    for group in inner_groups:
        
        if group is None:
            grouped = results # No inner grouping
        else:
            grouped = results[results[groupby[1]]==group] # Inner grouping
        grouped = grouped.groupby(groupby[0], sort=False) # Outer grouping
        
        # Calculating avg and std
        avg = grouped['F1'].mean()
        std = grouped['F1'].std()

        # Actual plotting
        x = np.arange(len(outer_groups))+(width*multiplier)
        ax.bar(x, avg, width=width, label=group)
        ax.errorbar(x, avg, yerr=std, fmt='none', ecolor='black')

        multiplier += 1

    ax.set_xticks(np.arange(len(outer_groups))+(len(inner_groups)-1)*0.5*width, 
                  outer_groups)
    ax.set_xlabel(groupby[0])
    ax.set_ylabel('F1')
    if len(groupby) == 2:
        fig.legend()
    fig.tight_layout()

    if filepath is not None:
        fig.savefig(filepath)
    return fig


def sorted_feature_importance(importances, method=None, trainset=None):
    '''Sorts feature `importances`, averaging over `method` or `trainset` if 
    specified.'''

    # Filtering for specified selection method and trainset
    if method:
        importances = importances[importances['Selection method'] == method]
    if trainset:
        importances = importances[importances['Trainset'] == trainset]
    if len(importances) < 1:
        raise RuntimeError()

    # Set to rank if multiple importance metrics are compared
    if len(importances['Importance metric'].unique()) == 1:
        metric = importances['Importance metric'].unique()[0]
    else:
        metric = 'Rank'
    
    # Calculate average importance and sort based on that
    importances = importances.drop(['Selection method', 'Importance metric', 
                                    'Trainset'], axis=1).T
    importances = importances.abs()
    importances = importances.fillna(0)
    if metric == 'Rank':
        importances = importances.rank(axis=0, ascending=False)
    avg, std = importances.abs().mean(axis=1), importances.abs().std(axis=1)
    importances['avg'] = avg
    importances['std'] = std
    ascending = True if metric == 'Rank' else False
    importances = importances.sort_values(by='avg', ascending=ascending)

    return importances, metric # NOTE metric is also returned!