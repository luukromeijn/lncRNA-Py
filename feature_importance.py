import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from rhythmnblues.data import Data
from rhythmnblues.features.selection import NoSelection, TTest, Regression, RandomForest, Permutation, RecFeatElim


def feature_importance(
        trainsets, testsets, k, tables_folder, methods=[NoSelection, TTest, 
        Regression, RandomForest, Permutation, RecFeatElim], excluded_features=
        ['id', 'label', 'sequence', 'ORF protein', 'SSE'], test=False
    ):
    '''Runs a feature importance analysis.''' # TODO expand documentation

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


def plot_feature_importance(importances, k, method=None, trainset=None, 
                            filepath=None):
    '''TODO'''

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
    importances = importances.sort_values(by='avg', ascending=ascending).head(k)

    # Plot
    fig, ax = plt.subplots()
    ax.bar(np.arange(k), importances['avg'])
    ax.errorbar(np.arange(k), importances['avg'], yerr=importances['std'], 
                fmt='none', ecolor='black')
    ax.set_xticks(np.arange(k), importances.index, rotation=90)
    ax.set_ylabel(metric)
    suffix = method if method else ''
    suffix = suffix + ', ' if method and trainset else suffix
    suffix = suffix + trainset if trainset else suffix
    ax.set_title(f'Feature importance ({suffix})')
    fig.tight_layout()

    if filepath is not None:
        fig.savefig(filepath)
    return fig


def plot_feature_selection_results(results, groupby, filepath=None):
    '''TODO'''
    
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