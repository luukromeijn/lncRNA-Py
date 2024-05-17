'''Classes for selecting features based on an importance asssesment.'''
# NOTE/TODO This submodule is currently not part of any unittests.

import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class FeatureSelection:
    '''Base class for feature selection / importance analysis.
    
    Attributes
    ----------
    `name`: `str`
        Name of the applied method.
    `metric_name`: `str`
        Name of the metric that describes feature importance.
    `k`: `int`
        Number of features that will be selected.'''
    
    def __init__(self, name, metric_name, k):
        self.name = name
        self.metric_name = metric_name
        self.k = k

    def select_features(self, data, feature_names):
        '''Selects features by assessing their importance for a given `Data` 
        object. Returns a tuple in which the first element corresponds a list of 
        selected feature names, and the second element is the feature
        importance array.'''
        raise NotImplementedError
    
    def k_most_important_indices(self, feature_importance):
        feature_importance = np.abs(feature_importance)
        feature_importance = np.nan_to_num(feature_importance, nan=0)
        return np.argsort(np.abs(feature_importance))[::-1][:self.k]


class NoSelection(FeatureSelection):
    '''Dummy class that does not select or analyze features at all.'''

    def __init__(self, k):
        super().__init__('No selection', 'Undefined', k)

    def select_features(self, data, feature_names):
        return feature_names, np.zeros(len(feature_names))


class TTest(FeatureSelection):
    '''Feature selection based on an association test statistic (t-test).'''

    def __init__(self, k, alpha=0.05):
        super().__init__('t-test', 'Test statistic', k)
        self.alpha = alpha

    def select_features(self, data, feature_names):
        stats = data.test_features(feature_names)
        alpha = self.alpha / len(feature_names)
        print(f"Found {len(stats[stats['P value'] < alpha])} " + 
              "statistically significant associations.")
        feature_importance = stats['test statistic'].values
        idx = self.k_most_important_indices(feature_importance)
        return np.array(feature_names)[idx], feature_importance
    

class Regression(FeatureSelection):
    '''Feature selection based on the size of regression coefficients.'''

    def __init__(self, k):
        super().__init__('Regression', 'Coefficient size', k)
        self.model = make_pipeline(
            SimpleImputer(missing_values=np.nan), 
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight='balanced')
        )
    
    def select_features(self, data, feature_names):
        self.model.fit(data.df[feature_names], data.df['label'])
        feature_importance = self.model[2].coef_[0]
        idx = self.k_most_important_indices(feature_importance)
        return np.array(feature_names)[idx], feature_importance
    

class RandomForest(FeatureSelection):
    '''Feature selection based on the feature importance of a random forest.'''

    def __init__(self, k):
        super().__init__('Random forest', 'Gini importance', k)
        self.model = make_pipeline(
            SimpleImputer(missing_values=np.nan), 
            StandardScaler(),
            RandomForestClassifier(max_features=k, class_weight='balanced')
        )
    
    def select_features(self, data, feature_names):
        self.model.fit(data.df[feature_names], data.df['label'])
        feature_importance = self.model[2].feature_importances_
        idx = self.k_most_important_indices(feature_importance)
        return np.array(feature_names)[idx], feature_importance
    

class Permutation(FeatureSelection):
    '''Calculates feature importance by performing permutations on them.'''

    def __init__(self, k):
        super().__init__('Permutation', 'Permutation importance', k)
        self.model = make_pipeline(
            SimpleImputer(missing_values=np.nan), 
            StandardScaler(),
            RandomForestClassifier(class_weight='balanced')
        )

    def select_features(self, data, feature_names):
        X, y = data.df[feature_names], data.df['label']
        self.model.fit(X, y)
        feature_importance = permutation_importance(self.model, X, y)
        feature_importance = feature_importance.importances_mean
        idx = self.k_most_important_indices(feature_importance)
        return np.array(feature_names)[idx], feature_importance
    

class RecFeatElim(FeatureSelection):
    '''Recursive Feature Elimination. Uses ranks as importance measure.'''

    def __init__(self, k):
        super().__init__('RFE', 'Rank', k)
        self.model = make_pipeline(
            SimpleImputer(missing_values=np.nan), 
            StandardScaler(),
            RFE(RandomForestClassifier(class_weight='balanced'), 
                n_features_to_select=k)
        )
    
    def select_features(self, data, feature_names):
        X, y = data.df[feature_names], data.df['label']
        self.model.fit(X, y)
        feature_importance = len(feature_names) - self.model[2].ranking_
        idx = self.k_most_important_indices(feature_importance)
        return np.array(feature_names)[idx], feature_importance