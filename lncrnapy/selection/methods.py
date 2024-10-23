'''Classes for selecting features based on an importance asssesment.'''

from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class FeatureSelectionBase:
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
        if self.k > len(feature_importance):
            raise RuntimeError(f"Number of to-be-selected features ({self.k}) "+ 
                               f"exceeds number of available features " + 
                               f"({len(feature_importance)}).")
        feature_importance = np.abs(feature_importance)
        feature_importance = np.nan_to_num(feature_importance, nan=0)
        return np.argsort(np.abs(feature_importance))[::-1][:self.k]


class NoSelection(FeatureSelectionBase):
    '''Dummy class that does not select or analyze features at all.'''

    def __init__(self, k):
        super().__init__('No selection', 'Undefined', k)

    def select_features(self, data, feature_names):
        return feature_names, np.zeros(len(feature_names))


class TTestSelection(FeatureSelectionBase):
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
    

class RegressionSelection(FeatureSelectionBase):
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
    

class ForestSelection(FeatureSelectionBase):
    '''Feature selection based on the feature importance of a random forest.'''

    def __init__(self, k):
        super().__init__('Random forest', 'Gini importance', k)
        self.model = make_pipeline(
            SimpleImputer(missing_values=np.nan), 
            StandardScaler(),
            RandomForestClassifier(max_features='log2', class_weight='balanced')
        )
    
    def select_features(self, data, feature_names):
        self.model.fit(data.df[feature_names], data.df['label'])
        feature_importance = self.model[2].feature_importances_
        idx = self.k_most_important_indices(feature_importance)
        return np.array(feature_names)[idx], feature_importance
    

class PermutationSelection(FeatureSelectionBase):
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
    

class RFESelection(FeatureSelectionBase): # TODO: not really a nice name I guess
    '''Recursive Feature Elimination. Uses ranks as importance measure.'''

    def __init__(self, k, step=0.01):
        super().__init__('RFE', 'Rank', k)
        self.model = make_pipeline(
            SimpleImputer(missing_values=np.nan), 
            StandardScaler(),
            RFE(RandomForestClassifier(max_features='log2', 
                                       class_weight='balanced'), 
                step=step, n_features_to_select=k)
        )
    
    def select_features(self, data, feature_names):
        X, y = data.df[feature_names], data.df['label']
        self.model.fit(X, y)
        feature_importance = len(feature_names) - self.model[2].ranking_
        idx = self.k_most_important_indices(feature_importance)
        return np.array(feature_names)[idx], feature_importance
    
class MDSSelection(FeatureSelectionBase):
    '''Method based on Minimum Distribution Similarity (mDS) as proposed by 
    DeepCPP. Uses relative entropy (Kullback-Leibler divergence) to calculate 
    the difference between feature distributions of pcRNA and ncRNA, selects 
    those that are most different from each other.
    
    Arguments
    ---------
    `k`: `int`
        Number of features that will be selected.
    `lower`: `float`
        Values below this percentile are considered outliers (default is 0.025).
    `upper`: `float`
        Values above this percentile are considered outliers (default is 0.975).
    `smoothing`: `int`
        Amount (sigma) of Gaussian smoothing applied to both distributions
        (default is 35).
    `n_bins`: `int`
        Number of bins to calculate distribution histgram (default is 1000).

    References
    ----------
    DeepCPP: Zhang et al. (2020) https://doi.org/10.1093/bib/bbaa039'''

    def __init__(self, k, lower=0.025, upper=0.975, smoothing=35, n_bins=1000):
        super().__init__('mDS', 'KL divergence', k)
        self.lower = lower
        self.upper = upper
        self.smoothing = smoothing
        self.n_bins = n_bins

    def select_features(self, data, feature_names):
        feature_importance = np.zeros(len(feature_names))
        for i, feature_name in enumerate(feature_names):
            feature_importance[i] = self.calculate(data, feature_name)
        idx = self.k_most_important_indices(feature_importance)
        return np.array(feature_names)[idx], feature_importance
    
    def calculate(self, data, feature_name):
        '''Calculates Minimum Distribution Similarity (mDS) for given
        `feature_name` in `data`.'''

        lower = data.df[feature_name].quantile(self.lower)
        upper = data.df[feature_name].quantile(self.upper)

        hist_pcrna, bins = np.histogram(
            data.df[data.df['label']=='pcRNA'][feature_name], 
            range=(lower,upper), bins=self.n_bins, density=True)
        hist_ncrna, _ = np.histogram(
            data.df[data.df['label']=='ncRNA'][feature_name], 
            bins=bins, density=True)

        hist_pcrna = gaussian_filter1d(hist_pcrna, self.smoothing) 
        hist_ncrna = gaussian_filter1d(hist_ncrna, self.smoothing) 

        return entropy(hist_pcrna + 1e-10, hist_ncrna + 1e-10)