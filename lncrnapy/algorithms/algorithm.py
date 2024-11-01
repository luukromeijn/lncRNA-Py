'''Contains `Algorithm` base class for the classificiation of RNA transcritps as
either protein-coding or long non-coding.'''

import torch
import numpy as np


class Algorithm:
    '''Base class for algorithms for the classification of RNA transcripts as 
    either protein-coding or long non-coding.
    
    Attributes
    ----------
    `model`: 
        Underlying classification model. Can be a trained `torch.nn.Module` 
        object with a single, sigmoid-activated output node (lncrnapy 
        `Model` recommended.), or a scikit-learn-style model with a `.fit` and 
        `.classify` method.
    `feature_extractors`: `list`
        Feature extractor or list of feature extractors that are applied to the 
        data if a feature in `used_features` is missing in the input.
    `used_features`: `list[str]`
        Specifies which feature names (data columns) serve as input variables 
        for the model. If None, will use all features from `feature_extractors`.
    '''

    def __init__(self, model, feature_extractors, used_features=None):
        '''Initializes `Algorithm` object.
        
        Arguments
        ---------
        `model`: 
            Underlying classification model. Can be a trained `torch.nn.Module` 
            object with a single, sigmoid-activated output node (lncrnapy 
            `Model` recommended.), or a scikit-learn-style model with a `.fit` 
            and `.classify` method.
        `feature_extractors`:
            Feature extractor or list of feature extractors that will be applied
            to the data if a feature in `used_features` is missing in the input.
        `used_features`: `list[str]`
            Specifies which feature names (data columns) will serve as input
            variables for the model. If None, will use all features from 
            `feature_extractors` (default is None).'''
        
        # Set alternative fit/predict methods in case of PyTorch module
        if isinstance(model, torch.nn.Module):
            self.fit = self._fit_disabled
            self.predict = self._predict_torch
        self.model = model

        # Convert to list in case of single feature extractor
        if type(feature_extractors)==list or type(feature_extractors)==tuple:
            self.feature_extractors = feature_extractors
        else:
            self.feature_extractors = [feature_extractors]

        # If no features specified, infer all names from feature extractors
        if not used_features:
            used_features = []
            for f in self.feature_extractors:
                # Some features have only one name (str) instead of a list
                feature_names = [f.name] if type(f.name) == str else f.name
                used_features = used_features + feature_names
        self.used_features = used_features

    def fit(self, data):
        '''Fits model on `data`, extracting features first if necessary. Will
        only fit on features as specified in the `used_features` attribute.
        This method is disabled when the `model` attribute is a 
        `torch.nn.Module` instance.'''
        data = self.feature_extraction(data)
        y = data.df['label'].replace({'pcRNA':1, 'ncRNA':0})
        y = y.infer_objects(copy=False) # Downcasting from str to int
        self.model.fit(data.df[self.used_features], y)

    def _fit_disabled(self, data):
        '''Disabled fit method for when `model` is `torch.nn.Module`.''' 
        raise AttributeError(
            "Can't fit model attribute of type torch.nn.Module. " +
            "Please use lncrnapy.train module to fit neural networks."
        )

    def predict(self, data):
        '''Classifies `data`, extracting features first if necessary. Will
        only use features as specified in the `used_features` attribute.'''
        self.feature_extraction(data)
        y = self.model.predict(data.df[self.used_features])
        y = np.vectorize({1:'pcRNA', 0:'ncRNA'}.get)(y)
        return y

    def _predict_torch(self, data):
        '''Alternative predict method for when `model` is `torch.nn.Module`.''' 
        self.feature_extraction(data)
        data.set_tensor_features(self.used_features)
        y = self.model.predict(data).round().squeeze().numpy()
        y = np.vectorize({1:'pcRNA', 0:'ncRNA'}.get)(y)
        return y

    def feature_extraction(self, data):
        '''Calls upon the object's feature extractors if a feature in the 
        `used_features` attribute is missing in `data`.'''
        if not data.check_columns(self.used_features, behaviour='bool'):
            for extractor in self.feature_extractors:
                data.calculate_feature(extractor)
        return data