'''Contains the Standardizer class, expanding the StandardScaler from 
scikit-learn to match and work with the rhythmnblues API.'''

from sklearn.preprocessing import StandardScaler


class Standardizer:
    '''Integrates the StandardScaler object from scikit-learn with the
    rhythmnblues API. This allows it to operate on `Data` objects and be
    considered a feature extractor. Furthermore, the `inverse_transform` method
    can be called during evaluation of a deep neural network, to get a realistic
    insight in what the model's current error is.'''

    def __init__(self, data, apply_to):
        '''Initializes (and fits) `Standardizer` object.'''
        self.apply_to = [apply_to] if type(apply_to) == str else apply_to
        self.name = [f'S({col})' for col in self.apply_to]
        data.check_columns(self.apply_to)
        self._standardizer = StandardScaler()
        self._standardizer.fit(data.df[self.apply_to])

    def calculate(self, data):
        '''Scales all rows in `data`.'''
        data.check_columns(self.apply_to)
        return self._standardizer.transform(data.df[self.apply_to])
    
    def inverse_transform(self, y):
        '''Scales back `y` into the original scaling.'''
        return self._standardizer.inverse_transform(y)