'''Contains wrapper classes that enhance a base architecture (which can be any
PyTorch module) with additional requirements for various (pre-)training tasks 
from `rhythmnblues`.'''

import pandas as pd
import torch
from torch.utils.data import DataLoader
from rhythmnblues.modules.bert import BERT


class WrapperBase(torch.nn.Module):
    '''Base class for all wrapper modules in `rhythmnblues`.
    
    Attributes
    ----------
    `base_arch`: `torch.nn.Module`
        PyTorch module to be used as base architecture of the classifier.
    `pred_batch_size`: `int`
        Batch size used by the `predict` method.
    `data_columns`: `list` | `str`
        Data column name for outcome of predict method.
    `latent_space_columns`: `list`
        Data column name for latent space columns (only defined after calling
        `latent_space` method.)'''

    def __init__(self, base_arch, pred_batch_size=64):
        '''Initializes the module for a given base architecture.
        
        Arguments
        ---------
        `base_arch`: `torch.nn.Module`
            PyTorch module to be used as base architecture of the classifier.
        `pred_batch_size`: `int`
            Batch size used by the `predict` method (default is 64).'''
        
        super().__init__()
        self.base_arch = base_arch
        self.pred_batch_size = pred_batch_size

    def forward(self, X):
        '''A forward pass through the neural network.'''
        return self.base_arch(X)
    
    def predict(self, data, inplace=False, **kwargs):
        '''Calls `forward` in batch-wise fashion for all rows in `data`.
        
        Arguments
        ---------
        `data`: `rhythmnblues.data.Data`
            Data object with `tensor_features` attribute.
        `**kwargs`:
            Any keyword argument accepted by the model's forward method.'''
        
        predictions = []
        self.eval()
        with torch.no_grad():
            for X, _ in self._get_predict_dataloader(data):
                predictions.append(self(X, **kwargs).cpu())
        predictions = torch.concatenate(predictions)
        if inplace:
            data.add_feature(predictions, 
                             self._get_data_columns(predictions.shape[-1]))
        else:
            return predictions
    
    def latent_space(self, data, inplace=False, CLS_only=True):
        '''Calculates latent representation for all rows in `data`.'''
        spaces = []
        self.eval()
        with torch.no_grad():
            for X, _ in self._get_predict_dataloader(data):
                y = self.base_arch(X)
                if CLS_only:
                    y = y[:,0,:] # CLS is assumed to be first input position
                spaces.append(y.cpu())
        spaces = torch.concatenate(spaces)
        if inplace:
            self.latent_space_columns = [
                f'L{dim}' for dim in range(spaces.shape[-1])
            ]
            data.add_feature(spaces, self.latent_space_columns)
        else:
            return spaces

    def _get_predict_dataloader(self, data):
        '''Returns unshuffled PyTorch DataLoader for given `data` object.'''
        return DataLoader(data, batch_size=self.pred_batch_size, shuffle=False)
    
    def _get_data_columns(self, n_dims):
        if hasattr(self, 'data_columns'):
            return self.data_columns
        else:
            return [f'P{dim}' for dim in range(n_dims)]


class Classifier(WrapperBase):
    '''Wrapper class that uses a base architecture to perform binary
    classification.'''

    def __init__(self, base_arch, dropout=0.0, pred_batch_size=64):
        super().__init__(base_arch, pred_batch_size)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.output = torch.nn.LazyLinear(1) 
        self.sigmoid = torch.nn.Sigmoid()
        if type(base_arch) == BERT:
            self._forward_base_arch = self._forward_base_arch_bert
        else:
            self._forward_base_arch = self.base_arch
        self.data_columns = 'P(pcrna)'

    def forward(self, X, return_logits=True):
        X = self.output(self.dropout(self._forward_base_arch(X)))
        if return_logits:
            return X
        else:
            return self.sigmoid(X)

    def _forward_base_arch_bert(self, X):
        '''Forward function that extracts the CLS token embedding from BERT.'''
        return self.base_arch(X)[:,0,:] # # CLS assumed at first input position

    def predict(self, data, inplace=False, return_logits=False):
        return super().predict(data, inplace, return_logits=return_logits)
    

class MLM(WrapperBase):
    '''Wrapper class for model that performs Masked Language Modelling.'''

    def __init__(self, base_arch, dropout=0.0, pred_batch_size=64):
        super().__init__(base_arch, pred_batch_size)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.mlm_layer = torch.nn.Linear(base_arch.d_model,base_arch.vocab_size)
        self.vocab_size = base_arch.vocab_size
        self.d_model = base_arch.d_model

    def forward(self, X):
        return self.mlm_layer(self.dropout(self.base_arch(X)))