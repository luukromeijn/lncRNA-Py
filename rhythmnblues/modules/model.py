'''Contains wrappe class for the classification of RNA transcripts as either 
protein-coding or long non-coding.'''

import torch
from torch.utils.data import DataLoader
from rhythmnblues import utils


class Model(torch.nn.Module):
    '''Wrapper class for the classification of RNA transcripts as either
    protein-coding or long non-coding (= binary classification). Combines a 
    base architecture with a single, sigmoid-activated output node. Implements
    a predict method that iterates over data in batch-wise fashion.'''

    def __init__(self, base_arch, pred_batch_size=64):
        '''Initializes `Model` object.
        
        Arguments
        ---------
        `base_arch`: `torch.nn.Module`
            PyTorch module to be used as base architecture of the classifier.
        `pred_batch_size`: `int`
            Batch size used by the `predict` method (default is 64).'''
        
        super().__init__()
        self.base_arch = base_arch
        self.output = torch.nn.LazyLinear(1) 
        self.sigmoid = torch.nn.Sigmoid()
        self.pred_batch_size = pred_batch_size
        self = self.to(utils.DEVICE)

    def forward(self, x):
        return self.output(self.base_arch(x))
    
    def predict(self, data, return_logits=False):
        '''Returns protein-coding probabilities for all rows in `data`, 
        predicted in batch-wise fashion.'''
        pred = []
        self.eval()
        with torch.no_grad():
            for X, _ in DataLoader(data, batch_size=self.pred_batch_size, 
                                   shuffle=False):
                X = self.forward(X)
                if return_logits:
                    pred.append(X.cpu())
                else:
                    pred.append(self.sigmoid(X).cpu())
        return torch.concatenate(pred)