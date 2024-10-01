'''Evaluates a Masked Language Model and adds the per-sequence accuracy as 
feature to the data.'''

import torch
from torch.utils.data import DataLoader
from rhythmnblues.train.masked_motif_modeling import mask_batch
from rhythmnblues import utils


# NOTE Experimental
class MLMAccuracy:
    '''TODO'''

    def __init__(self, model):
        '''TODO'''
        self.model = model
        self.name = ['Accuracy (MLM)']

    def calculate(self, data):
        '''TODO'''
        dataloader = DataLoader(data, batch_size=self.model.pred_batch_size, 
                                shuffle=False)
        accuracies = []
        print("Calculating MLM accuracies...")
        for X, _ in utils.progress(dataloader):
            X, y = mask_batch(X, self.model.base_arch.motif_size,0.15,0.8,0.1,1)
            accuracies.append( # num(correct) / num(targets)
                (y == torch.argmax(self.model(X), dim=1)).sum(dim=1) / 
                (y != -1).sum(dim=1) 
            )
        return torch.cat(accuracies).cpu()