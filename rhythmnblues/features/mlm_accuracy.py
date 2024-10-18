'''Evaluates a Masked Language Model and adds the per-sequence accuracy as 
feature to the data.'''

import torch
from torch.utils.data import DataLoader
from rhythmnblues.modules import MotifBERT
from rhythmnblues.train.mixed_precision import get_amp_args
from rhythmnblues.train.masked_motif_modeling import mask_batch as mask_motifs
from rhythmnblues.train.masked_token_modeling import mask_batch as mask_tokens
from rhythmnblues import utils


class MLMAccuracy:
    '''Calculates per-sequence MLM accuracy.'''

    def __init__(self, model, p_mlm=0.15, p_mask=0.8, p_random=0.1, 
                 mask_size=1):
        '''TODO'''
        self.model = model
        self.type = 'motif' if type(model.base_arch)==MotifBERT else 'token'
        self.p_mlm = p_mlm
        self.p_mask = p_mask
        self.p_random = p_random
        self.mask_size = mask_size
        self.name = ['Accuracy (MLM)']

    def calculate(self, data):
        '''Calculates MLM accuracy for every row in data.'''
        dataloader = DataLoader(data, batch_size=self.model.pred_batch_size, 
                                shuffle=False)
        self.model.eval()
        print("Calculating MLM accuracies...")
        if self.type == 'motif':
            return self._calculate_motif(dataloader)
        else: # self.type == 'token'
            return self._calculate_token(dataloader)

    def _calculate_motif(self, dataloader):
        '''Hidden method for calculating motif MLM accuracy.'''
        accuracies = []
        for X, _ in utils.progress(dataloader):
            X, y = mask_motifs(
                X, self.model.base_arch.motif_size, self.p_mlm, self.p_mask, 
                self.p_random, self.mask_size
            )
            with torch.no_grad(), torch.autocast(**get_amp_args(utils.DEVICE)):
                accuracies.append( # num(correct) / num(targets)
                    ((y == torch.argmax(self.model(X), dim=1)).sum(dim=1) / 
                     (y != -1).sum(dim=1)).cpu() 
                )
        return torch.cat(accuracies).cpu()
    
    def _calculate_token(self, dataloader):
        '''Hidden method for calculating tokenized MLM accuracy.'''
        accuracies = []
        for X, y in utils.progress(dataloader):
            X, y = mask_tokens(X, self.model.base_arch.vocab_size, self.p_mlm,
                               self.p_mask, self.p_random)
            with torch.no_grad(), torch.autocast(**get_amp_args(utils.DEVICE)):
                accuracies.append( # num(correct) / num(targets)
                    ((y == torch.argmax(self.model(X), dim=-1)).sum(dim=1) / 
                     (y != utils.TOKENS['PAD']).sum(dim=1)).cpu() 
                )
        return torch.cat(accuracies).cpu()