'''Contains modules related to motif encoding, which uses a simple 1D 
convolutional neural network to extract motifs from input sequences. This is
similar to how Vision Transformers (ViTs) work. 

References
----------
ViT: Dosovitskiy et al. (2020) https://doi.org/10.48550/arXiv.2010.11929'''

import torch
import matplotlib.pyplot as plt
import numpy as np


class MotifEncoding(torch.nn.Module):
    '''Implementation for motif encoding using a small 1D CNN.'''

    def __init__(self, n_motifs, motif_size=12):
        '''Initializes `MotifEncoding` object.
        
        Arguments
        ---------
        `n_motifs`: `int`
            Number of motifs to learn from the data.
        `motif_size`: `int`
            Number of nucleotides that make up a single motif (default is 12).
        '''

        super().__init__()
        self.motif_layer = torch.nn.Conv1d(
            in_channels=4,
            out_channels=n_motifs,
            kernel_size=motif_size,
            stride=motif_size
        )

    def forward(self, x):
        return self.motif_layer(x)
    
    def visualize(self, motif_idx, filepath=None):
        '''Visualizes a certain motif, indicated by `motif_idx`.'''

        # Get motif tensor
        motif_tensor = [p for p in self.motif_layer.parameters()][0].detach()
        motif_tensor = motif_tensor[motif_idx].cpu().numpy()

        labels = 'ACTG'
        for i in range(4):
            # Calculate bottom coordinates (based on cumulative sum)
            pos_case = np.where( # Sum all smaller values (to use as bottombase)
                [(motif_tensor[i] > motif_tensor) & (motif_tensor > 0)], 
                motif_tensor, 0
            ).squeeze().sum(axis=0)
            neg_case = np.where( # Sum all larger values (when negative)
                [(motif_tensor[i] < motif_tensor) & (motif_tensor < 0)], 
                motif_tensor, 0
            ).squeeze().sum(axis=0)
            # Apply either the positive or negative case based on value
            bottom = np.where(motif_tensor[i] >= 0, pos_case, neg_case)
            # And do the actual plotting
            fig = plt.bar(np.arange(1, motif_tensor.shape[-1]+1), 
                          motif_tensor[i], bottom=bottom, label=labels[i])

        # Making the plot nicer
        plt.xticks(np.arange(1, motif_tensor.shape[-1]+1))
        plt.axhline(c='black')
        plt.xlabel('Position')
        plt.ylabel('Kernel weight')
        plt.legend()
        plt.tight_layout()

        if filepath is not None:
            plt.savefig(filepath)
        return fig
    

class MotifEmbedding(torch.nn.Module):
    '''Projects motif encoding into space of pre-specified dimensionality.'''

    def __init__(self, n_motifs, d_model, motif_size=12, relu=True):
        '''Initializes `MotifEncoding` class.

        Arguments
        ---------
        `n_motifs`: `int`
            Number of motifs to learn from the data.
        d_model: int
            Dimension of sequence repr. (embedding) in BERT model.
        `motif_size`: `int`
            Number of nucleotides that make up a single motif (default is 12).
        `relu`: `bool`
            Whether or not motifs are relu-activated (default is True).'''
        
        super().__init__()
        self.motif_encoder = MotifEncoding(n_motifs, motif_size)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        self.linear = torch.nn.Linear(n_motifs, d_model)
        self.relu = torch.nn.ReLU() if relu else None

    def forward(self, x, mask=None):
        x = self.motif_encoder(x).transpose(1,2) # Run through motif layer
        if self.relu:
            x = self.relu(x)
        x = self.linear(x) # Project to model's dimensionality
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1) # Add CLS tokens
        x = torch.cat((cls_tokens, x), dim=1)
        if mask is not None: # If mask, add mask token embedding at these indexs
            mask = mask.unsqueeze(-1)
            x = torch.where(mask, self.mask_token, x)
        return x