'''Contains modules related to motif encoding, which uses a simple 1D 
convolutional neural network to extract motifs from input sequences. This is
similar to how Vision Transformers (ViTs) work. 

References
----------
ViT: Dosovitskiy et al. (2020) https://doi.org/10.48550/arXiv.2010.11929'''

import torch


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
        
        self.layers = torch.nn.ModuleList() # TODO implementation as list here is for possible future expansion including codons/frame pooling
        self.layers.append(torch.nn.Conv1d(
            in_channels=4,
            out_channels=n_motifs,
            kernel_size=motif_size,
            stride=motif_size
        ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class MotifEmbedding(torch.nn.Module):
    '''Projects motif encoding into space of pre-specified dimensionality.'''

    def __init__(self, n_motifs, d_model, motif_size=12):
        '''Initializes `MotifEncoding` class.

        Arguments
        ---------
        `n_motifs`: `int`
            Number of motifs to learn from the data.
        d_model: int
            Dimension of sequence repr. (embedding) in BERT model.
        `motif_size`: `int`
            Number of nucleotides that make up a single motif (default is 12).
        '''
        
        super().__init__()
        self.motif_encoder = MotifEncoding(n_motifs, motif_size)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        self.linear = torch.nn.Linear(n_motifs, d_model)

    def forward(self, x):
        x = self.motif_encoder(x).transpose(1,2) # Run through motif layer
        x = self.linear(x) # Project to model's dimensionality
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1) # Add CLS tokens
        x = torch.cat((cls_tokens, x), dim=1)
        return x