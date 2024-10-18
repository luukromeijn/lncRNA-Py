'''Contains modules related to motif encoding, which uses a simple 1D 
convolutional neural network to extract motifs from input sequences. This is
similar to how Vision Transformers (ViTs) work. 

References
----------
ViT: Dosovitskiy et al. (2020) https://doi.org/10.48550/arXiv.2010.11929'''

import torch
import matplotlib.pyplot as plt
import numpy as np


# TODO change name / remove if not working well
class KmerEncoding(torch.nn.Module):
    '''Efficient k-mer encoding using a small 1D CNN'''

    def __init__(self, motif_size=10, activate_motifs=True):
        '''Initializes `KmerEncoding` object.''' # TODO expand docs
        super().__init__()
        kernels = torch.zeros((motif_size*4, 4, motif_size))
        for i in range(40):
            kernels[i, int(i/10), i % 10] = 1
        self.kernels = torch.nn.Parameter(kernels, requires_grad=False)         # NOTE: now that this is an official parameter, we can't unfreeze() the model anymore without danger!
        self.relu = torch.nn.ReLU() if activate_motifs else False
        self.motif_size = motif_size

    def forward(self, x):
        x = torch.conv1d(x, self.kernels, stride=self.motif_size)
        return self.relu(x) if self.relu else x


class MotifEncoding(torch.nn.Module):
    '''Implementation for motif encoding using a small 1D CNN.'''

    def __init__(self, n_motifs, motif_size=10, activate_motifs=True, 
                 n_hidden_motifs=0):
        '''Initializes `MotifEncoding` object.
        
        Arguments
        ---------
        `n_motifs`: `int`
            Number of motifs to learn from the data.
        `motif_size`: `int`
            Number of nucleotides that make up a single motif (default is 10).
        `activate_motifs`: `bool`
            Whether or not motifs are relu-activated (default is True).
        `n_hidden_motifs`: `int`
            If > 0, adds an extra hidden convolutional layer with a kernel size
            and stride of 3 with the specified amount of output channels. This
            layer precedes the normal motif encoding layer (default is 0).'''

        super().__init__()
        self.motif_layers = torch.nn.ModuleList()
        in_channels = 4
        kernel_size = motif_size

        # Defining the hidden motif layer if specified by user
        if n_hidden_motifs > 0:
            if motif_size % 3 != 0:
                raise ValueError('motif_size should be multiple of 3 when ' + 
                                 'n_hidden_motifs > 0.')
            self.motif_layers.append(
                torch.nn.Conv1d(
                    in_channels=4, out_channels=n_hidden_motifs, kernel_size=3, 
                    stride=3
                )
            )
            in_channels = n_hidden_motifs
            kernel_size = int(motif_size/3)

        # Defining the main motif layer
        self.motif_layers.append(
            torch.nn.Conv1d(
                in_channels=in_channels, out_channels=n_motifs,
                kernel_size=kernel_size, stride=kernel_size
            )
        )
        if activate_motifs:
            self.motif_layers.append(torch.nn.ReLU())

    def forward(self, x):
        for motif_layer in self.motif_layers:
            x = motif_layer(x)
        return x
    
    def visualize(self, motif_idx, filepath=None):
        '''Visualizes a certain motif, indicated by `motif_idx`. In case of 
        hidden motifs, will visualize the first motif layer (= hidden).'''

        # Get motif tensor
        motif_tensor = [p for p in self.motif_layers[0].parameters()][0]
        motif_tensor = motif_tensor[motif_idx].detach().cpu().numpy()

        labels = 'ACGT'
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

    def __init__(self, n_motifs, d_model, motif_size=10, project_motifs=True,
                 activate_motifs=True, n_hidden_motifs=0, fixed_motifs=False):
        '''Initializes `MotifEncoding` class.

        Arguments
        ---------
        `n_motifs`: `int`
            Number of motifs to learn from the data.
        d_model: int
            Dimension of sequence repr. (embedding) in BERT model.
        `motif_size`: `int`
            Number of nucleotides that make up a single motif (default is 10).
        project_motifs: bool
            Whether or not motifs are projected with a linear layer onto 
            `d_model` dimensions. Must be True when `d_model != n_motifs` 
            (default is True)
        activate_motifs: bool
            Whether or not motifs are relu-activated (default is True)
        `n_hidden_motifs`: `int`
            If > 0, adds an extra hidden convolutional layer with a kernel size
            and stride of 3 with the specified amount of output channels. This
            layer precedes the normal motif encoding layer (default is 0).'''
        
        super().__init__()
        if not project_motifs and d_model != n_motifs:
            raise ValueError("project_motifs must be True when " + 
                             "d_model != n_motifs")
        if fixed_motifs: 
            self.motif_encoder = KmerEncoding(motif_size, activate_motifs)
        else:
            self.motif_encoder = MotifEncoding(n_motifs, motif_size, 
                                               activate_motifs, n_hidden_motifs)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        if project_motifs:
            self.linear = torch.nn.Linear(n_motifs, d_model)
        else:
            self.linear = False

    def forward(self, x):
        x = self.motif_encoder(x).transpose(1,2) # Run through motif layer
        if self.linear:
            x = self.linear(x) # Project to model's dimensionality
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1) # Add CLS tokens
        x = torch.cat((cls_tokens, x), dim=1)
        return x