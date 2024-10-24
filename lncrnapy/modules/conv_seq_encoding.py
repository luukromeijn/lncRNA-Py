'''Contains modules related to convolutional sequence encoding, which uses a 
simple 1D convolutional neural network to extract kernels from input sequences. 
This is similar to how Vision Transformers (ViTs) work. 

References
----------
ViT: Dosovitskiy et al. (2020) https://doi.org/10.48550/arXiv.2010.11929'''

import torch
import matplotlib.pyplot as plt
import numpy as np


class ConvSeqEncoding(torch.nn.Module):
    '''Implementation for convolutional sequence encoding using a small 1D CNN.
    '''

    def __init__(self, n_kernels, kernel_size=10, input_relu=True, 
                 n_hidden_kernels=0):
        '''Initializes `ConvSeqEncoding` object.
        
        Arguments
        ---------
        `n_kernels`: `int`
            Number of kernels to learn from the data.
        `kernel_size`: `int`
            Number of nucleotides that make up a single kernel (default is 10).
        `input_relu`: `bool`
            Whether or not convolutions are relu-activated (default is True).
        `n_hidden_kernels`: `int`
            If > 0, adds an extra hidden convolutional layer with a kernel size
            and stride of 3 with the specified amount of output channels. This
            layer precedes the normal convolutional sequence encoding layer 
            (default is 0).'''

        super().__init__()
        self.cse_layers = torch.nn.ModuleList()
        in_channels = 4
        _kernel_size = kernel_size

        # Defining the hidden kernel layer if specified by user
        if n_hidden_kernels > 0:
            if kernel_size % 3 != 0:
                raise ValueError('kernel_size should be multiple of 3 when ' + 
                                 'n_hidden_kernels > 0.')
            self.cse_layers.append(
                torch.nn.Conv1d(
                    in_channels=4, out_channels=n_hidden_kernels, kernel_size=3, 
                    stride=3
                )
            )
            in_channels = n_hidden_kernels
            _kernel_size = int(kernel_size/3)

        # Defining the main kernel layer
        self.cse_layers.append(
            torch.nn.Conv1d(
                in_channels=in_channels, out_channels=n_kernels,
                kernel_size=_kernel_size, stride=_kernel_size
            )
        )
        if input_relu:
            self.cse_layers.append(torch.nn.ReLU())

    def forward(self, x):
        for cse_layer in self.cse_layers:
            x = cse_layer(x)
        return x
    
    def visualize(self, kernel_idx, filepath=None):
        '''Visualizes a certain kernel, indicated by `kernel_idx`. In case of 
        hidden kernels, will visualize the first kernel layer (= hidden).'''

        # Get kernel tensor
        kernel_tensor = [p for p in self.cse_layers[0].parameters()][0]
        kernel_tensor = kernel_tensor[kernel_idx].detach().cpu().numpy()

        labels = 'ACGT'
        for i in range(4):
            # Calculate bottom coordinates (based on cumulative sum)
            pos_case = np.where( # Sum all smaller values (to use as bottombase)
                [(kernel_tensor[i] > kernel_tensor) & (kernel_tensor > 0)], 
                kernel_tensor, 0
            ).squeeze().sum(axis=0)
            neg_case = np.where( # Sum all larger values (when negative)
                [(kernel_tensor[i] < kernel_tensor) & (kernel_tensor < 0)], 
                kernel_tensor, 0
            ).squeeze().sum(axis=0)
            # Apply either the positive or negative case based on value
            bottom = np.where(kernel_tensor[i] >= 0, pos_case, neg_case)
            # And do the actual plotting
            fig = plt.bar(np.arange(1, kernel_tensor.shape[-1]+1), 
                          kernel_tensor[i], bottom=bottom, label=labels[i])

        # Making the plot nicer
        plt.xticks(np.arange(1, kernel_tensor.shape[-1]+1))
        plt.axhline(c='black')
        plt.xlabel('Position')
        plt.ylabel('Kernel weight')
        plt.legend()
        plt.tight_layout()

        if filepath is not None:
            plt.savefig(filepath)
        return fig
    

class ConvSeqEmbedding(torch.nn.Module):
    '''Projects convolutional sequence encoding into space of pre-specified 
    dimensionality.'''

    def __init__(self, n_kernels, d_model, kernel_size=10, input_linear=True,
                 input_relu=True, n_hidden_kernels=0):
        '''Initializes `ConvSeqEncoding` class.

        Arguments
        ---------
        `n_kernels`: `int`
            Number of kernels to learn from the data.
        d_model: int
            Dimension of sequence repr. (embedding) in BERT model.
        `kernel_size`: `int`
            Number of nucleotides that make up a single kernel (default is 10).
        input_linear: bool
            Whether or not convolutions are projected with a linear layer onto 
            `d_model` dimensions. Must be True when `d_model != n_kernels` 
            (default is True)
        input_relu: bool
            Whether or not convolutions are relu-activated (default is True)
        `n_hidden_kernels`: `int`
            If > 0, adds an extra hidden convolutional layer with a kernel size
            and stride of 3 with the specified amount of output channels. This
            layer precedes the normal convolutional sequence encoding layer 
            (default is 0).'''
        
        super().__init__()
        if not input_linear and d_model != n_kernels:
            raise ValueError("input_linear must be True when " + 
                             "d_model != n_kernels")
        self.conv_seq_encoder = ConvSeqEncoding(n_kernels, kernel_size, 
                                           input_relu, n_hidden_kernels)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        if input_linear:
            self.linear = torch.nn.Linear(n_kernels, d_model)
        else:
            self.linear = False

    def forward(self, x):
        x = self.conv_seq_encoder(x).transpose(1,2) # Run through kernel layer
        if self.linear:
            x = self.linear(x) # Project to model's dimensionality
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1) # Add CLS tokens
        x = torch.cat((cls_tokens, x), dim=1)
        return x