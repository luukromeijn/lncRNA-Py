'''Contains base architectures (without output layers) of different neural
network designs.'''

import torch


class MycoAICNN(torch.nn.Module):
    '''A simple CNN architecture with conv, batchnorm and maxpool layers, as
    used by MycoAI-CNN. 
    
    References
    ----------
    MycoAI: Romeijn et al. (2024) https://doi.org/todo''' # TODO update
    
    def __init__(self, kernel=5, conv_layers=[5,10], in_channels=1, pool_size=2,
                 batch_normalization=True):
        '''Initializes the `MycoAICNN` object.
        
        Arguments
        ---------
        `kernel`: `list[int]` | `int`
            Indicates the kernel size for all layers (default is 5).
        `conv_layers`: `list[int]`
            Number of convolutions per convolutional layer (default is [5,10]).
        `in_channels`: `int`
            Number of input channels accepted by the CNN (default is 1).
        `pool_size`: `int`
            Kernel size of the max pooling layers (default is 2).
        `batch_normalization`: `bool`
            Whether to apply batch normalization after every convolutional layer
            (default is True).'''
        
        super().__init__()

        conv = []
        kernels = [kernel]*len(conv_layers) if type(kernel)==int else kernel
        for i in range(len(conv_layers)):
            out_channels = conv_layers[i]
            conv.append(torch.nn.Conv1d(in_channels, out_channels, kernels[i], 
                                        padding='same'))
            conv.append(torch.nn.ReLU())
            if batch_normalization:
                conv.append(torch.nn.BatchNorm1d(out_channels))
            conv.append(torch.nn.MaxPool1d(pool_size, 1))
            in_channels = out_channels
        self.conv = torch.nn.ModuleList(conv)
        self.conv_layers = conv_layers
    
    def forward(self, x):
        x = x.unsqueeze(-2)
        for layer in self.conv:
            x = layer(x)
        x = torch.flatten(x, 1)
        return x