'''Contains wrapper classes that enhance a base architecture (which can be any
PyTorch module) with additional requirements for various (pre-)training tasks 
from `lncrnapy`.'''

import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from lncrnapy.modules.bert import BERT, CSEBERT
from lncrnapy.modules.cnn import MycoAICNN, ResNet, CSEResNet
from lncrnapy.data import reduce_dimensionality
from huggingface_hub import PyTorchModelHubMixin


class WrapperBase(torch.nn.Module, PyTorchModelHubMixin):
    '''Base class for all wrapper modules in `lncrnapy`.
    
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

    def __init__(self, base_arch_config, base_arch=None, pred_batch_size=8):
        '''Initializes the module for a given base architecture.
        
        Arguments
        ---------
        `base_arch_config`: `dict`
            Configuration property of the base architecture (retrieve this 
            through `base_arch.config`), specifying what type of base
            architecture this wrapper contains. Must be used in combination 
            with a matching `base_arch` by the user. 
        `base_arch`: `torch.nn.Module`
            PyTorch module to be used as base architecture of the classifier.
        `pred_batch_size`: `int`
            Batch size used by the `predict` method (default is 64).'''
        
        super().__init__()
        archs = {
            'BERT': BERT,
            'CSEBERT': CSEBERT,
            'MycoAICNN': MycoAICNN,
            'ResNet': ResNet,
            'CSEResNet': CSEResNet
        }
        if base_arch is None:
            base_arch = archs[base_arch_config.pop('type')]
            base_arch = base_arch(**base_arch_config)
        self.base_arch = base_arch
        self.pred_batch_size = pred_batch_size

    def forward(self, X):
        '''A forward pass through the neural network.'''
        return self.base_arch(X)
    
    def predict(self, data, inplace=False, **kwargs):
        '''Calls `forward` in batch-wise fashion for all rows in `data`.
        
        Arguments
        ---------
        `data`: `lncrnapy.data.Data`
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
    
    def latent_space(self, data, inplace=False, pooling=None, dim_red=TSNE()):
        '''Calculates latent representation for all rows in `data`.
        
        Arguments
        ---------
        `data`: `lncrnapy.data.Data`
            Data object for which latent space should be calculated.
        `inplace`: `bool`
            If True, adds latent space as feature columns to `data`.
        `pooling`: ['CLS', 'max', 'mean', None]
            How to aggregate token embeddings (for BERT architectures).
            * 'CLS': use only CLS token.
            * 'max': max pooling over (non-padding) token embeddings.
            * 'mean': mean pooling over (non-padding) token embeddings.
            * None (default): no pooling, e.g. for CNN base architectures.
        `dim_red`: `sklearn` | `NoneType`
            Dimensionality reduction algorithm from `sklearn` to use.'''
        
        if pooling is not None and type(self.base_arch) not in [BERT,CSEBERT]:
            raise TypeError("self.base_arch must be of type BERT or CSEBERT" + 
                            f" for {pooling} pooling.")

        spaces = []
        self.eval()
        with torch.no_grad():
            for X, _ in self._get_predict_dataloader(data):
                if pooling is None:
                    y = self.base_arch(X)
                else:
                    y = self.base_arch._forward_latent_space(X, pooling)
                spaces.append(y.cpu())
        spaces = torch.concatenate(spaces)

        if dim_red is not None:
            spaces = reduce_dimensionality(spaces, dim_red)

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

    def __init__(self, base_arch_config, base_arch=None, dropout=0.0, 
                 pooling='CLS', hidden_layers=[], pred_batch_size=8):
        super().__init__(base_arch_config, base_arch, pred_batch_size)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.hidden_layers = torch.nn.ModuleList([
            torch.nn.ModuleList([torch.nn.LazyLinear(nodes), torch.nn.ReLU()])
            for nodes in hidden_layers
        ])
        self.output = torch.nn.LazyLinear(1) 
        self.sigmoid = torch.nn.Sigmoid()
        if type(self.base_arch) == BERT or type(self.base_arch) == CSEBERT:
            self._forward_base_arch = self._forward_base_arch_bert
            self.pooling = pooling
            in_features = self.base_arch.d_model
            self.hidden_layers = torch.nn.ModuleList()
            for n_nodes in hidden_layers:
                self.hidden_layers.append(torch.nn.Linear(in_features, n_nodes))
                in_features = n_nodes
            self.output = torch.nn.Linear(in_features, 1)
        else:
            self._forward_base_arch = self.base_arch
        self.data_columns = 'P(pcRNA)'

    def forward(self, X, return_logits=True):
        X = self.dropout(self._forward_base_arch(X))
        for hidden_layer, relu in self.hidden_layers:
            X = hidden_layer(X)
            X = relu(X)
        X = self.output(X)
        if return_logits:
            return X
        else:
            return self.sigmoid(X)

    def _forward_base_arch_bert(self, X):
        '''Forward function that extracts the CLS token embedding from BERT.
        (or applies the preset pooling procedure)'''
        return self.base_arch._forward_latent_space(X, self.pooling)

    def predict(self, data, inplace=False, return_logits=False):
        return super().predict(data, inplace, return_logits=return_logits)
    

class MaskedTokenModel(WrapperBase):
    '''Wrapper class for model that performs Masked Language Modeling with 
    tokenized sequences as input.'''

    def __init__(self, base_arch_config, base_arch=None, dropout=0.0, 
                 pred_batch_size=8):
        super().__init__(base_arch_config, base_arch, pred_batch_size)
        if type(self.base_arch) != BERT:
            raise TypeError("Base architecture should be of type BERT.")
        self.dropout = torch.nn.Dropout(p=dropout)
        self.mlm_layer = torch.nn.Linear(self.base_arch.d_model,
                                         self.base_arch.vocab_size)

    def forward(self, X):
        return self.mlm_layer(self.dropout(self.base_arch(X)))


class MaskedConvModel(WrapperBase):
    '''Wrapper class for model that performs Masked Language Modeling with 
    cse-encoded sequences as input.'''

    def __init__(self, base_arch_config, base_arch=None, dropout=0.0, 
                 n_hidden_kernels=0, output_linear=True, output_relu=False, 
                 pred_batch_size=8):
        '''Initializes `MaskedConvModel` object.
        
        Arguments
        ---------
        `base_arch_config`: `dict`
            Configuration property of the base architecture (retrieve this 
            through `base_arch.config`), specifying what type of base
            architecture this wrapper contains. Must be used in combination 
            with a matching `base_arch` by the user. 
        `base_arch`: `lncrnapy.modules.CSEBERT`
            PyTorch module to be used as base architecture of the model.
        `dropout`: `float`
            Amount of dropout to apply to the pre-final layer (default is 0). 
        `output`: 'nucleotides' | 'triplets'
            Output prediction level (default is 'nucleotides').
        `n_hidden_kernels`: `int` 
            If > 0, adds an extra hidden convolutional layer with a kernel size
            and stride of 3 with the specified amount of output channels. This
            layer is added before the final output layer (default is 0).
        `output_linear`: `bool`
            Whether to project the base architecture's embeddings onto a 
            `n_kernels` number of dimensions before the output layer (default is
            True)
        `output_relu`: `bool`
            Whether to apply ReLU activation before transposed convolution(s) 
            (default is False).
        `pred_batch_size`: `int`
            Batch size used by the `predict` method (default is 64).'''
        
        super().__init__(base_arch_config, base_arch, pred_batch_size)
        self.dropout = torch.nn.Dropout(p=dropout)
        if type(self.base_arch) != CSEBERT:
            raise TypeError("Base architecture should be of type CSEBERT.")
        if output_linear:
            self.linear = torch.nn.Linear(self.base_arch.d_model,
                                          self.base_arch.n_kernels)
            in_channels = self.base_arch.d_model
        else:
            self.linear = False
            in_channels = self.base_arch.n_kernels
        kernel_size = self.base_arch.kernel_size
        self.transposed_conv_layers = torch.nn.ModuleList()
        
        # Defining the hidden kernel layer (if specified by user)
        if n_hidden_kernels > 0:
            if kernel_size % 3 != 0:
                raise AttributeError("base_arch.kernel_size should " +
                                  'be multiple of 3 when n_hidden_kernels > 0.')
            self.transposed_conv_layers.append(
                torch.nn.ConvTranspose1d(
                    in_channels=in_channels, out_channels=n_hidden_kernels,
                    kernel_size=3, stride=3
                )
            )
            kernel_size = int(kernel_size/3)
            in_channels = n_hidden_kernels
        
        # Defining the final output kernel layer
        if output_relu:
            self.transposed_conv_layers.append(torch.nn.ReLU())
        self.transposed_conv_layers.append(
            torch.nn.ConvTranspose1d( # 4 output channels (A,C,G,T)
                in_channels=in_channels, out_channels=4,kernel_size=kernel_size,
                stride=kernel_size
            )
        )

    def forward(self, X):
        X = self.base_arch(X)[:,1:,:] # Forward pass base arch, remove CLS
        X = self.dropout(X)
        if self.linear:
            X = self.linear(X) # Transform to kernel 'space' 
        X = X.transpose(1,2)
        for layer in self.transposed_conv_layers:
            X = layer(X) # Apply deconvolution
        return X
    

class Regressor(WrapperBase):
    '''Wrapper class for model that performs linear regression on the base 
    architecture embedding.'''

    def __init__(self, base_arch_config, base_arch=None, n_features=1, 
                 fcn_layers=[], dropout=0.0, pred_batch_size=8, pooling='CLS'):
        super().__init__(base_arch_config, base_arch, pred_batch_size)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.fcn_layers = torch.nn.ModuleList([torch.nn.LazyLinear(n_nodes) 
                                               for n_nodes in fcn_layers])
        self.output = torch.nn.LazyLinear(n_features)
        self.data_columns = [f'F{i}' for i in range(n_features)]
        if type(base_arch) == BERT or type(base_arch) == CSEBERT:
            self._forward_base_arch = self._forward_base_arch_bert
            self.pooling = pooling
            in_features = self.base_arch.d_model
            self.fcn_layers = torch.nn.ModuleList()
            for n_nodes in fcn_layers:
                self.fcn_layers.append(torch.nn.Linear(in_features, n_nodes))
                in_features = n_nodes
            self.output = torch.nn.Linear(in_features, n_features)
        else:
            self._forward_base_arch = self.base_arch

    def forward(self, X):
        X = self.dropout(self._forward_base_arch(X))
        for layer in self.fcn_layers:
            X = torch.relu(layer(X))
        return self.output(X)
    
    def _forward_base_arch_bert(self, X):
        '''Forward function that extracts the CLS token embedding from BERT.
        (or applies the preset pooling procedure)'''
        return self.base_arch._forward_latent_space(X, self.pooling)