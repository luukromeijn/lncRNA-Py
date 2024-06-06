'''Contains base architectures (without output layers) of different neural
network designs.'''

import math
import torch
from rhythmnblues import utils


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


class BERT(torch.nn.Module):
    '''BERT base model, transformer encoder to be used for various tasks
    
    References
    ----------
    Transformer: Vaswani et al. (2017) https://doi.org/10.48550/arXiv.1706.03762
    Code: Huang et al. (2022) https://nlp.seas.harvard.edu/annotated-transformer
    BERT: Devlin et al. (2019) https://doi.org/10.48550/arXiv.1810.04805
    MycoAI: Romeijn et al. (2024) https://doi.org/todo''' # TODO update

    def __init__(self, vocab_size, d_model=256, d_ff=512, h=8, N=6, dropout=0.1, 
                 mode='default'):
        '''Initializes the transformer given the source/target vocabulary.
        
        Parameters
        ----------
        vocab_size: int
            Number of unique tokens in vocabulary. Determine using the
            `vocab_size` method of `TokenizerBase` children.
        d_model: int
            Dimension of sequence repr. (embedding) in model (default is 256)
        d_ff: int
            Dimension of hidden layer FFN sublayers (default is 512)
        h: int
            Number of heads used for multi-head self-attention (default is 8)
        N: int
            How many encoder/decoder layers the transformer has (default is 6)
        dropout: float
            Dropout probability to use throughout network (default is 0.1)
        mode: str
            BERT will use different forward method when mode=='classification'
        '''

        super().__init__()
        self.src_pos_embed = PositionalEmbedding(d_model, vocab_size, dropout)
        self.encoder = Encoder(d_model, d_ff, h, N, dropout)
        self.mlm_layer = torch.nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.N = N
        self.set_mode(mode)

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def set_mode(self, mode):
        '''Uses alternative forward method when mode == 'classificiation'.'''
        if mode == 'classification':
            self.forward = self._forward_classification
        elif mode == 'mlm':
            self.forward = self._forward_mlm
        else:
            self.forward = self._forward_default

    def _forward_default(self, src, src_mask):
        '''Given a source (and mask), retrieve encoded representation'''
        src_embedding = self.src_pos_embed(src)
        return self.encoder(src_embedding, src_mask)
    
    def _forward_classification(self, src):
        '''Given input sequence, retrieve embedding aggregated in CLS token'''
        src_mask = (src != utils.TOKENS['PAD']).unsqueeze(-2) # Mask padding
        src_embedding = self.src_pos_embed(src) 
        # Return only the part of the tensor that corresponds to CLS token
        return self.encoder(src_embedding, src_mask)[:,0,:]

    def _forward_mlm(self, src):
        '''Given input sequence, predict masked tokens'''
        src_mask = (src != utils.TOKENS['PAD']).unsqueeze(-2) # Mask padding
        src = self.encoder(self.src_pos_embed(src), src_mask)
        return self.mlm_layer(src)
    
    def get_config(self):
        config = {}
        if hasattr(self, 'pretraining'):
            config = {'pretraining': self.pretraining}
        return {
            'type':         utils.get_type(self),
            'd_model':      self.d_model,
            'd_ff':         self.d_ff,
            'h':            self.h,
            'N':            self.N,
            **config
        }


class EncoderDecoder(torch.nn.Module):
    '''Transformer encoder-decoder model'''

    def __init__(self, vocab_size, classes, d_model=256, d_ff=512, h=8, 
                 N_encoder=4, N_decoder=2, decoder_self_attn=True, dropout=0.1):
        '''Initializes the transformer given the source/target vocabulary.
        
        Parameters
        ----------
        vocab_size: int
            Number of unique tokens in vocabulary. Can be the vocab_size
            attribute of a BytePairEncoder or KmerTokenizer object. 
        classes: list 
            List that indicates the number of taxonomic classes per level. Can 
            be the classes attribute of a TaxonEncoder object.
        d_model: int
            Dimension of sequence repr. (embedding) in model (default is 512)
        d_ff: int
            Dimension of hidden layer FFN sublayers (default is 2048)
        h: int
            Number of heads used for multi-head self-attention (default is 8)
        N_encoder: int
            How many encoder layers the transformer has (default is 4)
        N_decoder: int
            How many encoder layers the transformer has (default is 2)
        decoder_self_attn: bool
            Whether or not the decoder should have self-attention (default True)
        dropout: float
            Dropout probability to use throughout network (default is 0.1)'''

        super().__init__()
        self.src_pos_embed = PositionalEmbedding(d_model, vocab_size, dropout)
        self.encoder = Encoder(d_model, d_ff, h, N_encoder, dropout)
        self.tgt_pos_embed = PositionalEmbedding(d_model,sum(classes)+2,dropout)
        self.decoder = Decoder(d_model, d_ff, h, N_decoder, dropout, 
                               decoder_self_attn)
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.N_encoder = N_encoder
        self.N_decoder = N_decoder
        self.N = N_encoder + N_decoder
        self.decoder_self_attn = decoder_self_attn

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        '''Full pass through transformer (embedding, encoding, decoding)'''
        src_mask = (src != utils.TOKENS['PAD']).unsqueeze(-2) # Mask padding
        # No target mask used (model always only inputted with 'allowed' part)
        tgt_mask = torch.ones_like(tgt, dtype=torch.bool).unsqueeze(-2) # All 1
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        '''Given a source (and mask), retrieve encoded representation'''
        src_embedding = self.src_pos_embed(src)
        return self.encoder(src_embedding, src_mask)

    def decode(self, encoding, src_mask, tgt, tgt_mask):
        '''Given an encoding and target (+ masks), retrieve decoded repr.'''
        tgt_embedding = self.tgt_pos_embed(tgt)
        return self.decoder(tgt_embedding, encoding, src_mask, tgt_mask)
    
    def get_config(self):
        config = {}
        if hasattr(self, 'pretraining'):
            config = {'pretraining': self.pretraining}
        return {
            'type':                 utils.get_type(self),
            'd_model':              self.d_model,
            'd_ff':                 self.d_ff,
            'h':                    self.h,
            'N':                    self.N,
            'N_encoder':            self.N_encoder,
            'N_decoder':            self.N_decoder,
            'decoder_self_attn':    self.decoder_self_attn,
            **config
        }


class Encoder(torch.nn.Module):
    '''N layers of consisting of self-attention and feed forward sublayers,
    gradually transforms input into encoded representation.'''

    def __init__(self, d_model, d_ff, h, N, dropout):
        super().__init__()
        layers = []
        for i in range(N): 
            sublayers = torch.nn.ModuleList([
                MultiHeadAttention(h, d_model, dropout),
                ResidualConnection(d_model, dropout),
                FeedForward(d_model, d_ff, dropout),
                ResidualConnection(d_model, dropout)
            ])
            layers.append(sublayers)
        self.layers = torch.nn.ModuleList(layers)
        self.norm = torch.nn.LayerNorm(d_model) 

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer[1](x, lambda x: layer[0](x, x, x, mask))
            x = layer[3](x, layer[2])
        return self.norm(x)
    

class Decoder(torch.nn.Module):
    '''N layers of consisting of (masked) (self-)attention and FF sublayers,
    gradually transforms encoder's output and output embedding into decoding'''

    def __init__(self, d_model, d_ff, h, N, dropout, self_attention):
        super().__init__()
        layers = []
        for i in range(N): 
            self_attention_layers = []
            if self_attention:
                self_attention_layers = [
                    MultiHeadAttention(h, d_model, dropout),  
                    ResidualConnection(d_model, dropout)]
            sublayers = torch.nn.ModuleList(
                self_attention_layers + 
                [MultiHeadAttention(h, d_model, dropout), # src attention
                ResidualConnection(d_model, dropout),
                FeedForward(d_model, d_ff, dropout), # feed forward network
                ResidualConnection(d_model, dropout)])
            layers.append(sublayers)
        self.layers = torch.nn.ModuleList(layers)
        self.norm = torch.nn.LayerNorm(d_model) 
        self.self_attention = self_attention

    def forward(self, x, m, src_mask, tgt_mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            if self.self_attention:
                x = layer[1](x, lambda x: layer[0](x, x, x, tgt_mask)) # self att.
            x = layer[-3](x, lambda x: layer[-4](x, m, m, src_mask)) # src att.
            x = layer[-1](x, layer[-2])

        return self.norm(x)


class MultiHeadAttention(torch.nn.Module):
    ''''Performs scaled dot product attention on h uniquely learned linear 
    projections (allowing model to attend to info from different subspaces)'''

    def __init__(self, h, d_model, dropout):
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model)
                                            for i in range(4)]) #NOTE is 4 best?
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2 from 'Attention Is All You Need'"

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
    

class FeedForward(torch.nn.Module):
    '''Simple feed forward network (with dropout applied to mid layer)'''

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.lin_1 = torch.nn.Linear(d_model, d_ff)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.lin_2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.relu(self.lin_1(x))
        x = self.dropout(x)
        return self.lin_2(x)


class PositionalEmbedding(torch.nn.Module):
    '''Converts input into sum of a learned embedding and positional encoding'''

    def __init__(self, d_model, vocab, dropout, max_len=5000):
        super().__init__()

        self.embedder = torch.nn.Embedding(vocab, d_model)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = self.embedder(x) * math.sqrt(self.d_model) # Get embedding
        x = x + self.pe[:, : x.size(1)].requires_grad_(False) # + positional enc
        return self.dropout(x) # Apply dropout
    

class ResidualConnection(torch.nn.Module):
    '''Employs a normalized residual connection followed by dropout'''

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = torch.nn.LayerNorm(size)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, layer):
        '''Adds layer(x) to x and applies normalization/dropout'''
        return x + self.dropout(layer(self.norm(x)))


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask_value = -1e9 if scores.dtype == torch.float32 else -1e4
        scores = scores.masked_fill(mask == 0, mask_value)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn) 
    return torch.matmul(p_attn, value), p_attn