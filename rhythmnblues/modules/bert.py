'''Contains base architectures (without output layers) of different neural
network designs.'''

import math
import torch
from rhythmnblues import utils
from rhythmnblues.modules.motif_encoding import MotifEmbedding


class BERT(torch.nn.Module):
    '''BERT base model, transformer encoder to be used for various tasks
    
    References
    ----------
    Transformer: Vaswani et al. (2017) https://doi.org/10.48550/arXiv.1706.03762
    Code: Huang et al. (2022) https://nlp.seas.harvard.edu/annotated-transformer
    BERT: Devlin et al. (2019) https://doi.org/10.48550/arXiv.1810.04805
    MycoAI: Romeijn et al. (2024) https://github.com/MycoAI/MycoAI/'''

    def __init__(self, vocab_size, d_model=256, d_ff=512, h=8, N=6, 
                 dropout=0.1):
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
            Dropout probability to use throughout network (default is 0.1)'''

        super().__init__()
        self.embedder = torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(d_model, d_ff, h, N, dropout)
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, src):
        '''Given a source, retrieve encoded representation'''
        src_mask = (src != utils.TOKENS['PAD']).unsqueeze(-2) # Mask padding
        src_embedding = self.embedder(src) * math.sqrt(self.d_model) # Get embedding
        src_embedding = self.pos_encoder(src_embedding)
        return self.encoder(src_embedding, src_mask)

    def _forward_latent_space(self, src, pooling):
        '''Given a source, retrieve the latent space and apply pooling.'''
        
        y = self.forward(src)

        if pooling == 'CLS':
            y = y[:,0,:] # CLS is assumed to be first input position
        else:
            not_padding = src != utils.TOKENS['PAD']
            if pooling == 'max':
                y[~not_padding] = -torch.inf # Set padding tokens to -inf
                y, _ = y.max(dim=1)
            elif pooling == 'mean':
                y = (y.sum(axis=1) / 
                     not_padding.sum(axis=1).unsqueeze(dim=1))
            else: 
                raise ValueError('Invalid `pooling` value.') 
            
        return y
    

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
    

class PositionalEncoding(torch.nn.Module):
    '''Adds positional information to an inputted embedding.'''

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()

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


class MotifBERT(torch.nn.Module):
    '''BERT variant that takes learnt sequence motifs (instead of tokens) as 
    input. Based on vision transformer. 
    
    References
    ----------
    ViT: Dosovitskiy et al. (2020) https://doi.org/10.48550/arXiv.2010.11929'''

    def __init__(self, n_motifs, motif_size=12, d_model=256, d_ff=512, h=8, N=6, 
                 dropout=0.1, relu=True):
        '''Initializes `MotifBERT`.
        
        Parameters
        ----------
        n_motifs: int
            Number of motifs to learn from the data.
        motif_size: int
            Number of nucleotides that make up a single motif (default is 12).
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
        relu: bool
            Whether or not motifs are relu-activated (default is False)'''

        super().__init__()
        self.motif_embedder = MotifEmbedding(n_motifs, d_model, motif_size,relu)
        self.positional_encoder = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(d_model, d_ff, h, N, dropout)
        self.motif_size = motif_size
        self.n_motifs = n_motifs
        self.d_model = d_model

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, src, mmm_mask=None):
        '''Given a source, retrieve encoded representation'''

        src_lengths = ( # Calculate start of zero-padding in convolved output...
            torch.count_nonzero(src.sum(axis=1), dim=1) / self.motif_size
        ).to(torch.int32).unsqueeze(-1) # ... and round down to nearest int
        
        # Embed using motif encoding and add positional encoding
        src_embedding = self.motif_embedder(src, mmm_mask)
        src_embedding = src_embedding*math.sqrt(self.d_model)
        src_embedding = self.positional_encoder(src_embedding)

        # Calculate mask
        src_mask = ( # Range to full length
            torch.arange(src_embedding.shape[1], device=utils.DEVICE) 
            <= src_lengths # True when CLS or not part of the zero-padding
        ).unsqueeze(-2)

        return self.encoder(src_embedding, src_mask)
    
    def _forward_latent_space(self, src, pooling):
        '''Given a source, retrieve the latent space and apply pooling.'''
        
        y = self.forward(src)

        if pooling == 'CLS':
            y = y[:,0,:] # CLS is first input position
        else:
            len_seqs = ( # Length of motif-encoded sequences
                torch.count_nonzero(src.sum(axis=1), dim=1) / self.motif_size
            ).to(torch.int32).unsqueeze(-1)
            hide = ~( # Set True part of the zero-padding...
                torch.arange(y.shape[1], device=utils.DEVICE) <= len_seqs)
            hide[:,0] = False # ... or for the CLS token.
            if pooling == 'max':
                y[hide] = -torch.inf
                y, _ = y.max(dim=1)
            elif pooling == 'mean':
                y[hide] = torch.nan
                y = torch.nanmean(y, dim=1)
            else: 
                raise ValueError('Invalid `pooling` value.') 
            
        return y
