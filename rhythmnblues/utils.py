'''Constants and supporting objects for `rhythmnblues`.'''

import torch

##### CONSTANTS #####
dummy = lambda x: x
# Makes tqdm an optional library
progress = None
'''Yields progress bar if `tqdm` library is installed.'''
try:
    from tqdm import tqdm
    progress = tqdm
except ImportError:
    progress = dummy

DEVICE = (torch.device('cuda') if torch.cuda.is_available() 
                               else torch.device('cpu'))
'''Torch device object to use for tensor operations (GPU/CPU) Defaults to 
`torch.device('cuda')` if available, else defaults to CPU.'''

CLIP_NORM = 1.0 
'''Maximum L2-norm of gradients (all above are clipped) (default is 1.0).'''

TOKENS = {'MASK': 0, 'CLS': 1, 'SEP': 2, 'PAD': 3, 'UNK': 4}
'''Look-up table of special tokens and their associated values. Note that we set
BPE to assume that MASK=0, hence this value should not be changed.'''

NUC_TO_4D = {
    'A': [1,    0,    0,    0   ],
    'C': [0,    1,    0,    0   ],
    'G': [0,    0,    1,    0   ],
    'T': [0,    0,    0,    1   ],
    'R': [0.5,  0,    0.5,  0   ],
    'Y': [0,    0.5,  0,    0.5 ],
    'S': [0,    0.5,  0.5,  0   ],
    'W': [0.5,  0,    0,    0.5 ],
    'K': [0,    0,    0.5,  0.5 ],
    'M': [0.5,  0.5,  0,    0   ],
    'B': [0,    1/3,  1/3,  1/3 ],
    'D': [1/3,  0,    1/3,  1/3 ],
    'H': [1/3,  1/3,  0,    1/3 ],
    'V': [1/3,  1/3,  1/3,  0   ],
    'N': [0.25, 0.25, 0.25, 0.25],
}
'''Mapping from nucleotide indicator to 4D_DNA representation, which is the 
format suitable for convolutional neural network layers.'''

LEN_4D_DNA = 8000
'''Length of the 4D-DNA representation. Longer sequences are truncated, shorter
sequences are zero-padded.'''

##### HELPER FUNCTIONS #####
def watch_progress(on=True):
    '''Switches on/off the `tqdm` progress indicator around large for loops.'''
    global progress
    if on:
        from tqdm import tqdm
        progress = tqdm
    else:
        progress = dummy