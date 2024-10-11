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
    'A': torch.tensor([1.0,  0.0,  0.0,  0.0 ], device=DEVICE),
    'C': torch.tensor([0.0,  1.0,  0.0,  0.0 ], device=DEVICE),
    'G': torch.tensor([0.0,  0.0,  1.0,  0.0 ], device=DEVICE),
    'T': torch.tensor([0.0,  0.0,  0.0,  1.0 ], device=DEVICE),
    'R': torch.tensor([0.5,  0.0,  0.5,  0.0 ], device=DEVICE),
    'Y': torch.tensor([0.0,  0.5,  0.0,  0.5 ], device=DEVICE),
    'S': torch.tensor([0.0,  0.5,  0.5,  0.0 ], device=DEVICE),
    'W': torch.tensor([0.5,  0.0,  0.0,  0.5 ], device=DEVICE),
    'K': torch.tensor([0.0,  0.0,  0.5,  0.5 ], device=DEVICE),
    'M': torch.tensor([0.5,  0.5,  0.0,  0.0 ], device=DEVICE),
    'B': torch.tensor([0.0,  1/3,  1/3,  1/3 ], device=DEVICE),
    'D': torch.tensor([1/3,  0.0,  1/3,  1/3 ], device=DEVICE),
    'H': torch.tensor([1/3,  1/3,  0.0,  1/3 ], device=DEVICE),
    'V': torch.tensor([1/3,  1/3,  1/3,  0.0 ], device=DEVICE),
    'N': torch.tensor([0.25, 0.25, 0.25, 0.25], device=DEVICE),
}
'''Mapping from nucleotide indicator to 4D_DNA representation, which is the 
format suitable for convolutional neural network layers.'''


##### HELPER FUNCTIONS #####
def change_device(device): # NOTE: not sure if this is neccessary
    '''Safely change PyTorch device, also moves certain constant variables 
    contained within `rhythmnblues.utils` to the specified device.'''
    global NUC_TO_4D, DEVICE
    DEVICE = torch.device(device)
    for nuc in NUC_TO_4D:
        NUC_TO_4D[nuc] = NUC_TO_4D[nuc].to(DEVICE)

def watch_progress(on=True):
    '''Switches on/off the `tqdm` progress indicator around large for loops.'''
    global progress
    if on:
        from tqdm import tqdm
        progress = tqdm
    else:
        progress = dummy

def freeze(model, unfreeze=False):
    '''(Un-)Freezes all weights in specified model.'''
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = unfreeze