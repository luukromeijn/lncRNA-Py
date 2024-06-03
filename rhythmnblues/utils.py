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


##### HELPER FUNCTIONS #####
def watch_progress(on=True):
    '''Switches on/off the `tqdm` progress indicator around large for loops.'''
    global progress
    if on:
        from tqdm import tqdm
        progress = tqdm
    else:
        progress = dummy