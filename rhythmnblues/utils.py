'''Constants and supporting objects for `rhythmnblues`.'''

import torch
dummy = lambda x: x
# Makes tqdm an optional library
try:
    from tqdm import tqdm
    progress = tqdm
except ImportError:
    progress = dummy


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def watch_progress(on=True):
    '''Switches on/off the `tqdm` progress indicator around large for loops.'''
    global progress
    if on:
        from tqdm import tqdm
        progress = tqdm
    else:
        progress = dummy