'''Constants and supporting objects for `rhythmnblues`.'''

dummy = lambda x: x

# Makes tqdm an optional library
try:
    from tqdm import tqdm
    progress = tqdm
except ImportError:
    progress = dummy

def watch_progress(on=True):
    '''Switches on/off the `tqdm` progress indicator around large for loops.'''
    global progress
    if on:
        from tqdm import tqdm
        progress = tqdm
    else:
        progress = dummy