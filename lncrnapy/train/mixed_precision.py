'''For automatically en-/disabling mixed precision depending on whether or not
cuda is recognized.'''

import torch


def get_amp_args(device):
    ''''Returns mixed precision keyword arguments (as dictionary) depending on
    whether or not `device.type=='cuda'`.'''
    if device.type == 'cuda':
        return {'device_type': 'cuda', 'dtype': torch.float16}
    else: # CPU
        return {'device_type': 'cpu', 'dtype': torch.bfloat16}


def get_gradient_scaler(device):
    '''Returns gradient scaler or dummy object depending on whether or not 
    `device.type=='cuda'`.'''
    if device.type == 'cuda':
        return torch.cuda.amp.GradScaler()
    else: # CPU
        return DummyScaler()


class DummyScaler:
    '''A dummy gradient scaler that does not do anything and serves as a
    placeholder for when gradient scaling is not desired.'''

    def __init__(self):
        pass

    def scale(self, loss):
        '''Identity function'''
        return loss

    def unscale_(self, optimizer):
        '''Empty function'''
    
    def step(self, optimizer):
        '''Optimizer step'''
        optimizer.step()

    def update(self):
        '''Empty function'''