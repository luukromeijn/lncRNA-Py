'''Masked Language Modeling pre-training task for nucleotide sequences that are
encoded using Motif Encoding.

References
----------
MycoAI: Romeijn et al. (2024) https://doi.org/10.1111/1755-0998.14006
Huang et al. (2022) https://nlp.seas.harvard.edu/annotated-transformer'''

import torch
from torch.utils.data import DataLoader, RandomSampler
from lncrnapy import utils
from lncrnapy.train.mixed_precision import get_gradient_scaler, get_amp_args
from lncrnapy.train.loggers import LoggerBase
from lncrnapy.train.lr_schedule import LrSchedule
from lncrnapy.train.metrics import mmm_metrics


def train_masked_motif_modeling(
        model, train_data, valid_data, epochs, n_samples_per_epoch=None,
        batch_size=8, p_mlm=0.15, p_mask=0.8, p_random=0.1, warmup_steps=32000,
        loss_function=None, mask_size=1, random_reading_frame=True, logger=None,
        metrics=mmm_metrics
    ):
    '''Trains `model` for Masked Language Modeling task, using `train_data`, 
    for specified amount of `epochs`. Assumes sequence data is inputted in 
    four channels (using `Data.set_tensor_features('4D-DNA')`), and a model of 
    type `MaskedMotifModel`.
    
    Arguments
    ---------
    `model`: `torch.nn.Module` | `lncrnapy.modules.MaskedMotifModel`
        Neural network that is to be trained.
    `train_data`: `lncrnapy.data.Data`
        Data to use for training, must call `set_tensor_features(4D-DNA)` first.
        After every training epoch, the performance of the model on a random 
        subset of the training set is determined. The length of this subset is 
        `min(len(train_data), len(valid_data))`. 
    `valid_data`: `lncrnapy.data.Data`
        Data to use for validation, must call `set_tensor_features` first.
    `epochs`: `int`
        How many epochs (data run-throughs) to train for.
    `n_samples_per_epoch`: `int`
        If specified, indicates the number of samples per training epoch. If 
        None, will sample the full training set.
    `batch_size`: `int`
        Number of examples per batch (default is 64).
    `p_mlm`: `float`
        Probability for a nucleotide to be selected for MLM (default is 0.15).
    `p_mask`: `float`
        Probability for a nucleotide to be masked when selected (default 0.8).
    `p_random`: `float`
        Probability for a nucleotide to be randomly replaced when selected 
        (default is 0.1).
    `warmup_steps`: `int`
        Number of training steps in which learning rate linearly increases. 
        After this amount of steps, the learning rate decreases proportional to
        the invserse square root of the step number (default is 32000).
    `loss_function`: `torch.nn.Module`
        Loss function that is to be optimized, assuming logits (so no Softmax) 
        and ignore_index=-1. Uses `torch.nn.CrossEntropyLoss` if None (default).
    `mask_size`: `int`:
        Number of contiguous nucleotides that make up a mask (default is 1).
    `random_reading_frame`: `bool`:
        If True (default), trains the model with sequences that have been
        frameshifted by a random number (between [0,motif_size]).
    `logger`: `lncrnapy.train.loggers`
    	Logger object whose `log` method will be called at every epoch. If None
        (default), will use LoggerBase, which only keeps track of the history.
    `metrics`: `dict[str:callable]`
        Metrics (name + function) that will be evaluated at every epoch.'''

    # Initializing required objects
    if n_samples_per_epoch is None:
        n_samples_per_epoch = len(train_data)
    if random_reading_frame:
        train_data.set_random_reading_frame(model.base_arch.motif_size-1)
    sampler = RandomSampler(train_data, num_samples=n_samples_per_epoch)
    train_dataloader = DataLoader(train_data, batch_size, sampler=sampler)
    train_subset = train_data.sample(N=min(len(valid_data), len(train_data)))
    mask_size = model.base_arch.motif_size if mask_size is None else mask_size
    if loss_function is None:
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9,0.98))
    lr_scheduler = LrSchedule(optimizer, model.base_arch.d_model, warmup_steps)
    scaler = get_gradient_scaler(utils.DEVICE)
    logger = logger if logger else LoggerBase()
    logger.start(metrics)

    print("Training MLM...")
    for i in utils.progress(range(epochs)):
        model = epoch(model, train_dataloader, p_mlm, p_mask, p_random, 
                      mask_size, loss_function, optimizer, scaler, lr_scheduler)
        train_results = evaluate(model, train_subset, p_mlm, p_mask, p_random, 
                                 mask_size, loss_function, metrics) 
        valid_results = evaluate(model, valid_data, p_mlm, p_mask, p_random,
                                 mask_size, loss_function, metrics) 
        logger.log(train_results+valid_results, model)

    # Finish
    logger.finish()
    train_data.set_random_reading_frame(0)
    return model, logger.history


def epoch(model, dataloader, p_mlm, p_mask, p_random, mask_size, loss_function, 
          optimizer, scaler, lr_scheduler):
    '''Trains `model` for a single epoch.'''
    model.train() # Set training mode
    for X, _ in dataloader: # Loop through data
        X, y = mask_batch(X, model.base_arch.motif_size, p_mlm, p_mask, 
                          p_random, mask_size)
        optimizer.zero_grad() # Zero out gradients
        with torch.autocast(**get_amp_args(utils.DEVICE)):
            y_pred = model(X) # Make prediction
            y = y.flatten()
            y_pred = y_pred.transpose(1,2).reshape(-1,4)
            skip = y.sum() == -1 * y.shape[0] # Check if all -1 (ignore_index)
            if not skip: 
                loss = loss_function(y_pred, y) # Calculate loss
        if not skip:
            scaler.scale(loss).backward() # Calculate gradients
            scaler.unscale_(optimizer) # Unscale before gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), utils.CLIP_NORM)
            scaler.step(optimizer) # Optimize parameters
            lr_scheduler.step() # Update learning rate
            scaler.update() # Updates scale
    return model # Return model


def mask_batch(X, motif_size, p_mlm, p_mask, p_random, mask_size):
    '''Maks a batch of sequence data for MMM'''

    # Correct p_mlm by mask size
    p_mlm = p_mlm / mask_size

    # Select bases with corrected p_mlm probability
    len_emb = int(X.shape[2] / motif_size)
    len_out = len_emb * motif_size
    X = X[:,:,:len_out]
    mask_shape = (X.shape[0], len_out)
    num_selected = int(p_mlm*len_out)
    selected = torch.multinomial(torch.ones(mask_shape,device=utils.DEVICE),
                                 num_selected)

    # If necessary, expand selected with mask_size consecutive indices
    if mask_size > 1:
        selected = get_consecutive_indices(selected, mask_size, len_out)

    # Divide selected over masked and random
    num_i_masked = int(p_mask*num_selected)*(mask_size) # Calculate number of...
    num_i_random = int(p_random*num_selected)*(mask_size) # ...indices to select 
    masked = selected[:,:num_i_masked] # Divide by slicing
    random = selected[:,num_i_masked:num_i_masked+num_i_random]

    # Convert selected, masked, and random from indices to boolean arrays...
    not_padding = X.sum(axis=1) > 0 # ... and unselect padding idx
    selected = index_to_bool(selected, mask_shape) & not_padding
    masked = index_to_bool(masked, mask_shape) & not_padding
    random = index_to_bool(random, mask_shape) & not_padding

    # Use selected to define y
    y = X.clone()
    y = torch.argmax(y, axis=1)
    y = torch.where(selected, y, -1)

    # Use masked and random to mask/mutate X
    random = random.unsqueeze(1).repeat(1,4,1) # Repeat along DNA channels
    masked = masked.unsqueeze(1).repeat(1,4,1)
    X = torch.where(random, get_random_nucs(X.shape), X)
    X[masked] = 0.25 # We accept that this might overwrite some random nucs

    return X, y


def get_consecutive_indices(indices, mask_size, max_len):
    '''Expands `indices` with up to `mask_size` follow-up indices. Stays 
    within the maximum bound as specified by `max_len`.'''
    indices = indices.unsqueeze(-1)
    to_add = torch.arange(0, mask_size, device=utils.DEVICE)
    indices = indices + to_add
    indices = indices.flatten(1,2)
    indices[indices >= max_len] = max_len-1    
    return indices


def index_to_bool(indices, shape):
    '''Creates a boolean Tensor of specified shape, where `indices` are True.'''
    boolean_tensor = torch.zeros(shape, dtype=torch.bool, device=utils.DEVICE)
    boolean_tensor.scatter_(1, indices, True)
    return boolean_tensor


def get_random_nucs(X_shape):
    '''Returns a tensor of random 4D-DNA encoded nucleotides of `X_shape`.'''
    random_nucs = torch.zeros(X_shape, device=utils.DEVICE)
    batch_i = torch.arange(X_shape[0], device=utils.DEVICE).unsqueeze(1)
    rand_i = torch.randint(4, (X_shape[0], X_shape[2]), device=utils.DEVICE)
    bases_i = torch.arange(X_shape[2], device=utils.DEVICE).unsqueeze(0)
    random_nucs[batch_i, rand_i, bases_i] = 1
    return random_nucs


def evaluate(model, data, p_mlm, p_mask, p_random, mask_size, loss_function, 
             metrics):
    '''Evaluation function to keep track of in-training progress for MMM.'''
    
    # Initialization
    loss = 0 # Running loss
    y_true_all, y_pred_all = [], [] # All predictions/targets
    dataloader = DataLoader(data, model.pred_batch_size, shuffle=False) # Data

    # Prediction
    model.eval() # Set in evaluation mode and turn off gradient calculation
    with torch.no_grad():
        for X, _ in dataloader: # Loop through + mask data
            X, y_true = mask_batch(X, model.base_arch.motif_size, p_mlm, 
                                   p_mask, p_random, mask_size)                                         
            y_pred = model(X) # Make a prediction
            y_pred = y_pred.transpose(1,2).reshape(-1,4)
            y_true = y_true.flatten()
            select = y_true != -1 # Remove non-selected bases
            y_true, y_pred = y_true[select], y_pred[select] 
            loss += len(y_true)*loss_function(y_pred, y_true).item()
            # Save predictions & targets
            y_true_all.append(y_true.cpu())
            y_pred_all.append(torch.argmax(y_pred, axis=-1).cpu())
    y_true_all = torch.concat(y_true_all) # Concatenate predictions & targets
    y_pred_all = torch.concat(y_pred_all)

    # Evaluation
    scores = [loss / len(y_true_all)] # Total average loss
    for metric in metrics: # Loop through metrics, append score
        metric_function = metrics[metric]
        scores.append(metric_function(y_true_all, y_pred_all))

    return scores