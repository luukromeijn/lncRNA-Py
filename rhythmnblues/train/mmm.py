'''Masked Language Modelling pre-training task for nucleotide sequences.

References
----------
MycoAI: Romeijn et al. (2024) https://github.com/MycoAI/MycoAI/
Huang et al. (2022) https://nlp.seas.harvard.edu/annotated-transformer'''

import torch
from torch.utils.data import DataLoader, RandomSampler
from rhythmnblues import utils
from rhythmnblues.train.mixed_precision import get_gradient_scaler, get_amp_args
from rhythmnblues.train.loggers import LoggerBase
from rhythmnblues.train.lr_schedule import LrSchedule
from rhythmnblues.train.metrics import mmm_metrics

# TODO: update documentation to replace all MLM references to MMM!!!
def train_mmm(
        model, train_data, valid_data, epochs, batch_size=64, p_mlm=0.15, 
        p_mask=0.8, p_random=0.1, loss_function=None, warmup_steps=8000, 
        label_smoothing=0.1, n_samples_per_epoch=None, logger=None, 
        metrics=mmm_metrics
    ):
    '''Trains `model` for Masked Language Modelling task, using `train_data`, 
    for specified amount of `epochs`.
    
    Arguments
    ---------
    `model`: `torch.nn.Module` | `rhythmnblues.modules.MLM`
        Neural network that is to be trained.
    `train_data`: `rhythmnblues.data.Data`
        Data to use for training, must call `set_tensor_features` first. After 
        every training epoch, the performance of the model on a subset of the 
        training set is determined. The length of this subset is 
        `min(len(train_data), len(valid_data))`. 
    `valid_data`: `rhythmnblues.data.Data`
        Data to use for validation, must call `set_tensor_features` first.
    `epochs`: `int`
        How many epochs (data run-throughs) to train for.
    `batch_size`: `int`
        Number of examples per batch (default is 64).
    `p_mlm`: `float`
        Probability for a token to be selected for MLM (default is 0.15).
    `p_mask`: `float`
        Probability for a token to be masked when selected (default is 0.8).
    `p_random`: `float`
        Probability for a token to be randomly replaced when selected (default
        is 0.1).
    `loss_function`: `torch.nn.Module`
        Loss function that is to be optimized. If None, falls back to 
        `torch.nn.CrossEntropyLoss`) (default is None).
    `warmup_steps`: `int`
        Number of training steps in which learning rate linearly increases. 
        After this amount of steps, the learning rate decreases proportional to
        the invserse square root of the step number (default is 8000).
    `label_smoothing`: `float`
        How much weight should be subtracted from the target token and divided
        over the remaining tokens, for regularization (default is 0.1).
    `n_samples_per_epoch`: `int`
        If specified, indicates the number of samples per training epoch. If 
        None, will sample the full training set.
    `logger`: `rhythmnblues.train.loggers`
    	Logger object whose `log` method will be called at every epoch. If None
        (default), will use LoggerBase, which only keeps track of the history.
    `metrics`: `dict[str:callable]`
        Metrics (name + function) that will be evaluated at every epoch.'''

    # Initializing required objects
    if n_samples_per_epoch is None:
        n_samples_per_epoch = len(train_data)
    sampler = RandomSampler(train_data, num_samples=n_samples_per_epoch)
    train_dataloader = DataLoader(train_data, batch_size, sampler=sampler)
    train_subset = train_data.sample(N=min(len(valid_data), len(train_data)))
    if loss_function is None:
        loss_function = torch.nn.KLDivLoss(reduction='batchmean') # TODO might need to implement label smoothing?
    optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9,0.98))
    lr_scheduler = LrSchedule(optimizer, model.base_arch.d_model, warmup_steps)
    scaler = get_gradient_scaler(utils.DEVICE)
    logger = logger if logger else LoggerBase()
    logger.start(metrics)

    print("Training MLM...")
    for epoch in utils.progress(range(epochs)):
        model = epoch_mlm(model, train_dataloader, p_mlm, p_mask, p_random, 
                          loss_function, optimizer, scaler, lr_scheduler)
        train_results = evaluate_mmm(model, train_subset, p_mlm, p_mask, 
                                     p_random, loss_function, metrics) 
        valid_results = evaluate_mmm(model, valid_data, p_mlm, p_mask, p_random,
                                     loss_function, metrics) 
        logger.log(train_results+valid_results, model)

    # Finish
    logger.finish()
    return model, logger.history


def epoch_mlm(model, dataloader, p_mlm, p_mask, p_random,
              loss_function, optimizer, scaler, lr_scheduler):
    '''Trains `model` for a single epoch.'''
    model.train() # Set training mode
    for X, _ in utils.progress(dataloader): # Loop through data
        X, mask, y = mask_batch(X, model.base_arch.motif_size, p_mlm, p_mask, 
                                p_random)
        optimizer.zero_grad() # Zero out gradients
        with torch.autocast(**get_amp_args(utils.DEVICE)):
            y_pred = model(X, mask) # Make prediction
            y = y.view(-1, y.shape[-2]) # Combine batch/position axes
            y_pred = y_pred.view(-1, y_pred.shape[-2])
            selected = y.sum(dim=1) > 0 # Only select non-zero entries
            y, y_pred = y[selected], y_pred[selected] 
            if y.shape[0] > 0: # Check that at least one is selected
                loss = loss_function(y_pred, y) # Calculate loss
        if y.shape[0] > 0:
            scaler.scale(loss).backward() # Calculate gradients
            scaler.unscale_(optimizer) # Unscale before gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), utils.CLIP_NORM)
            scaler.step(optimizer) # Optimize parameters
            lr_scheduler.step() # Update learning rate
            scaler.update() # Updates scale
    return model # Return model


def mask_batch(X, motif_size, p_mlm, p_mask, p_random):
    '''Maks a batch of sequence data for MLM'''

    len_embedding = int(X.shape[2]/motif_size)+1 # int(len(seq)/motif_size)+CLS
    len_out = (len_embedding-1)*motif_size # length output tensor 
    len_seqs = (torch.count_nonzero(X.sum(axis=1), dim=1) / motif_size
               ).to(torch.int32).unsqueeze(-1) # round down to nearest int
    
    # Calculate boolean tensors using selection probabilities
    shape = (X.shape[0], len_embedding)
    indices = torch.arange(len_embedding, device=utils.DEVICE)
    select = ((torch.rand(shape, device=utils.DEVICE) < p_mlm) & # Select masked
              (indices > 0) & # But no CLS patch
              (indices <= len_seqs)) # Or padding parts
    
    # Creating the target before making any modifications to X
    y = X[:,:,:len_out].clone()
    select_nucs_b = motifs_to_nucs_mask(select, y.shape, motif_size, len_out)
    y = torch.where(select_nucs_b, y, 0)
               
    probs = torch.rand(shape, device=utils.DEVICE)
    masked = select & (probs < p_mask)
    random = select & (probs >= p_mask) & (probs < p_mask + p_random)

    # Convert random mask to sequence-level, replace with random nucl. if True
    random_nucs_b = motifs_to_nucs_mask(random, X.shape, motif_size, len_out)
    random_nucs = get_random_nucs(X.shape)
    X = torch.where(random_nucs_b, random_nucs, X)

    return X, masked, y


def motifs_to_nucs_mask(motif_boolean_tensor, X_shape, motif_size, len_out):
    '''Converts a motif-level Boolean tensor to sequence-level, by repeating its
    elements a motif_size amount of times.'''

    # Initialize at full length
    nucs_boolean_tensor = torch.zeros((X_shape[0], X_shape[2]),
                                      dtype=torch.bool, device=utils.DEVICE)
    
    nucs_boolean_tensor[:,:len_out] = ( # Replace until len_out
        motif_boolean_tensor[:,1:].unsqueeze(-1) # Add dimension
        .repeat(1, 1, motif_size) # Repeat in that dimension
        .view(motif_boolean_tensor.size(0), -1) # Then flatten that dimension
    )
    
    return nucs_boolean_tensor.unsqueeze(-2).repeat(1,4,1) # Repeat 4 channels


def get_random_nucs(X_shape):
    '''Returns a tensor of random 4D-DNA encoded nucleotides of `X_shape`.'''
    random_nucs = torch.zeros(X_shape, device=utils.DEVICE)
    batch_i = torch.arange(X_shape[0], device=utils.DEVICE).unsqueeze(1)
    rand_i = torch.randint(4, (X_shape[0], X_shape[2]), device=utils.DEVICE)
    bases_i = torch.arange(X_shape[2], device=utils.DEVICE).unsqueeze(0)
    random_nucs[batch_i, rand_i, bases_i] = 1
    return random_nucs


def evaluate_mmm(model, data, p_mlm, p_mask, p_random, loss_function, metrics):
    '''Evaluation function to keep track of in-training progress for MMM.'''
    
    # Initialization
    loss = 0 # Running loss
    y_true_all, y_pred_all = [], [] # All predictions/targets
    dataloader = DataLoader(data, model.pred_batch_size, shuffle=False) # Data

    # Prediction
    model.eval() # Set in evaluation mode and turn off gradient calculation
    with torch.no_grad():
        for X, _ in dataloader: # Loop through + mask data
            X, mask, y_true = mask_batch(X, model.base_arch.motif_size, p_mlm, 
                                         p_mask, p_random)                                         
            y_pred = model(X, mask) # Make a prediction
            y_true = y_true.view(-1, y_true.shape[-2]) # Flatten target
            y_pred = y_pred.view(-1, y_pred.shape[-2]) # Flatten prediction
            select = y_true.sum(dim=1) > 0 # Remove non-selected bases
            y_true, y_pred = y_true[select], y_pred[select] 
            loss += len(y_true)*loss_function(y_pred, y_true).item()
            # Save predictions & targets
            y_true_all.append(torch.argmax(y_true, axis=-1).cpu())
            y_pred_all.append(torch.argmax(y_pred, axis=-1).cpu())
    y_true_all = torch.concat(y_true_all) # Concatenate predictions & targets
    y_pred_all = torch.concat(y_pred_all)

    # Evaluation
    scores = [loss / len(y_true_all)] # Total average loss
    for metric in metrics: # Loop through metrics, append score
        metric_function = metrics[metric]
        scores.append(metric_function(y_true_all, y_pred_all))

    return scores