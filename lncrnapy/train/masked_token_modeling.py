'''Masked Language Modeling pre-training task for tokenized nucleotide 
sequences.

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
from lncrnapy.train.metrics import mtm_metrics


def train_masked_token_modeling(
        model, train_data, valid_data, epochs, n_samples_per_epoch=None, 
        batch_size=8, p_mlm=0.15, p_mask=0.8, p_random=0.1, warmup_steps=32000, 
        loss_function=None, logger=None, metrics=mtm_metrics
    ):
    '''Trains `model` for Masked Language Modeling task, using `train_data`, 
    for specified amount of `epochs`. Assumes sequence data is tokenized (see
    `lncrnapy.features.tokenizers`) and model of type `MaskedTokenModel`.
    
    Arguments
    ---------
    `model`: `torch.nn.Module` | `lncrnapy.modules.MaskedTokenModel`
        Neural network that is to be trained.
    `train_data`: `lncrnapy.data.Data`
        Data to use for training, must call `set_tensor_features` first. After 
        every training epoch, the performance of the model on a subset of the 
        training set is determined. The length of this subset is 
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
        Probability for a token to be selected for MLM (default is 0.15).
    `p_mask`: `float`
        Probability for a token to be masked when selected (default is 0.8).
    `p_random`: `float`
        Probability for a token to be randomly replaced when selected (default
        is 0.1).
    `warmup_steps`: `int`
        Number of training steps in which learning rate linearly increases. 
        After this amount of steps, the learning rate decreases proportional to
        the invserse square root of the step number (default is 32000).
    `loss_function`: `torch.nn.Module`
        Loss function that is to be optimized, assuming logits (so no Softmax) 
        and `ignore_index=utils.TOKENS['PAD']`. Uses `torch.nn.CrossEntropyLoss` 
        if None (default).
    `label_smoothing`: `float`
        How much weight should be subtracted from the target token and divided
        over the remaining tokens, for regularization (default is 0.1).
    `logger`: `lncrnapy.train.loggers`
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
        loss_function = torch.nn.CrossEntropyLoss(
                          label_smoothing=0.1, ignore_index=utils.TOKENS['PAD'])
    optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9,0.98))
    lr_scheduler = LrSchedule(optimizer, model.base_arch.d_model, warmup_steps)
    scaler = get_gradient_scaler(utils.DEVICE)
    logger = logger if logger else LoggerBase()
    logger.start(metrics)

    print("Training MLM...")
    for i in utils.progress(range(epochs)):
        model = epoch(model, train_dataloader, p_mlm, p_mask, p_random, 
                      loss_function, optimizer, scaler, lr_scheduler)
        train_results = evaluate(model, train_subset, p_mlm, p_mask, p_random, 
                                 loss_function, metrics) 
        valid_results = evaluate(model, valid_data, p_mlm, p_mask, p_random,
                                 loss_function, metrics) 
        logger.log(train_results+valid_results, model)

    # Finish
    logger.finish()
    return model, logger.history


def epoch(model, dataloader, p_mlm, p_mask, p_random, loss_function, optimizer,
          scaler, lr_scheduler):
    '''Trains `model` for a single epoch.'''
    model.train() # Set training mode
    for X, _ in dataloader: # Loop through data
        X, y = mask_batch(X,model.base_arch.vocab_size, p_mlm, p_mask, p_random)
        optimizer.zero_grad() # Zero out gradients
        with torch.autocast(**get_amp_args(utils.DEVICE)):
            y_pred = model(X) # Make prediction
            y = y.view(-1) # Flatten labels
            y_pred = y_pred.view(-1, model.base_arch.vocab_size) # Flatten 
            loss = loss_function(y_pred, y) # Calculate loss
        scaler.scale(loss).backward() # Calculate gradients
        scaler.unscale_(optimizer) # Unscale before gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), utils.CLIP_NORM)
        scaler.step(optimizer) # Optimize parameters
        lr_scheduler.step() # Update learning rate
        scaler.update() # Updates scale
    return model # Return model


def mask_batch(X, vocab_size, p_mlm, p_mask, p_random):
    '''Maks a batch of sequence data for MLM'''

    # Calculate boolean tensors using selection probabilities
    select = ((torch.rand(X.shape, device=utils.DEVICE) < p_mlm) & # Select 
              (X != utils.TOKENS['PAD']) & # Can't select...
              (X != utils.TOKENS['SEP']) & # ... special tokens
              (X != utils.TOKENS['CLS'])) 
                
    probs = torch.rand(X.shape, device=utils.DEVICE)
    masked = select & (probs < p_mask)
    random = select & (probs >= p_mask) & (probs < p_mask + p_random)

    # Replace with masks/random tokens using the selection tensors
    y = X.clone() # Create a copy as target
    X[masked] = utils.TOKENS['MASK'] # Apply mask
    X[random] = torch.randint( # Apply random tokens
        len(utils.TOKENS), # Exclude special tokens 
        vocab_size,
        (torch.sum(random).item(),), 
        dtype=torch.long,
        device=utils.DEVICE)
    # The rest for which select is True remains unchanged
    y[~select] =  utils.TOKENS['PAD'] # Pad those not selected

    return X, y


def evaluate(model, data, p_mlm, p_mask, p_random, loss_function, metrics):
    '''Evaluation function to keep track of in-training progress for MLM.'''
    
    # Initialization
    loss = 0 # Running loss
    y_true_all, y_pred_all = [], [] # All predictions/targets
    dataloader = DataLoader(data, model.pred_batch_size, shuffle=False) # Data

    # Prediction
    model.eval() # Set in evaluation mode and turn off gradient calculation
    with torch.no_grad():
        for X, _ in dataloader: # Loop through + mask data
            X, y_true = mask_batch(X, model.base_arch.vocab_size, p_mlm, p_mask,
                                   p_random)
            y_pred = model(X) # Make a prediction
            y_true = y_true.view(-1) # Flatten target
            y_pred = y_pred.view(-1, model.base_arch.vocab_size) # Flatten pred.
            select = y_true != utils.TOKENS['PAD'] # Remove non-selected tokens
            y_true, y_pred = y_true[select], y_pred[select] 
            # NOTE: We verified that evaluating loss after selection is good
            loss += len(y_true)*loss_function(y_pred, y_true).item()
            y_true_all.append(y_true.cpu()) # Save predictions & targets
            y_pred_all.append(torch.argmax(y_pred, axis=-1).cpu())
    y_true_all = torch.concat(y_true_all) # Concatenate predictions & targets
    y_pred_all = torch.concat(y_pred_all)

    # Evaluation
    scores = [loss / len(y_true_all)] # Total average loss
    for metric in metrics: # Loop through metrics, append score
        metric_function = metrics[metric]
        scores.append(metric_function(y_true_all, y_pred_all))

    return scores