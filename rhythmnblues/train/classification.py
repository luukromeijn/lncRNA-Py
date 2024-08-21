'''Functions for training a deep learning model for the classification of RNA 
transcripts as either protein-coding or long non-coding.'''

import torch
from torch.utils.data import DataLoader, RandomSampler
from rhythmnblues import utils
from rhythmnblues.train.loggers import LoggerBase
from rhythmnblues.train.mixed_precision import get_gradient_scaler, get_amp_args
from rhythmnblues.train.metrics import classification_metrics


def train_classifier(
        model, train_data, valid_data, epochs, batch_size=64, 
        loss_function=None, optimizer=None, n_samples_per_epoch=None, 
        logger=None, metrics=classification_metrics
    ):
    '''Trains `model` for classification task, using `train_data`, for specified
    amount of `epochs`.
    
    Arguments
    ---------
    `model`: `torch.nn.Module` | `rhythmnblues.modules.Classifier`
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
    `loss_function`: `torch.nn.Module`
        Loss function that is to be optimized. If None, falls back to weighted 
        Binary Cross Entropy (`torch.nn.BCEWithLogitsLoss`) (default is None). 
    `optimizer`: `torch.optim`
        Optimizer to update the network's weights during training. If None 
        (default), will use Adam with learning rate 0.0001.
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
        loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = optimizer if optimizer else torch.optim.Adam(model.parameters(), 
                                                             lr=0.0001)
    scaler = get_gradient_scaler(utils.DEVICE)
    logger = logger if logger else LoggerBase()
    logger.start(metrics)
    
    print("Training classifier...")
    for epoch in utils.progress(range(epochs)): # Looping through epochs
        model = epoch_classifier(model, train_dataloader, loss_function, 
                                 optimizer, scaler) # Train
        train_results = evaluate_classifier(model, train_subset, loss_function, 
                                            metrics) # Evaluate on trainset
        valid_results = evaluate_classifier(model, valid_data, loss_function, 
                                            metrics) # Evaluate on valid set
        logger.log(train_results+valid_results, model) # Log

    # Finish
    logger.finish()
    return model, logger.history


def epoch_classifier(model, dataloader, loss_function, optimizer, scaler):
    '''Trains `model` for a single epoch.'''
    model.train() # Set training mode
    for X, y in dataloader: # Loop through data
        optimizer.zero_grad() # Zero out gradients
        with torch.autocast(**get_amp_args(utils.DEVICE)):
            pred = model(X) # Make prediction
            loss = loss_function(pred, y) # Calculate loss
        scaler.scale(loss).backward() # Calculate gradients
        scaler.unscale_(optimizer) # Unscale before gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), utils.CLIP_NORM)
        scaler.step(optimizer) # Optimize parameters 
        scaler.update() # Updates scale
    return model # Return model


def evaluate_classifier(model, data, loss_function, 
                        metrics=classification_metrics): 
    '''Simple evaluation function to keep track of in-training progress.'''
    scores = []
    y_pred = model.predict(data, return_logits=True) # Return as logits
    target = data.df[data.y_name].values
    y_true = torch.zeros(len(target),1)
    y_true[target == 'pcrna'] = 1.0
    scores.append(loss_function(y_pred, y_true).item()) # Calculate loss
    y_pred = torch.sigmoid(y_pred).round() # Then convert to classes
    for metric in metrics: # (these metrics assume classes, not logits)
        metric_function = metrics[metric]
        scores.append(metric_function(y_true, y_pred))
    return scores