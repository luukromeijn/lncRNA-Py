'''Functions for training a deep learning model for the classification of RNA 
transcripts as either protein-coding or long non-coding.'''

import time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score
from rhythmnblues import utils
from rhythmnblues.train.loggers import LoggerBase
from rhythmnblues.train.mixed_precision import get_gradient_scaler, get_amp_args


METRICS = {
    'Accuracy': accuracy_score,
    'Precision (pcrna)': lambda y_t, y_p: precision_score(y_t,y_p,pos_label=1),
    'Recall (pcrna)': lambda y_t, y_p: recall_score(y_t,y_p,pos_label=1),
    'Precision (ncrna)': lambda y_t, y_p: precision_score(y_t,y_p,pos_label=0),
    'Recall (ncrna)': lambda y_t, y_p: recall_score(y_t,y_p,pos_label=0),
}


def train_classifier(
        model, train_data, valid_data, epochs, batch_size=64, 
        loss_function=None, optimizer=None, logger=None, metrics=METRICS
    ):
    '''Trains `model` with `train_data` for specified amount of `epochs`.
    
    Arguments
    ---------
    `model`: `torch.nn.Module` | `rhythmnblues.modules.Model`
        Neural network that is to be trained.
    `train_data`: `rhythmnblues.data.Data`
        Data to use for training, must call `set_tensor_features` first. 
    `valid_data`: `rhythmnblues.data.Data`
        Data to use for validation, must call `set_tensor_features` first.
    `epochs`: `int`
        How many epochs (data run-throughs) to train for.
    `batch_size`: `int`
        Number of examples per batch (default is 64).
    `loss_function`: `torch.nn.Module`
        Loss function that is to be optimized. If None, falls back to Binary 
        Cross Entropy (`torch.nn.BCEWithLogitsLoss`) (default is None). 
    `optimizer`: `torch.optim`
        Optimizer to update the network's weights during training. If None 
        (default), will use Adam with learning rate 0.0001.
    `logger`: `rhythmnblues.train.loggers`
    	Logger object whose `log` method will be called at every epoch. If None
        (default), will use LoggerBase, which only keeps track of the history.
    `metrics`: `dict[str:callable]`
        Metrics (name + function) that will be evaluated at every epoch.'''

    # Initializing required objects
    train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
    loss_function = (loss_function if loss_function 
                                   else torch.nn.BCEWithLogitsLoss())
    optimizer = optimizer if optimizer else torch.optim.Adam(model.parameters(), 
                                                             lr=0.0001)
    scaler = get_gradient_scaler(utils.DEVICE)
    logger = logger if logger else LoggerBase()
    logger.set_columns(metrics)

    t0 = time.time()
    print("Training...")
    for epoch in utils.progress(range(epochs)): # Looping through epochs
        model = epoch_classifier(model, train_dataloader, loss_function, 
                                 optimizer, scaler) # Train
        train_results = evaluate_classifier(model, train_data, loss_function, 
                                            metrics) # Evaluate on trainset
        valid_results = evaluate_classifier(model, valid_data, loss_function, 
                                            metrics) # Evaluate on valid set
        logger.log(train_results + valid_results) # Log

    # Finish
    print(f"Training finished in {round(time.time()-t0, 2)} seconds.")
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


def evaluate_classifier(model, data, loss_function, metrics=METRICS): 
    '''Simple evaluation function to keep track of in-training progress.'''
    scores = []
    pred = model.predict(data, return_logits=True) # Return as logits 
    scores.append(loss_function(pred, data[:][1].cpu()).item()) # Calculate loss
    pred = torch.sigmoid(pred).round() # Then convert to classes
    for metric in metrics: # (these metrics assume classes, not logits)
        metric_function = metrics[metric]
        scores.append(metric_function(data[:][1].cpu(), pred))
    return scores