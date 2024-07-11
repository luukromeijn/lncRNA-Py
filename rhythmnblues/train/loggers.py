'''Logger objects used by training functions. Loggers should inherit from 
`LoggerBase` and contain the following methods:
* `set_columns`: Specifies which columns will be logged.
* `log`: Called every epoch, logs new results.'''

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from rhythmnblues import utils


class LoggerBase:
    '''Base logger that adds a new row to its `history` DataFrame with every 
    log.
    
    Attributes
    ----------
    `history`: `pd.DataFrame`
        Contains all logged data throughout training. 
    `columns`: `list[str]`
        Column names of the values that are received every log.''' 

    def __init__(self):
        '''Initializes `LoggerBase object.`'''
        self.history = pd.DataFrame()
        self.columns = None

    def start(self, metrics):
        '''This method should be called right before the training loop starts. 
        It sets columns according to the specified `metrics`, and starts the
        timrer. The class assumes the loss function as first logged value and 
        train/validation results).'''
        self.columns = self._get_metric_columns(['Loss'] + list(metrics.keys()))
        self.t0 = time.time()

    def finish(self):
        '''Finishes logging, reports training time and final performance.'''
        self.t1 = time.time()
        print(f"Training finished in {round(self.t1-self.t0, 2)} seconds.")
        print("Final performance:")
        print(self.history.iloc[-1])
    
    def log(self, epoch_results, model):
        '''Logs `epoch_results` for given `model`.'''
        self._add_to_history(epoch_results)
        self._action(model)

    def _add_to_history(self, epoch_results):
        '''Adds row to history DataFrame.'''
        epoch_results = pd.DataFrame([epoch_results], columns=self.columns)
        self.history = pd.concat([self.history, epoch_results], 
                                 ignore_index=True)

    def _action(self, model):
        '''Action to perform at every call of `.log`. Assumes the latest epoch
        has been added to the `.history` attribute.'''
        pass # LoggerBase has no action

    def _get_metric_columns(self, metric_names):
        '''Adds train/valid to `metric_names` (list) to get column names.'''
        return [f'{metric}|{t_or_v}' for t_or_v in ['train', 'valid'] 
                for metric in metric_names]


class LoggerPrint(LoggerBase):
    '''Prints the results per epoch.
    
    Attributes
    ----------
    `epoch`: `int`
        Epoch counter.
    `metric_names`: `list[str]`
        Indicates which metrics to print per epoch.
    `history`: `pd.DataFrame`
        Contains all logged data throughout training. 
    `columns`: `list[str]`
        Column names of the values that are received every log.''' 

    def __init__(self, metric_names=None):
        '''Initializes `LoggerPrint` object for specified `metric_names`.'''
        super().__init__()
        self.epoch = 0
        self.metric_names = (None if metric_names is None 
                                  else self._get_metric_columns(metric_names))

    def _action(self, model):
        self.epoch += 1

        last_epoch = self.history.iloc[[-1]]
        if self.metric_names is not None:
            row = last_epoch[self.metric_names]
        else:
            row = last_epoch

        row = row.__str__()
        header_eol_idx = row.find('\n')
        if self.epoch == 1:
            print(row[:header_eol_idx])
        print(row[header_eol_idx+1:])

class LoggerWrite(LoggerBase):
    '''Writes the results to a file at every epoch.
    
    Attributes
    ----------
    `filepath`: `str`
        Path to new .csv file to which to write the results to.
    `metric_names`: `list[str]`
        Indicates which metrics to write per epoch.
    `history`: `pd.DataFrame`
        Contains all logged data throughout training. 
    `columns`: `list[str]`
        Column names of the values that are received every log.''' 

    def __init__(self, filepath, metric_names=None):
        '''Initializes `LoggerWrite` object for given `filepath`.'''
        super().__init__()
        self.filepath = filepath
        self.metric_names = (None if metric_names is None 
                                  else self._get_metric_columns(metric_names))

    def _action(self, model):
        history = self.history
        if self.metric_names is not None:
            history = history[self.metric_names]
        history.to_csv(self.filepath, float_format="%.6f")


class LoggerPlot(LoggerBase):
    '''Plots and saves the results as figures at every epoch.
    
    Attributes
    ----------
    `dir_path`: `str`
        Path to new/existing directory in which figures will be stored.
    `metric_names`: `list[str]`
        Indicates which metrics to plot per epoch.
    `history`: `pd.DataFrame`
        Contains all logged data throughout training. 
    `columns`: `list[str]`
        Column names of the values that are received every log.''' 

    def __init__(self, dir_path, metric_names=None):
        '''Initializes `LoggerPlot` object.'''
        super().__init__()
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        self.dir_path = dir_path
        self.metric_names = metric_names # We don't add '|train'/'|valid' here

    def start(self, metrics):
        super().start(metrics)
        if self.metric_names is None:
            self.metric_names = ['Loss'] + list(metrics.keys())

    def _action(self, model):
        '''Logs `epoch_results`.'''
        for metric in self.metric_names:
            fig = self.plot_history(
                metric, filepath=f'{self.dir_path}/{metric.lower()}.png'
            )
            plt.close(fig)

    def plot_history(self, metric_name, filepath=None, figsize=None):
        '''Plots the history data for a given `metric_name`, saving the 
        resulting figure to a (optionally) specified `filepath`.'''
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(np.arange(1, len(self.history)+1), 
                self.history[f'{metric_name}|train'], label='Training')
        ax.plot(np.arange(1, len(self.history)+1), 
                self.history[f'{metric_name}|valid'], label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        fig.legend()
        fig.tight_layout()
        if filepath is not None:
            fig.savefig(filepath)
        return fig


class LoggerList(LoggerBase):
    '''Combine multiple loggers into one, executing all of their actions per 
    logging event while keeping track of a single, shared history.
    
    Attributes
    ----------
    `loggers`: `list`
        List of loggers from rhythmnblues.train.loggers.'''

    def __init__(self, *args):
        '''Initializes `LoggerList` object with specified loggers.'''
        super().__init__()
        self.loggers = [logger for logger in args]

    def start(self, metrics):
        super().start(metrics)
        for logger in self.loggers:
            logger.start(metrics)

    def log(self, epoch_results, model):
        self._add_to_history(epoch_results)
        for logger in self.loggers:
            logger.history = self.history
            logger._action(model)


# TODO add unittest
class EarlyStopping(LoggerBase):
    '''Special logger that saves the model if it has the best-so-far 
    performance.
    
    Attributes
    ----------
    `metric_name`: `str`
        Column name of the metric that the early stopping is based on.
    `filepath`: `str`
        Path to and name of file where model should be saved to.
    `sign`: `int`
        Whether the goal is max-/minimization (1/-1).
    `best_score`: `float`
        Current best score of metric.
    `epoch`: `int`
        Epoch counter.'''

    def __init__(self, metric_name, filepath, maximize=True):
        '''Initializes `EarlyStopping` object.'''
        super().__init__()
        self.metric_name = metric_name
        self.filepath = filepath
        self.sign = 1 if maximize else -1
        self.best_score = self.sign * -np.inf
        self.epoch = 0

    def _action(self, model):
        self.epoch += 1
        if (self.sign * self.history.iloc[-1][self.metric_name] > 
            self.sign * self.best_score):
            torch.save(model, self.filepath)
            print(f"Model saved at epoch {self.epoch}.")


# TODO add unittest
class LoggerMLMCounts(LoggerBase):
    '''Plots, at every epoch, the true token count vs the predicted token
    counts, based on the 'Counts' metric.'''

    def __init__(self, vocab_size, filepath, valid=True):
        super().__init__()
        self.filepath = filepath
        self.vocab_size = vocab_size
        self.train_or_valid = 'valid' if valid else 'train'

    def _action(self, model):
        
        # Format counts from history data into an array
        counts = np.zeros((2, self.vocab_size-len(utils.TOKENS)))
        data = self.history.iloc[-1][f"Counts|{self.train_or_valid}"]
        for i, token_counts in enumerate(data):
            for token, count in zip(token_counts[0], token_counts[1]):
                if token < len(utils.TOKENS):
                    continue
                else:
                    counts[i, token-len(utils.TOKENS)] = count
        
        # Sort based on true count
        order = np.argsort(counts[0])[::-1]

        # Plot
        fig, ax = plt.subplots()
        bar1 = ax.bar(np.arange(self.vocab_size-len(utils.TOKENS)), 
                     counts[0][order], width=1, alpha=0.5, label='Target')
        bar2 = ax.bar(np.arange(self.vocab_size-len(utils.TOKENS)), 
                      counts[1][order], width=1, alpha=1.0, label='Predicted',
                      color=bar1[0].get_facecolor(), )
        ax.set_yscale('log')
        ax.set_xlabel('Tokens')
        ax.set_ylabel('log(count)')
        fig.legend()
        fig.tight_layout()
        fig.savefig(self.filepath)
        plt.close(fig)


class LoggerDistribution(LoggerBase):

    def __init__(self, filepath, apply_to=None):
        super().__init__()
        self.filepath = filepath
        self.sufx = f' ({apply_to})' if apply_to else ''

    def _action(self, model):

        # Preparing the data
        medians = [[] for j in range(2)]
        ranges = [[[] for i in range(6)] for j in range(2)]
        for _, row in self.history.iterrows():
            for j, t_and_p in enumerate(row[f'Distribution{self.sufx}|valid']):
                for i, perc in enumerate(t_and_p):
                    if i == 3:
                        medians[j].append(perc)
                    elif i < 3:
                        ranges[j][i].append(perc)
                    elif i > 3:
                        ranges[j][i-1].append(perc)
        
        # Plotting
        fig, ax = plt.subplots()
        for j in range(2):
            line = ax.plot(np.arange(1, len(self.history)+1), medians[j])
            color = line[0].get_color()
            for i in range(3):
                ax.fill_between(np.arange(1, len(self.history)+1), ranges[j][i],
                                ranges[j][5-i], alpha=0.15, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Output')
        
        fig.tight_layout()
        fig.savefig(self.filepath)