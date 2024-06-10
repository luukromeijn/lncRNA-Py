'''Logger objects used by training functions. Loggers should inherit from 
`LoggerBase` and contain the following methods:
* `set_columns`: Specifies which columns will be logged.
* `log`: Called every epoch, logs new results.'''

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


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
        self.columns = [f'{metric}|{t_or_v}' for t_or_v in ['train', 'valid'] 
                        for metric in ['Loss'] + list(metrics.keys())]
        self.t0 = time.time()

    def finish(self):
        '''Finishes logging, reports training time and final performance.'''
        self.t1 = time.time()
        print(f"Training finished in {round(self.t1-self.t0, 2)} seconds.")
        print("Final performance:")
        print(self.history.iloc[-1])
        
    def log_history(self, epoch_results):
        '''Adds row to history DataFrame.'''
        epoch_results = pd.DataFrame([epoch_results], columns=self.columns)
        self.history = pd.concat([self.history, epoch_results], 
                                 ignore_index=True)
    
    def log(self, epoch_results):
        '''Logs `epoch_results`.'''
        self.log_history(epoch_results)


class LoggerPrint(LoggerBase):
    '''Prints the results per epoch.
    
    Attributes
    ----------
    `epoch`: `int`
        Epoch counter.
    `metric_names`: `list[str]`
        Indicates which names in `columns` to print per epoch.
    `history`: `pd.DataFrame`
        Contains all logged data throughout training. 
    `columns`: `list[str]`
        Column names of the values that are received every log.''' 

    def __init__(self, metric_names=None):
        '''Initializes `LoggerPrint` object for specified `metric_names`.'''
        super().__init__()
        self.epoch = 0
        self.metric_names = metric_names

    def log(self, epoch_results):
        '''Logs `epoch_results`.'''
        
        self.log_history(epoch_results)
        self.epoch += 1

        print(f'Epoch {self.epoch}:')
        last_epoch = self.history.iloc[-1]
        if self.metric_names:
            print(last_epoch[self.metric_names])
        else:
            print(last_epoch)


class LoggerWrite(LoggerBase):
    '''Writes the results to a file at every epoch.
    
    Attributes
    ----------
    `filepath`: `str`
        Path to new .csv file to which to write the results to.
    `history`: `pd.DataFrame`
        Contains all logged data throughout training. 
    `columns`: `list[str]`
        Column names of the values that are received every log.''' 

    def __init__(self, filepath):
        '''Initializes `LoggerWrite` object for given `filepath`.'''
        super().__init__()
        self.filepath = filepath

    def log(self, epoch_results):
        '''Logs `epoch_results`.'''
        self.log_history(epoch_results)
        self.history.to_csv(self.filepath, float_format="%.6f")


class LoggerPlot(LoggerBase):
    '''Plots and saves the results as figures at every epoch.
    
    Attributes
    ----------
    `dir_path`: `str`
        Path to new/existing directory in which figures will be stored.
    `history`: `pd.DataFrame`
        Contains all logged data throughout training. 
    `columns`: `list[str]`
        Column names of the values that are received every log.''' 

    def __init__(self, dir_path):
        '''Initializes `LoggerPlot` object.'''
        super().__init__()
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        self.dir_path = dir_path

    def log(self, epoch_results):
        '''Logs `epoch_results`.'''
        self.log_history(epoch_results)
        for m in self.history.columns[:int(len(self.history.columns)/2)]:
            m = m.split('|')[0]
            fig = self.plot_history(m, 
                                    filepath=f'{self.dir_path}/{m.lower()}.png')
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
