'''Contains `Data` class for containing, analyzing, and manipulating RNA 
sequence data.'''

import copy
import random 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq 
from Bio.SeqRecord import SeqRecord
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
from torch.utils.data import Dataset
import torch
from lncrnapy import utils


class Data(Dataset):
    '''Container for RNA sequence data. Contains methods for data analysis and 
    manipulation.
    
    Attributes
    ----------
    `df`: `pd.DataFrame`
        The underlying `DataFrame` object containing the data.
    `labelled`: `bool`
        Whether the data has labels or not.
    `X_name`: `list[str]`
        List of predictory feature names (columns) to be retrieved as 
        tensors when `__getitem__` is called.
    `X_dtype`: type
        Data type for X.
    `y_name`: `list[str]`
        List of target feature names (columns) to be retrieved as tensors 
        when `__getitem__` is called.
    `y_dtype`: type
        Data type for y.'''

    def __init__(self, fasta_filepath=None, hdf_filepath=None, 
                 csv_filepath=None):
        '''Initializes `Data` object based on FASTA and/or .h5/.csv file(s).
        
        Arguments
        ---------
        `fasta_filepath`: `str` | `tuple[str, str]`
            Path to FASTA file of RNA sequences or pair of paths to two FASTA 
            files containing protein- and non-coding RNAs, respectively.
        `hdf_filepath`: `str`
            When `hdf_filepath` is provided, will load features from this file, 
            retrieving the sequences from the provided FASTA files using their 
            sequence IDs.
        `csv_filepath`: `str`
            When `csv_filepath` is provided, will load features from this file, 
            retrieving the sequences from the provided FASTA files using their 
            sequence IDs.'''
        
        print("Importing data...")
        if hdf_filepath is not None and csv_filepath is not None:
            raise ValueError("Data cannot be imported from .h5 and .csv file. "
                             "Can only import tabular data from one file.")
        if hdf_filepath is not None:
            self.df = self._read_tabular(hdf_filepath, fasta_filepath, 'hdf')
        elif csv_filepath is not None:
            self.df = self._read_tabular(csv_filepath, fasta_filepath, 'csv')
        elif fasta_filepath is not None:
            self.df = self._read_fasta(fasta_filepath)
        else:
            raise ValueError("Please specify data filepath(s).")

        message = 'Imported '
        if self.labelled:
            coding, noncoding = self.num_coding_noncoding()
            message += f'{coding} protein-coding and {noncoding} non-coding '
        else:
            message += f'{len(self.df)} '
        message +=f'RNA transcripts with {len(self.all_features())} feature(s).'
        print(message)

        self.rrf = 0 # (= no random reading frame by default)
        self.len_4d_dna = 7670
    
    def __str__(self):
        return self.df.__str__()
    
    def __len__(self):
        return self.df.__len__()

    def __getitem__(self, idx):
        
        # Input (X)
        if self.X_name:
            if self.X_name[0] == '4D-DNA':
                X = self._get_4d_batch(idx)
            else:
                X = self.df.iloc[idx][self.X_name].values.astype(np.float32)
                X = torch.tensor(X, dtype=self.X_dtype, device=utils.DEVICE)
        else:
            raise AttributeError(
                'No `tensor_features` set. Please call `set_tensor_features` ' +
                'first to specify which features to retrieve as tensors, or ' +
                'use the `tensor_features` argument during initialization.'
            )
        
        # Target (y)
        if self.y_name is None:
            y = -1.0 # Placeholder
        elif self.y_name[0] == 'label':
            target = self.df.iloc[idx][self.y_name].values
            y = np.zeros_like(target, dtype=np.float32)
            y[target == 'pcRNA'] = 1.0
        else:
            y = self.df.iloc[idx][self.y_name].values.astype(np.float32)
        y = torch.tensor(y, dtype=self.y_dtype, device=utils.DEVICE)
        
        return X, y
        
    def _get_4d_batch(self, idx):
        '''Retrieves 4D-DNA encoded batch of data (rows specified by `idx`.)'''
        if type(idx) == int:
            X = self._get_4d_seq(self.df.iloc[idx]['sequence'])
            return X
        else:
            X = [self._get_4d_seq(seq) for seq in self.df.iloc[idx]['sequence']]
            return torch.stack(X, dim=0)
    
    def _get_4d_seq(self, sequence):
        '''Encodes sequence in 4D-DNA encoding, returns a list.'''
        encoding = torch.zeros((4, self.len_4d_dna), device=utils.DEVICE, 
                               dtype=self.X_dtype)
        if len(sequence) > self.len_4d_dna + self.rrf:
            rrf = len(sequence)-self.len_4d_dna
        else:
            rrf = random.randint(0,self.rrf)
        encoding[:,:min(self.len_4d_dna, len(sequence)-rrf)] = torch.stack(
            [utils.NUC_TO_4D[base] for base in # Encode
             sequence[rrf:self.len_4d_dna + rrf]], dim=1
        )
        return encoding

    def _read_tabular(self, tabular_filepath, fasta_filepath, tabular_type):
        '''Loads features from `tabular_filepath`, retrieving sequences from 
        `fasta_filepath` using their sequence IDs. `tabular_filepath` can be
        csv or hdf file.'''

        files = ([fasta_filepath] if type(fasta_filepath) == str 
                 else fasta_filepath)
        if tabular_type == 'csv':
            data = pd.read_csv(tabular_filepath)
        else: # tabular_type == 'hdf':
            data = pd.read_hdf(tabular_filepath)

        if fasta_filepath is not None:
            seq_dict = {}
            for file in files:
                seq_dict.update(SeqIO.to_dict(SeqIO.parse(file, 'fasta')))
            sequences = []
            for _, row in data.iterrows():
                sequences.append(
                    str(seq_dict[row['id']].seq)
                )
            data['sequence'] = sequences

        return data

    def _read_fasta(self, fasta_filepath):
        '''Reads RNA sequences from FASTA file or list of FASTA files specified
        by `fasta_filepath`, returning a `pd.DataFrame` object.''' 

        labelled = type(fasta_filepath) != str

        data = {'id':[], 'sequence':[]}
        if labelled: 
            files = {'pcRNA': fasta_filepath[0], 'ncRNA': fasta_filepath[1]}
            data['label'] = []
        else:
            files = {'any': fasta_filepath}

        for file in files:
            for seqrecord in SeqIO.parse(files[file], 'fasta'):
                data['id'].append(seqrecord.id)
                data['sequence'].append(str(seqrecord.seq).upper())
                if labelled:
                    data['label'].append(file)

        return pd.DataFrame(data)
    
    def _write_fasta(self, data, filepath):
        '''Writes sequences in `data` to `filepath` in FASTA format.'''
        seqs = [SeqRecord(Seq(seq),id) for id, seq in 
                zip(data['id'].values, data['sequence'].values)]
        SeqIO.write(seqs, filepath, 'fasta')

    @property
    def labelled(self):
        return self.check_columns(['label'], behaviour='bool')

    def set_tensor_features(self, X_name, X_dtype=torch.float32, y_name=None, 
                            y_dtype=torch.float32, len_4d_dna=7670):
        '''Configures `Data` object to return a tuple of tensors (X,y) whenever 
        `__getitem__` is called.
        
        Arguments
        ---------
        `X_name`: `list[str]` | str
            Predictory feature names (columns) to be retrieved as tensors when 
            `__getitem__` is called.
        `X_dtype`: type
            Data type for X (default is `torch.float32`)
        `y_name`: `list[str]` | str
            Target feature names (columns) to be retrieved as tensors when 
            `__getitem__` is called. If None (default) and the data is labelled,
            will set 'label' as target feature, with 1 indicating pcRNA and 0 
            lncRNA. 
        `y_dtype`: type
            Data type for y (default is `torch.float32`)
        `len_4d_dna`: `int`
            Max length of returned seq. when `X_name=='4D-DNA'` (default 7670)
        '''
        
        X_name = [X_name] if type(X_name) == str else X_name
        y_name = [y_name] if type(y_name) == str else y_name
        self.check_columns(X_name)
        self.X_name = X_name
        self.X_dtype = X_dtype
        if y_name is None:
            if self.labelled:
                y_name = ['label']
        else:
            self.check_columns(y_name)
        self.y_name = y_name
        self.y_dtype = y_dtype
        self.len_4d_dna = len_4d_dna

    def set_random_reading_frame(self, rrf):
        '''Sample data in a random reading frame by deleting a random number (
        within the range [0,`rrf`]) of nucleotides from the start of a sequence.
        Only works with 4D-DNA encoding.'''
        if self.X_name[0] != '4D-DNA' and rrf > 0:
            raise AttributeError("Can only set random reading frame when " + 
                                 "self.X_name=='4D-DNA'.")
        self.rrf = rrf
        
    def num_coding_noncoding(self):
        '''Returns a tuple of which the elements are the number of coding and 
        non-coding sequences in the `Data` object, respectively.'''
        self.check_columns(['label'])
        return (
            len(self.df[self.df["label"]=='pcRNA']),
            len(self.df[self.df["label"]=='ncRNA'])
        )
    
    def pos_weight(self):
        '''Ratio of non-coding/coding samples, used as weight for positive class
        in weighted loss calculation.'''
        coding, noncoding = self.num_coding_noncoding()
        return torch.tensor(noncoding/coding)

    def to_hdf(self, path_or_buf, except_columns=['sequence'], **kwargs):
        '''Write data to .h5 file.
        
        Arguments
        ---------
        path_or_buf:
            Target file or buffer.
        except_columns: list[str]:
            Column names specified here won't be exported (default is 
            ['sequence'])
        kwargs: 
            Any keyword argument accepted by `pd.DataFrame.to_hdf`'''
        
        data = self.df[[column for column in self.df.columns
                        if column not in except_columns]]
        data.to_hdf(path_or_buf, index=False, key='data', mode='w', **kwargs)

    def to_csv(self, path_or_buf, except_columns=['sequence'], **kwargs):
        '''Write data to .csv file.
        
        Arguments
        ---------
        path_or_buf:
            Target file or buffer.
        except_columns: list[str]:
            Column names specified here won't be exported (default is 
            ['sequence'])
        kwargs: 
            Any keyword argument accepted by `pd.DataFrame.to_csv`'''
        
        data = self.df[[column for column in self.df.columns
                        if column not in except_columns]]
        data.to_csv(path_or_buf, index=False, **kwargs)

    def to_fasta(self, fasta_filepath):
        '''Writes sequence data to FASTA file(s) specified by `fasta_filepath`,
        which can be a string or a list of strings (length 2) indicating the 
        filepaths for for coding and non-coding transcripts, respectively.'''

        if type(fasta_filepath) != str:
            paths = {'pcRNA': fasta_filepath[0], 'ncRNA': fasta_filepath[1]}
            for label in paths:
                data = self.df[self.df['label']==label]
                self._write_fasta(data, paths[label])
        else:
            self._write_fasta(self.df, fasta_filepath)

    def calculate_feature(self, feature_extractor):
        '''Extracts feature(s) from `Data` using `feature_extractor`.'''
        self.add_feature(feature_extractor.calculate(self), 
                         feature_extractor.name)

    def add_feature(self, feature_data, feature_names):
        '''Safely adds `feature_data` as new columns to `Data`.'''
        if type(feature_names) == list:
            new = pd.DataFrame(feature_data, columns=feature_names, 
                               index=self.df.index)
            self.df = pd.concat([self.df, new], axis=1)
        else:
            self.df[feature_names] = feature_data

    def all_features(self, except_columns=['id', 'sequence', 'label']):
        '''Returns a list of all features present in data.'''
        return [feature for feature in self.df.columns 
                if feature not in except_columns]

    def check_columns(self, columns, behaviour='error'):
        '''Raises an error when a column from `columns` does not appear in this 
        `Data` object.
        
        Arguments
        ---------
        `columns`: `list[str]`
            List of column names that are checked for present in `Data` object.
        `behaviour`: ['error'|'bool']
            Whether to raise an error in case of a missing column or whether to 
            return `False` or `True` (default is 'error').'''
        
        if type(columns) != list:
            raise TypeError("Argument 'columns' should be of type list.")

        if behaviour != 'error' and behaviour != 'bool':
            raise ValueError(behaviour)

        for column in columns:
            if column == '4D-DNA':
                continue
            elif column not in self.df.columns:
                if behaviour == 'error':
                    raise RuntimeError(f"Column '{column}' missing.")
                elif behaviour == 'bool':
                    return False
                
        return True
    
    def train_test_split(self, test_size, **kwargs):
        '''Splits data up in train and test datasets, as specified by 
        `test_size`. Accepts all keyword arguments from 
        `sklearn.model_selection.train_test_split`.'''
        df1, df2 = train_test_split(self.df, test_size=test_size, **kwargs)
        train, test = copy.deepcopy(self), copy.deepcopy(self)
        train.df, test.df = df1, df2
        return train, test
    
    def coding_noncoding_split(self):
        '''Returns two `Data` objects, the first containing all pcRNA, the
        second containing all ncRNA'''
        self.check_columns(['label'])
        pc = copy.deepcopy(self)
        pc.df = pc.df[pc.df['label']=='pcRNA']
        nc = copy.deepcopy(self)
        nc.df = nc.df[nc.df['label']=='ncRNA']
        return pc, nc
    
    def test_features(self, feature_names):
        '''Evaluates statistical significance of features specified in 
        `feature_names` using a t-test.'''
        self.check_columns(['label'] + feature_names)
        coding = self.df[self.df['label']=='pcRNA'][feature_names]
        non_coding = self.df[self.df['label']=='ncRNA'][feature_names]
        means = self.df.groupby('label')[feature_names].mean()
        pcrna_means = means.loc['pcRNA']
        ncrna_means = means.loc['ncRNA']
        ttest = ttest_ind(coding, non_coding, nan_policy='omit')
        statistics = ttest.statistic
        p_values = ttest.pvalue
        return pd.DataFrame({'mean (pcrna)': pcrna_means,
                             'mean (ncrna)': ncrna_means,
                             'test statistic': statistics,
                             'P value': p_values})
    
    def plot_feature_boxplot(self, feature_name, filepath=None, figsize=None, 
                             **kwargs):
        '''Returns a boxplot of the feature specified by `feature_name`, 
        saving the plot to `filepath` if provided.'''
        fig, ax = plt.subplots(figsize=figsize)
        group_by = 'label' if self.labelled else None
        self.df.boxplot(feature_name, ax=ax, by=group_by, **kwargs)
        fig.tight_layout()
        if filepath is not None:
            fig.savefig(filepath)
        return fig
    
    def plot_feature_violin(self, feature_name, filepath=None, figsize=None,
                            **kwargs):
        '''Returns a violin plot of the feature specified by `feature_name`, 
        saving the plot to `filepath` if provided.'''
        fig, ax = plt.subplots(figsize=figsize)
        if self.labelled:
            data = [self.df[self.df['label']==label][feature_name] 
                    for label in ['pcRNA', 'ncRNA']]
        else:
            data = self.df[feature_name]
        ax.violinplot(data, widths=0.8, **kwargs)
        ax.set_xticks(np.arange(1,3), ['pcRNA', 'ncRNA'])
        ax.set_ylabel(feature_name)
        if filepath is not None:
            fig.savefig(filepath)
        return fig
    
    def plot_feature_density(self, feature_name, filepath=None, lower=0.025,
                             upper=0.975, figsize=None, **kwargs):
        '''Returns a density plot of the feature specified by `feature_name`, 
        saving the plot to `filepath` if provided.
        
        Arguments
        ---------
        `feature_name`: `str`
            Name of the to-be-plotted feature.
        `filepath`: `str`
            If specified, will save plot to this path.
        `lower`: `float`
            Lower limit of density plot, indicated as quantile (default is 
            0.025).
        `upper`: `float`
            Upper limit of density plot, indicated as quantile (default is 
            0.975). 
        `figsize`: `tuple[int]`
            Matplotlib figure size (default is None).
        `kwargs`:
            Any keyword argument from `pd.DataFrame.plot.density`.'''
        
        fig, ax = plt.subplots(figsize=figsize)
        lower = self.df[feature_name].quantile(lower)
        upper = self.df[feature_name].quantile(upper)
        if self.labelled:
            for label in ['pcRNA', 'ncRNA']:
                data = self.df[self.df['label']==label][feature_name]
                data.plot.density(ind=np.arange(lower,upper,(upper-lower)/1000), 
                                label=label, **kwargs)
            fig.legend()
        else:
            data = self.df[feature_name]
            data.plot.density(ind=np.arange(lower,upper,(upper-lower)/1000), 
                              **kwargs)
        ax.set_xlabel(feature_name)
        fig.tight_layout()
        if filepath is not None:
            fig.savefig(filepath)

        return fig
    
    def plot_feature_scatter(self, x_feature_name, y_feature_name, 
                             c_feature_name=None, c_lower=0.025,
                             c_upper=0.975, filepath=None, axis_labels=True,
                             xlim=None, ylim=None, figsize=None):
        '''Returns a scatter plot with `x_feature_name` on the x-axis plotted
        against `y_feature_name` on the y-axis.'''

        self.check_columns([x_feature_name, y_feature_name])
        fig, ax = plt.subplots(figsize=figsize)
        if self.labelled and c_feature_name is None:
            for label in ['pcRNA', 'ncRNA']:
                ax.scatter(
                    x_feature_name, y_feature_name, s=1, alpha=0.5,
                    data=self.df[self.df['label']==label], label=label,
                    rasterized=True
                )
            fig.legend(markerscale=5)
        else:
            map = ax.scatter(x_feature_name, y_feature_name, s=1, data=self.df,
                             c=c_feature_name, rasterized=True)
            if c_feature_name is not None:
                map.set_clim(self.df[c_feature_name].quantile(c_lower), 
                             self.df[c_feature_name].quantile(c_upper))
                cb = fig.colorbar(map)
                cb.ax.set_ylabel(c_feature_name)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if axis_labels:
            ax.set_xlabel(x_feature_name)
            ax.set_ylabel(y_feature_name)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        if filepath is not None:
            fig.savefig(filepath, dpi=300)

        return fig
    
    def plot_feature_space(self, feature_names, dim_red=TSNE(), filepath=None, 
                           figsize=None):
        '''Returns a visualization of the feature space of the data, reducing
        the dimensionality to 2.'''

        fig, ax = plt.subplots(figsize=figsize)

        # Normalize data
        feature_space = self.df[feature_names].copy()
        for name in feature_names:
            feature_space[name] = (
                feature_space[name] - feature_space[name].mean() / 
                (feature_space[name].std() + 1e-7)
            )
        feature_space = feature_space.fillna(0) # Removing rows with NaN values

        # Calculate and plot dimensionality reduced space
        feature_space = reduce_dimensionality(feature_space, dim_red)
        df = self.df[["label"]].copy() if self.labelled else pd.DataFrame()
        df["Dim 1"] = feature_space[:,0]
        df["Dim 2"] = feature_space[:,1]
        if self.labelled: 
            for label in ['pcRNA', 'ncRNA']:
                ax.scatter("Dim 1", "Dim 2", label=label, s=1, alpha=0.5,
                        data=df[df["label"]==label])
            fig.legend(markerscale=5)
        else:
            ax.scatter("Dim 1", "Dim 2", s=1, data=df, alpha=0.5)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        fig.tight_layout()
        if filepath is not None:
            fig.savefig(filepath)
        
        return fig
    
    def plot_feature_correlation(self, feature_names, filepath=None, 
                                 figsize=None):
        '''Plots heatmap of absolute correlation values.'''
        correlation = self.feature_correlation(feature_names)
        fig, ax = plt.subplots(figsize=figsize)
        plot = ax.imshow(np.abs(correlation))
        s = 10 - 0.09*len(feature_names)
        ax.set_xticks(np.arange(len(feature_names)), feature_names, fontsize=s, 
                      rotation=90)
        ax.set_yticks(np.arange(len(feature_names)), feature_names, fontsize=s)
        fig.colorbar(plot)
        fig.tight_layout()
        if filepath is not None:
            fig.savefig(filepath)
        return fig
    
    def feature_correlation(self, feature_names):
        '''Calculates the correlation between features in `feature_names`.'''
        return self.df[feature_names].corr()[feature_names]
    
    def filter_outliers(self, feature_name, tolerance):
        '''Removes data entries for which the value of `feature_name` falls
        outside of the tolerated range.
        
        Arguments
        ---------
        `feature_name`: `str`
            Name of the feature that should be considered.
        `tolerance`: `float`|`int`|`list`|`range`
            When numeric, refers to the tolerated amount of standard deviation
            from the mean. When a list or range, refers to the exact tolerated
            lower (inclusive) and upper (exclusive) bound.'''

        if type(tolerance) != int:
            self.df = self.df[
                (self.df[feature_name] >= tolerance[0]) &
                (self.df[feature_name] < tolerance[-1])
            ]
        else:
            std = self.df[feature_name].std()
            avg = self.df[feature_name].mean()
            self.df = self.df[
                (np.abs(self.df[feature_name] - avg) < tolerance*std)
            ]

    def filter_sequence_quality(self, tolerance):
        '''Removes data entries for which the percentage of uncertain bases
        (non-ACGT) exceeds a `tolerance` fraction.'''
        certain = self.df['sequence'].str.count("A|C|T|G")
        length = self.df['sequence'].str.len()
        ratio = (length - certain) / length
        self.df = self.df[(ratio <= tolerance)]

    def sample(self, pc=None, nc=None, N=None, replace=False, 
               random_state=None):
        '''Returns a randomly sampled `Data` object.
        
        Arguments
        ---------
        `pc`: `int`|`float`|`None`
            Number/fraction of protein-coding sequences in resulting dataset.
        `nc`: `int`|`float`|`None`
            Number/fraction of protein-coding sequences in resulting dataset.
        `N`: `int`|`None`
            Total number of sequences in resulting dataset. When specified 
            together with `pc` and `nc`, will consider the latter two as 
            fractions.
        `replace`: `bool`
            Whether or not to sample with replacement. Required if `N` or 
            `pc+nc` exceeds the number of samples in the `Data` object.
        `random_state`: `int`
            Seed for random number generator.'''

        df = copy.deepcopy(self) # Create object copy
        if pc is not None and nc is not None:
            if not self.labelled: # Data must be labelled
                raise ValueError("Can't use pc and nc for unlabelled data. " +
                                 "Please use N.")
            pcrna = self.df[self.df['label']=='pcRNA']
            ncrna = self.df[self.df['label']=='ncRNA']
            if N is not None: # N is also specified? -> use nc and pc as ratio
                _pc = (pc/(pc+nc))*N
                nc = (nc/(pc+nc))*N
                pc = _pc
            df.df = pd.concat((
                pcrna.sample(n=round(pc),replace=replace,
                             random_state=random_state),
                ncrna.sample(n=round(nc),replace=replace,
                             random_state=random_state)
            ))
        elif N is not None: # Only N? Sample N samples directly.
            df.df = self.df.sample(n=N, replace=replace, 
                                   random_state=random_state)
        else:
            raise ValueError("Please specify N and/or pc and nc.")
        return df
    

def merge_fasta(in_filepaths, out_filepath):
    '''Merges FASTA files in `in_filepaths` into a single `out_filepath`.'''
    merged = []
    for path in in_filepaths:
        merged += [seq for seq in SeqIO.parse(path, 'fasta')]
    SeqIO.write(merged, out_filepath, 'fasta')


# Some RefSeq-related functions
def get_rna_type_refseq(fasta_header):
    '''Extract the RNA type from an input FASTA header line'''
    return fasta_header.split(",")[-1].strip()

def plot_refseq_labels(fasta_filepath, filepath=None, figsize=None):
    '''Plots the distribution of RNA labels of a FASTA file that follows the 
    RefSeq format, optionally saving the figure to `filepath`.'''

    # Calculating frequencies
    rna_types = dict()
    for seq in SeqIO.parse(fasta_filepath, 'fasta'):
        rna_type = get_rna_type_refseq(seq.description)
        if rna_types.get(rna_type) is not None:
            rna_types[rna_type] += 1
        else:
            rna_types[rna_type] = 1

    values = np.array(list(rna_types.values()))
    sorted = np.argsort(values)[::-1]
    values = values[sorted]
    rna_types = [list(rna_types.keys())[i][:20] + "..." 
                 if len(list(rna_types.keys())[i]) > 15 
                 else list(rna_types.keys())[i] 
                 for i in sorted]

    # Creatin pie chart
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(len(rna_types)), values)
    ax.set_xticks(np.arange(len(rna_types)), rna_types, rotation=90)
    ax.set_yscale('log')
    ax.set_xlabel('RNA type')
    ax.set_ylabel('log(#)')
    fig.tight_layout()

    if filepath is not None:
        fig.savefig(filepath)
    return fig

def split_refseq(in_filepath, pc_filepath, nc_filepath, pc_types=['mRNA'], 
                 nc_types=['long non-coding RNA']):
    '''Splits RefSeq FASTA file into two files, coding and noncoding.'''

    pcrna = []
    ncrna = []
    for seq in SeqIO.parse(in_filepath, 'fasta'):
        rna_type = get_rna_type_refseq(seq.description)
        if rna_type in pc_types:
            pcrna.append(seq)
        elif rna_type in nc_types:
            ncrna.append(seq)
    
    SeqIO.write(pcrna, pc_filepath, 'fasta')
    SeqIO.write(ncrna, nc_filepath, 'fasta')


def plot_cross_dataset_violins(data_objects, data_names, feature_name, 
                               filepath=None, upper=0.975, lower=0.025, 
                               figsize=None, **kwargs):
    '''Creates violin plots for multiple `data_objects` for a given 
    `feature_name`. This allows to compare datasets.'''
    
    labels = []
    stats = []
    fig, ax = plt.subplots(figsize=figsize)
    for i, label in enumerate(['pcRNA', 'ncRNA']): # Loop through labels

        # Filter for given label
        data = [data.df[data.df['label']==label][feature_name] 
                for data in data_objects]
        
        # Remove outliers
        upper_bounds = [df.quantile(upper) for df in data]
        lower_bounds = [df.quantile(lower) for df in data]
        data = [df[df < bound] for df, bound in zip(data, upper_bounds)] 
        data = [df[df > bound] for df, bound in zip(data, lower_bounds)]

        # Calculate mean and std
        stats.append([label, 'avg']  +[df.mean() for df in data])
        stats.append([label, 'med']  +[df.median() for df in data])
        stats.append([label, 'std'] + [df.std() for df in data])

        # Plot only non-emtpy datasets
        new_data, pos = [], []
        for j, df in enumerate(data):
            if len(df) > 0:
                pos.append(i + (2.5*j)) 
                new_data.append(df)

        # Plot & add label to legend
        violin = ax.violinplot(new_data, pos, widths=0.8, **kwargs)
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    # Set figure
    ax.set_xticks(np.arange(0.5,len(data_objects)*2.5, 2.5), data_names)
    ax.set_ylabel(feature_name)
    plt.legend(*zip(*labels))
    fig.tight_layout()

    # Report mean and std
    print(pd.DataFrame(stats, columns=['Label', 'Stat'] + data_names))

    if filepath is not None:
        fig.savefig(filepath)
    return fig


def reduce_dimensionality(data, dim_red=TSNE()):
    '''Reduces the dimensionality of `data` using `dim_red`.'''
    if len(data.shape) != 2:
        raise ValueError("Input `data` should have 2 dimensions.")
    if type(dim_red) == TSNE and data.shape[1] > 50:
        data = PCA().fit_transform(data)[:,:50]
    data = dim_red.fit_transform(data)
    return data


def get_gencode_gene_names(data):
    '''Returns list of GENCODE gene names extracted from "id" column.'''
    get_gene_id = lambda id: id.split('|')[1].split('.')[0]
    return data.df['id'].apply(get_gene_id).tolist()