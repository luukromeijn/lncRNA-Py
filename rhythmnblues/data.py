'''Contains `Data` class for containing, analyzing, and manipulating RNA seq
data.'''

import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq 
from Bio.SeqRecord import SeqRecord
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind


class Data:
    '''Container for RNA seq data. Contains methods for data analysis and 
    manipulation.
    
    Attributes
    ----------
    `df`: `pd.DataFrame`
        The underlying `DataFrame` object containing the data.'''

    def __init__(self, pc_filepath=None, nc_filepath=None, hdf_filepath=None):
        '''Initializes `Data` object based on two FASTA files for 
        protein-coding and non-coding RNA provided in `pc_filepath` and 
        `nc_filepath`, respectively.
        
        Arguments
        ---------
        `pc_filepath`: `str`
            Path to FASTA file of protein-coding sequences.
        `nc_filepath`: `str`
            Path to FASTA file of non-coding sequences.
        `hdf_filepath`: `str`
            When `hdf_filepath` is provided, will load features from this file, 
            retrieving the sequences from the provided FASTA files using their 
            sequence IDs.'''
        
        print("Importing data...")
        if hdf_filepath is not None:
            self.df = self._read_hdf(hdf_filepath, pc_filepath, nc_filepath)
            print('done')
        elif pc_filepath is not None and nc_filepath is not None:
            self.df = self._read_fasta(pc_filepath, nc_filepath)
        else:
            raise ValueError("Please specify data filepaths.")

        coding, noncoding = self.num_coding_noncoding()
        print(
            f'Imported {coding} protein-coding and {noncoding} non-coding ' +
            f'RNA transcripts with {len(self.all_features())} feature(s).'
        )
    
    def __str__(self):
        return self.df.__str__()
    
    def __len__(self):
        return self.df.__len__()

    def _read_hdf(self, hdf_filepath, pc_filepath, nc_filepath):
        '''Loads features from `hdf_filepath`, retrieving sequences from 
        `pc_filepath` and `nc_filepath` using their sequence IDs.'''

        files = {'pcrna': pc_filepath, 'ncrna': nc_filepath}
        data = pd.read_hdf(hdf_filepath)

        if pc_filepath is not None and nc_filepath is not None:
            seq_dicts = {}
            for file in files:
                seq_dicts[file] = SeqIO.to_dict(
                    SeqIO.parse(files[file], 'fasta')
                )
            sequences = []
            for _, row in data.iterrows():
                sequences.append(
                    str(seq_dicts[row['label']][row['id']].seq)
                )
            data['sequence'] = sequences

        return data

    def _read_fasta(self, pc_filepath, nc_filepath):
        '''Reads protein-coding and non-coding sequences from FASTA files
        specified by `pc_filepath` and `nc_filepath`, respectively, returning
        a `pd.DataFrame` object.'''

        files = {'pcrna': pc_filepath, 'ncrna': nc_filepath}
        data = {'id':[], 'sequence':[], 'label':[]}
        for file in files:
            for seqrecord in SeqIO.parse(files[file], 'fasta'):
                data['id'].append(seqrecord.id)
                data['sequence'].append(str(seqrecord.seq))
                data['label'].append(file)

        return pd.DataFrame(data)
    
    def _write_fasta(self, data, filepath):
        '''Writes sequences in `data` to `filepath` in FASTA format.'''
        seqs = [SeqRecord(Seq(seq),id) for id, seq in 
                zip(data['id'].values, data['sequence'].values)]
        SeqIO.write(seqs, filepath, 'fasta')
    
    def num_coding_noncoding(self):
        '''Returns a tuple of which the elements are the number of coding and 
        non-coding sequences in the `Data` object, respectively.'''
        return (
            len(self.df[self.df["label"]=='pcrna']),
            len(self.df[self.df["label"]=='ncrna'])
        )

    
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
            Any keyword argument accepted by `pd.DataFrame.to_csv`'''
        
        data = self.df[[column for column in self.df.columns
                        if column not in except_columns]]
        data.to_hdf(path_or_buf, index=False, key='data', mode='w', **kwargs)

    def to_fasta(self, pc_filepath=None, nc_filepath=None, filepath=None):
        '''Writes sequence data to FASTA files(s). Either writes to two separate
        files for coding and non-coding transcripts to `pc_filepath` and `
        `nc_filepath`, respectively, or writes to a single `filepath`.'''

        if filepath is None:
            paths = {'pcrna': pc_filepath, 'ncrna': nc_filepath}
            for label in paths:
                data = self.df[self.df['label']==label]
                self._write_fasta(data, paths[label])
        else:
            self._write_fasta(self.df, filepath)

    def calculate_feature(self, feature_extractor):
        '''Adds feature(s) from `feature_extractor` as column(s) to `Data`.'''
        self.check_columns(['sequence'])
        self.df[feature_extractor.name] = feature_extractor.calculate(self)

    def all_features(self):
        '''Returns a list of all features present in data.'''
        return [feature for feature in self.df.columns 
                if feature not in ['id', 'sequence', 'label']]

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
        
        if behaviour != 'error' and behaviour != 'bool':
            raise ValueError(behaviour)

        for column in columns:
            if column not in self.df.columns:
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
        pc = copy.deepcopy(self)
        pc.df = pc.df[pc.df['label']=='pcrna']
        nc = copy.deepcopy(self)
        nc.df = nc.df[nc.df['label']=='ncrna']
        return pc, nc
    
    def test_features(self, feature_names):
        '''Evaluates statistical significance of features specified in 
        `feature_names` using a t-test.'''
        coding = self.df[self.df['label']=='pcrna'][feature_names]
        non_coding = self.df[self.df['label']=='ncrna'][feature_names]
        means = self.df.groupby('label')[feature_names].mean()
        pcrna_means = means.loc['pcrna']
        ncrna_means = means.loc['ncrna']
        ttest = ttest_ind(coding, non_coding, nan_policy='omit')
        statistics = ttest.statistic
        p_values = ttest.pvalue
        return pd.DataFrame({'mean (pcrna)': pcrna_means,
                             'mean (ncrna)': ncrna_means,
                             'test statistic': statistics,
                             'P value': p_values}).T
    
    def plot_feature_boxplot(self, feature_name, filepath=None, **kwargs):
        '''Returns a boxplot of the feature specified by `feature_name`, 
        saving the plot to `filepath` if provided.'''
        fig, ax = plt.subplots()
        self.df.boxplot(feature_name, ax=ax, by='label', **kwargs)
        fig.tight_layout()
        if filepath is not None:
            fig.savefig(filepath)
        return fig
    
    def plot_feature_density(self, feature_name, filepath=None, lower=0.025,
                             upper=0.975, **kwargs):
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
        `kwargs`:
            Any keyword argument from `pd.DataFrame.plot.density`.'''
        
        fig, ax = plt.subplots()
        lower = self.df[feature_name].quantile(lower)
        upper = self.df[feature_name].quantile(upper)
        for label in ['pcrna', 'ncrna']:
            data = self.df[self.df['label']==label][feature_name]
            data.plot.density(ind=np.arange(lower,upper,(upper-lower)/1000), 
                              label=label, **kwargs)
        ax.set_xlabel(feature_name)
        fig.legend()
        fig.tight_layout()
        if filepath is not None:
            fig.savefig(filepath)

        return fig
    
    def plot_feature_scatter(self, x_feature_name, y_feature_name, 
                             filepath=None):
        '''Returns a scatter plot with `x_feature_name` on the x-axis plotted
        against `y_feature_name` on the y-axis.'''

        self.check_columns([x_feature_name, y_feature_name])
        fig, ax = plt.subplots()
        for label in ['pcrna', 'ncrna']:
            ax.scatter(x_feature_name, y_feature_name, 
                        data=self.df[self.df['label']==label], label=label)
        ax.set_xlabel(x_feature_name)
        ax.set_ylabel(y_feature_name)
        fig.legend()
        fig.tight_layout()
        if filepath is not None:
            fig.savefig(filepath)

        return fig
    
    def plot_feature_space(self, feature_names, filepath=None, dim_red=TSNE()):
        '''Returns a visualization of the feature space of the data, reducing
        the dimensionality to 2.'''

        fig, ax = plt.subplots()

        # Normalize data
        feature_space = self.df[feature_names].copy()
        for name in feature_names:
            feature_space[name] = (
                feature_space[name] - feature_space[name].mean() / 
                feature_space[name].std() + 1e-7
            )

        # Calculate and plot dimensionality reduced space
        if type(dim_red) == TSNE and len(feature_names) > 50:
            feature_space = PCA().fit_transform(feature_space)[:,:50]
        feature_space = dim_red.fit_transform(feature_space)
        df = self.df[["label"]].copy()
        df["Dim 1"] = feature_space[:,0]
        df["Dim 2"] = feature_space[:,1]
        for label in ["pcrna", "ncrna"]:
            ax.scatter("Dim 1", "Dim 2", label=label, s=1,
                       data=df[df["label"]==label])
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        fig.legend(markerscale=5)
        fig.tight_layout()
        if filepath is not None:
            fig.savefig(filepath)
        
        return fig
    
    def plot_feature_correlation(self, feature_names, filepath=None):
        '''Plots heatmap of absolute correlation values.'''
        correlation = self.feature_correlation(feature_names)
        fig, ax = plt.subplots()
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

    def sample(self, pc=None, nc=None, N=None, replace=False, seed=None):
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
        `seed`: `int`
            Seed for random number generator.'''

        pcrna = self.df[self.df['label']=='pcrna']
        ncrna = self.df[self.df['label']=='ncrna']

        if pc is not None and nc is not None:
            if N is not None: # Use pc/nc as target ratio
                _pc = (pc/(pc+nc))*N
                nc = (nc/(pc+nc))*N
                pc = _pc
        elif N is not None:
            pc = len(pcrna)/(len(self.df))*N
            nc = len(ncrna)/(len(self.df))*N
        else:
            raise ValueError("Please specify N and/or pc and nc.")

        df = copy.deepcopy(self)
        df.df = pd.concat((
            pcrna.sample(n=int(np.round(pc)), replace=replace),
            ncrna.sample(n=int(np.round(nc)), replace=replace)
        ))
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

def plot_refseq_labels(fasta_filepath, filepath=None):
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
    fig, ax = plt.subplots()
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
        