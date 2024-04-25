'''Contains feature extractor classes that can calculate several features of RNA
sequences, such as Most-Like Coding Sequences and nucleotide frequencies. 

Every feature extractor class contains:
* A `name` attribute of type `str`, indicating what name a `Data` column for
this feature will have.
* A `calculate` method with a `Data` object as argument, returning a list or
array of the same length as the `Data` object.'''

import itertools
import os
import re
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.lcc import lcc_simp
from scipy.stats import entropy
from rhythmnblues import utils


class Length:
    '''Calculates lengths of sequences.
    
    Attributes
    ----------
    `name`: 'length'
        Column name for sequence length.
    '''

    def __init__(self):
        self.name = 'length'

    def calculate(self, data):
        return data.df['sequence'].str.len()


class KmerBase:
    '''Base class for k-mer-based feature extractors.
    
    Attributes
    ----------
    `k`: `int`
        Length of to-be-generated nucleotide combinations in the vocabulary.
    `uncertain`: `str`
        Optional character that indicates any base that falls outside of ACGT.
    `k-mers`: `dict[str:int]`
        Dictionary containing k-mers (keys) and corresponding indices (values).
    '''

    def __init__(self, k, uncertain=''):
        '''Initializes `KmerBase` object. 
        
        Arguments
        ---------
        `k`: `int`
            Length of to-be-generated nucleotide combinations in the vocabulary.
        `uncertain`: `str`
            Optional character that indicates any base that falls outside of 
            ACGT (default is `''`).'''
        
        self.k = k
        self.uncertain = uncertain
        self.kmers = {
            ''.join(list(kmer)):i for i, kmer in enumerate(
                itertools.product('ACGT' + uncertain, repeat=self.k)
            )
        }

    # Decide if we wan't to keep this (currently not used)
    def replace_uncertain_bases(self, sequence):
        '''Replaces non-ACGT bases in `sequence` with `self.uncertain`.'''
        
        if self.uncertain != '':
            return re.sub('[^ACGT]', self.uncertain, sequence)
        else: 
            return sequence
        

class KmerFreqsBase(KmerBase):
    '''Base class for feature extractors relying on k-mer frequency spectra.
    
    Attributes
    ----------
    `k`: `int`
        Length of to-be-generated nucleotide combinations in the vocabulary.
    `apply_to`: `str`
        Indicates what (sub)sequence to apply the calculation to (default is 
        full transcript, can also be 'ORF'). 
    `stride`: `int`
        Step size of sliding window during calculation.
    `uncertain`: `str`
        Optional character that indicates any base that falls outside of ACGT.
    `k-mers`: `dict[str:int]`
        Dictionary containing k-mers (keys) and corresponding indices (values).
    `name`: `list[str]`
        Column names for frequency features (= all k-mers).'''

    def __init__(self, k, apply_to, stride):
        '''Initializes `KmerFreqsBase` object.
        
        Arguments
        ---------
        `k`: `int`
            Length of to-be-generated nucleotide combinations in the vocabulary.
        `apply_to`: `str`
            Indicates what (sub)sequence to apply the calculation to (default is 
            full transcript, can also be 'ORF'). 
        `stride`: `int`
            Step size of sliding window during calculation.'''
        
        super().__init__(k)
        self.apply_to = apply_to
        self.stride = stride

    def get_sequence(self, data_row):
        '''Extract sequence from `data_row` for which distance should be 
        calculated based on `apply_to` attribute.'''
        sequence = data_row['sequence']
        if self.apply_to != 'sequence':
            sequence = sequence[data_row[f'{self.apply_to} (start)']:
                                data_row[f'{self.apply_to} (end)']]
        return sequence
    
    def calculate_kmer_freqs(self, sequence):
        '''Calculates k-mer frequency spectrum for given `sequence`.'''
        sequence = self.replace_uncertain_bases(sequence)
        freqs = np.zeros(len(self.kmers)) # Initialize
        for i in range(self.k, len(sequence)+1, self.stride): # Loop through seq
            try:
                freqs[self.kmers[sequence[i-self.k:i]]] += 1
            except KeyError: # In case of non-canonical bases (e.g. N)
                continue
        return freqs / (freqs.sum() + 1e-7) # Divide by total number of k-mers
    

class KmerFreqs(KmerFreqsBase):
    '''For every k-mer, calculate its occurrence frequency in the sequence
    divided by the total number of k-mers appearing in that sequence.
    
    Attributes
    ----------
    `k`: `int`
        Length of to-be-generated nucleotide combinations in the vocabulary.
    `apply_to`: `str`
        Indicates what (sub)sequence to apply the calculation to (default is 
        full transcript, can also be 'ORF'). 
    `stride`: `int`
        Step size of sliding window during calculation.
    `uncertain`: `str`
        Optional character that indicates any base that falls outside of ACGT.
    `k-mers`: `dict[str:int]`
        Dictionary containing k-mers (keys) and corresponding indices (values).
    `name`: `list[str]`
        Column names for frequency features (= all k-mers).'''

    def __init__(self, k, apply_to='sequence', stride=1):
        '''Initializes `KmerFreqs` object.
        
        Arguments
        ---------
        `k`: `int`
            Length of to-be-generated nucleotide combinations in the vocabulary.
        `apply_to`: `str`
            Indicates what (sub)sequence to apply the calculation to (default is 
            full transcript, can also be 'ORF'). 
        `stride`: `int`
            Step size of sliding window during calculation.
        '''
        super().__init__(k, apply_to, stride)
        prefix = '' if apply_to == 'sequence' else apply_to + ' '
        suffix = '' if stride == 1 else f' s={stride}'
        self.name = [prefix + kmer + suffix for kmer in self.kmers]

    def calculate(self, data):
        '''Calculates k-mer frequencies for every row in `data`.'''
        print(f"Calculating {self.k}-mer frequencies...")
        if self.apply_to == 'sequence':
            data.check_columns(['sequence'])
        else:
            data.check_columns([self.apply_to + suffix 
                                for suffix in [' (start)', ' (end)']])
        freqs = []
        for _, row in utils.progress(data.df.iterrows()):
            sequence = self.get_sequence(row)
            freqs.append(self.calculate_kmer_freqs(sequence)) 
        return np.stack(freqs)


class KmerFreqsPLEK:
    '''Scales k-mer frequencies by 1/(4^(5-k)), compensating for small k-mers 
    occuring more often than large k-mers. Calculates based on original k-mer 
    frequencies (does not recalculate k-mer frequencies.)
    
    Attributes
    ----------
    `k`: `int`
        Length of to-be-generated nucleotide combinations in the vocabulary.
    `name`: `list[str]`
        Column names for frequency features (= all k-mers).
        
    References
    ----------
    PLEK: Li et al. (2014) https://doi.org/10.1186/1471-2105-15-311'''

    def __init__(self, k):
        '''Initializes `KmerFreqsPLEK` object.
        
        Arguments
        ---------
        `k`: `int`
            Length of nucleotide combinations in the vocabulary.'''
        self.k = k 
        self.name = [f'{kmer} (PLEK)' for kmer in KmerBase(k).kmers]

    def calculate(self, data):
        '''Scales k-mer frequencies for every row in `data`.'''
        kmers = list(KmerBase(self.k).kmers)
        data.check_columns(kmers)
        return 1/(4**(5-self.k)) * data.df[kmers]
    

class KmerScore(KmerBase):
    '''Calculates k-mer score, indicating how likely a sequence is to be 
    protein-coding (the higher, the more likely). Sums the log-ratios of k-mer
    frequencies in protein-coding RNA over non-coding RNA. Introduced by CPAT 
    as hexamer score or hexamer usage bias. 

    Attributes
    ----------
    `k`: `int`
        Length of to-be-generated nucleotide combinations in the vocabulary.
    `uncertain`: `str`
        Optional character that indicates any base that falls outside of ACGT.
    `k-mers`: `dict[str:int]`
        Dictionary containing k-mers (keys) and corresponding indices (values).
    `name`: `list[str]`
        Column names for frequency features (= all k-mers).
    `kmer_freqs`: `np.ndarray`
        Log-ratios of k-mer frequencies in protein-coding RNA over non-coding 
        RNA. 
       
    References
    ----------
    CPAT: Wang et al. (2013) https://doi.org/10.1093/nar/gkt006
    FEELnc: Wucher et al. (2017) https://doi.org/10.1093/nar/gkw1306'''

    def __init__(self, data, k, export_path=None):
        '''Initializes `KmerScore` object, calculates log-ratios of k-mer 
        frequencies in protein-coding RNA over non-coding RNA.
        
        Arguments
        ---------
        `data`: `Data` | `str`
            `Data` object containing sequences for calculating k-mer frequency
            bias (training set), or path to file that contains these values. 
        `k`: `int`
            Length of to-be-generated nucleotide combinations in the vocabulary.
        `export_path`: `str`
            Path to save k-mer frequency bias matrix to for later use.'''
        
        super().__init__(k)
        self.name = f'{k}-mer score'
        if type(data) == str:
            self.kmer_freqs = np.loadtxt(data)
        else:
            print(f"Initializing {self.k}-mer score...")
            kmer_freqs = np.zeros((len(self.kmers), 2))
            all_seqs = data.df.groupby('label')['sequence']
            all_seqs = all_seqs.apply(lambda x: "!".join(x.tolist()))
            for j, label in enumerate(['pcrna', 'ncrna']):
                kmer_freqs[:,j] = count_kmers(all_seqs[label],self.kmers,self.k)
            kmer_freqs = kmer_freqs / (kmer_freqs.sum(axis=0) + 1e-7)
            kmer_freqs = kmer_freqs[:,0] / (kmer_freqs[:,1] + 1e-10)
            kmer_freqs = np.log(kmer_freqs + 1e-10)
            kmer_freqs = np.nan_to_num(kmer_freqs, nan=0)
            self.kmer_freqs = kmer_freqs
            if export_path is not None:
                np.savetxt(
                    export_path, self.kmer_freqs, header=f'{k}-mer ' + 
                    'frequency bias matrix for KmerScore object.\n' +  
                    f'Load using KmerScore(data="<filepath>", k={k})', 
                    fmt="%.6f"
                )

    def calculate(self, data):
        '''Calculates k-mer score for every row in `data`.'''
        print(f"Calculating {self.k}-mer scores...")
        scores = []
        for _, row in utils.progress(data.df.iterrows()):
            scores.append(self.calculate_per_sequence(row['sequence']))
        return scores
    
    def calculate_per_sequence(self, sequence):
        '''Calculates k-mer score of `sequence`.'''
        sequence = self.replace_uncertain_bases(sequence)
        score = 0
        for i in range(len(sequence)-self.k+1):
            try: 
                score += self.kmer_freqs[self.kmers[sequence[i:i+self.k]]]
            except KeyError:
                pass
        return score / (len(sequence)-self.k+1)
    
    def plot_bias(self, filepath=None):
        '''Plots log ratio of usage frequency of k-mers in pcRNA/ncRNA.'''
        fig, ax = plt.subplots()
        plot = ax.bar(np.arange(len(self.kmers)), self.kmer_freqs)
        ax.set_xticks(np.arange(len(self.kmers)), self.kmers, fontsize=5, 
                      rotation=90)
        ax.set_ylabel('Frequency coding/non-coding (log ratio)')
        fig.tight_layout()
        if filepath is not None:
            fig.savefig(filepath)
        return fig
    

class KmerDistance(KmerFreqsBase):
    '''Calculates distance to average k-mer profiles of coding and non-coding
    RNA transcripts, as introduced by LncFinder. 
    
    Attributes
    ----------
    `k`: `int`
        Length of to-be-generated nucleotide combinations in the vocabulary.
    `pc_kmer_profile`: `np.ndarray`
        Average k-mer frequency spectrum of protein-coding transcripts.
    `nc_kmer_profile`: `np.ndarray`
        Average k-mer frequency spectrum of non-coding transcripts.
    `dist_type`: 'euc'|'log'
        Whether to use euclididan or logarithmic distance.
    `apply_to`: `str`
        Indicates what (sub)sequence to apply the calculation to (default is 
        full transcript, can also be 'ORF'). 
    `stride`: `int`
        Step size of sliding window during calculation.
    `uncertain`: `str`
        Optional character that indicates any base that falls outside of ACGT.
    `k-mers`: `dict[str:int]`
        Dictionary containing k-mers (keys) and corresponding indices (values).
    `name`: `list[str]`
        Column names for k-mer distance (to protein-/non-coding) features.

    References
    ----------
    LncFinder: Han et al. (2019) https://doi.org/10.1093/bib/bby065'''

    def __init__(self, data, k, dist_type, apply_to='sequence', stride=1,
                 export_path=None):
        '''Initializes `KmerDistance` object, calculating or importing the 
        k-mer profile of coding vs non-coding RNA.
        
        Arguments
        ---------
        `data`: `Data` | `str`
            `Data` object containing sequences for calculating the k-mer profile
             of (non)-coding transcripts (training set), or path to file that c
             ontains these profiles. 
        `k`: `int`
            Length of to-be-generated nucleotide combinations in the vocabulary.
        `dist_type`: 'euc'|'log'
            Whether to use euclididan or logarithmic distance.
        `apply_to`: `str`
            Indicates what (sub)sequence to apply the calculation to (default is
            full transcript, can also be 'ORF'). 
        `stride`: `int`
            Step size of sliding window during calculation (default is 1).
        `export_path`: `str`
            Path to save calculated k-mer profiles to for later use.'''
        
        super().__init__(k, apply_to, stride) # Initialize KmerBase parent
        if dist_type == 'euc':
            self.calculate_distance = lambda a,b: np.sqrt(np.sum((a - b)**2))
        elif dist_type == 'log':
            self.calculate_distance = lambda a,b: (
                np.sum(np.log(a/(b+1e-7) + 1e-7)) # / (n+1e-7)
            )
        else:
            raise ValueError("dist_type should be one of 'euc', 'log'.")

        if type(data) == str: # Load k-mer profiles from file
            kmer_freqs = np.loadtxt(data)
        
        else: # Calculate k-mer profiles from data   
            print(f"Initializing {self.k}-mer distance...")
            kmer_freqs = {label: np.zeros(len(self.kmers)) for label in 
                          ['pcrna', 'ncrna']}

            for _, row in utils.progress(data.df.iterrows()):
                sequence = self.get_sequence(row)
                kmer_freqs[row['label']] += self.calculate_kmer_freqs(sequence)

            for label in ['pcrna', 'ncrna']:
                kmer_freqs[label] = (kmer_freqs[label] / # Average over total
                                     (kmer_freqs[label].sum() + 1e-7))
            kmer_freqs = np.stack((kmer_freqs['pcrna'], kmer_freqs['ncrna']))

            if export_path is not None: # Export k-mer profiles
                np.savetxt(
                    export_path, kmer_freqs, header=f'{k}-mer frequency ' + 
                    'profiles for KmerDistance object.\n' +  'Load using ' +
                    f'KmerDistance("<filepath>", k={k}, "<dist_type>"' + 
                    f', apply_to="{apply_to}", stride={stride})', fmt="%.6f"
                )
        
        prefix = '' if apply_to == 'sequence' else apply_to + ' '
        self.name = [prefix + f'{k}-mer {dist_type}Dist {i} s={stride}'
                     for i in ['pc', 'nc']]
        self.pc_kmer_profile = kmer_freqs[0]
        self.nc_kmer_profile = kmer_freqs[1]

    def calculate(self, data):
        '''Calculates k-mer distance for every row in `data`.'''
        
        print(f"Calculating {self.k}-mer distances...")
        distances = []
        for _, row in utils.progress(data.df.iterrows()):
            sequence = self.get_sequence(row)
            kmer_freqs = self.calculate_kmer_freqs(sequence)
            distances.append([
                self.calculate_distance(kmer_freqs, self.pc_kmer_profile),
                self.calculate_distance(kmer_freqs, self.nc_kmer_profile)
            ])

        return distances
    

class KmerDistanceRatio: 
    '''Calculates the ratio of k-mer distance to protein-coding over k-mer 
    distance to non-coding, as introduced by LncFinder.
    
    Attributes
    ----------
    `k`: `int`
        Length of to-be-generated nucleotide combinations in the vocabulary.
    `dist_type`: 'euc'|'log'
        Whether to use euclididan or logarithmic distance.
    `apply_to`: `str`
        Indicates what (sub)sequence to apply the calculation to (default is 
        full transcript, can also be 'ORF'). 
    `stride`: `int`
        Step size of sliding window during calculation.
    `name`: `list[str]`
        Column names for k-mer distance (to protein-/non-coding) features.
    
    References
    ----------
    LncFinder: Han et al. (2019) https://doi.org/10.1093/bib/bby065'''

    def __init__(self, k, dist_type, apply_to='sequence', stride=1):
        '''Initializes `KmerDistanceRatio` object.
        
        Arguments
        ---------
        `k`: `int`
            Length of to-be-generated nucleotide combinations in the vocabulary.
        `dist_type`: 'euc'|'log'
            Whether to use euclididan or logarithmic distance.
        `apply_to`: `str`
            Indicates what (sub)sequence to apply the calculation to (default is
            full transcript, can also be 'ORF'). 
        `stride`: `int`
            Step size of sliding window during calculation (default is 1).'''

        self.k = k
        self.dist_type = dist_type
        self.apply_to = apply_to
        self.stride = stride
        prefix = '' if apply_to == 'sequence' else apply_to + ' '
        self.name = prefix + f'{k}-mer {dist_type}DistRatio s={stride}'
    
    def calculate(self, data):
        '''Calculates k-mer distance ratio for every row in `data`.'''
        prefix = '' if self.apply_to == 'sequence' else self.apply_to + ' '
        columns = [prefix + f'{self.k}-mer {self.dist_type}Dist {i} ' + 
                   f's={self.stride}' for i in ['pc', 'nc']]
        data.check_columns(columns)
        return data.df[columns[0]] / (data.df[columns[1]] + 1e-7)
    

class ORFCoordinates:
    '''Determines Open Reading Frame (ORF) coordinates, similar to NCBI's 
    ORFFinder (https://www.ncbi.nlm.nih.gov/orffinder/)
    
    Attributes
    ----------
    `name`: `list[str]`
        Column names for ORF coordinates ('ORF (start)', 'ORF (end)').
    `min_length`: `int`
        Minimum required length for an ORF.
    `relaxation`: `int`
        Relaxation type of the ORF algorithm, as defined by FEELnc. 
        * 0: Start and stop codon is required.
        * 1: Start codon is required.
        * 2: Stop codon is required.
        * 3: Start or stop codon is required.
        * 4: If no ORF found, use full-length transcript.

    References
    ----------
    FEELnc: Wucher et al. (2017) https://doi.org/10.1093/nar/gkw1306'''

    def __init__(self, min_length=75, relaxation=0):
        '''Initializes `ORFCoordinates` object.
        
        Arguments
        ---------
        `min_length`: `int`
            Minimum required length for an ORF (default=75).
        `relaxation`: `int`
            Relaxation type of the ORF algorithm, as defined by FEELnc. 
            * 0: Start and stop codon is required (= default).
            * 1: Start codon is required.
            * 2: Stop codon is required.
            * 3: Start or stop codon is required.
            * 4: If no ORF found, use full-length transcript.

        References
        ----------
        FEELnc: Wucher et al. (2017) https://doi.org/10.1093/nar/gkw1306'''

        self.min_length = min_length
        self.relaxation = relaxation
        suffix = '' if relaxation == 0 else str(relaxation)
        self.name = [f'ORF{suffix} (start)', f'ORF{suffix} (end)']

    def calculate(self, data):
        '''Calculates ORF for every row in `data`.'''
        print("Finding Open Reading Frames...")

        # Relaxation 0, 1, and 2: calculate per sequence
        if self.relaxation < 3:
            orfs = []
            for _, row in utils.progress(data.df.iterrows()):
                orfs.append(self.calculate_per_sequence(row['sequence']))

        # Relaxation 3: Start OR stop codon (= longest of ORF1 and ORF2)
        elif self.relaxation == 3: 
            columns = ['ORF1 (start)','ORF1 (end)','ORF2 (start)','ORF2 (end)']
            data.check_columns(columns)
            values = data.df[columns].values
            condition = (values[:,1]-values[:,0] <= 
                         values[:,3]-values[:,2]).astype(int)
            indices = np.stack((0 + 2*condition, 1 + 2*condition),axis=1)
            orfs = values[np.arange(len(data))[:, None], indices]

        # Relaxation 4: If no ORF found, return sequence length
        elif self.relaxation == 4: # 
            columns = ['ORF3 (start)', 'ORF3 (end)']
            data.check_columns(columns)
            values = data.df[columns].values
            lengths = (Length().calculate(data).values / 3).astype(int)*3
            values = np.concatenate(
                (values, np.zeros((len(data),1), dtype=int), lengths[:, None]), 
                axis=1
            )
            condition = ((values[:,0] == -1) & (values[:,1] == -1)).astype(int)
            indices = np.stack((0 + 2*condition, 1 + 2*condition),axis=1)
            orfs = values[np.arange(len(data))[:, None], indices]

        return orfs
    
    def calculate_per_sequence(self, sequence): 
        '''Returns start (incl.) and stop (excl.) position of longest ORF in
        `sequence`.'''

        if self.relaxation not in [0,1,2]:
            raise ValueError("Only use this method for 0 <= relaxation < 3")

        start_codons, stop_codons = [], []
        for i in range(len(sequence)-2): # Loop through sequence
            codon = sequence[i:i+3]
            if codon == 'ATG': # Store positions of start/stop codons
                start_codons.append(i)
            elif codon in ['TAA', 'TAG', 'TGA']:
                stop_codons.append(i+3)

        if self.relaxation == 1: # Stop codon not required...
            # ... add final three positions as possible stop positions
            stop_codons = stop_codons + [len(sequence)-i for i in range(3)]
        elif self.relaxation == 2: # Start codon not required...
            # ... add first three positions as possible start positions
            start_codons = start_codons + [i for i in range(3)]

        # If no start/stop codons found, no ORF    
        if len(start_codons) < 0 or len(stop_codons) < 0:
            return -1, -1
        
        # Convert to arrays, calculate distance for every start/stop combi        
        start_codons = np.array(start_codons).reshape(-1,1)
        stop_codons = np.array(stop_codons).reshape(1,-1)
        lengths = stop_codons - start_codons

        # Filter to mark 'illegal' distances (negative/non-triplets)
        filter = (lengths < 0) | (lengths % 3 != 0)
        if filter.sum() == lengths.shape[0]*lengths.shape[1]:
            return -1, -1 # No ORF if all distances are illegal

        # For every start codon, retrieve index of first encountered stop codon
        lengths[filter] = np.max(lengths) + 1 # High number to mask argmin
        first_stop = np.argmin(lengths, axis=1) 
        # Retrieve the max length of every start/stop combinations
        lengths[filter] = 0 # Low number to mask argmax
        start_i = np.argmax(lengths[np.arange(len(lengths)),first_stop])
        stop_i = first_stop[start_i]

        if lengths[start_i, stop_i] < self.min_length:  
            return -1, -1
        else:
            return start_codons[start_i,0], stop_codons[0,stop_i]
        
    
class ORFLength:
    '''Calculates length of Open Reading Frame (ORF) based on coordinates.
    
    Attributes
    ----------
    `relaxation`: `list`|`int`
        The relaxation level(s) of the ORFs for which this feature must be 
        calculated (default is 0).
    `name`: `list[str]`
        Column names for ORF length (given relaxation type).'''

    def __init__(self, relaxation=0):
        '''Initializes `ORFLength` object.
        
        Arguments
        ---------
        `relaxation`: `list`|`int`
            The relaxation level(s) of the ORFs for which this feature must be 
            calculated (default is 0).'''
        
        self.relaxation = relaxation
        self.name = orf_column_names(['length'], relaxation)

    def calculate(self, data):
        '''Calculates ORF length for every row in `data`.'''
        data.check_columns(orf_column_names(['(end)', '(start)'], 
                                            self.relaxation))
        return (data.df[orf_column_names(['(end)'],self.relaxation)].values - 
                data.df[orf_column_names(['(start)'],self.relaxation)].values) 
    

class ORFCoverage:
    '''Calculates ORF coverage (ORF length / sequence length).

    Attributes
    ----------
    `relaxation`: `list`|`int`
        The relaxation level(s) of the ORFs for which this feature must be 
        calculated (default is 0).
    `name`: `list[str]`
        Column names for ORF length ('ORF length').'''    

    def __init__(self, relaxation=0):
        '''Initializes `ORFCoverage` object.
        
        Arguments
        ---------
        `relaxation`: `list`|`int`
            The relaxation level(s) of the ORFs for which this feature must be 
            calculated (default is 0).'''
        
        self.relaxation = relaxation
        self.name = orf_column_names(['coverage'], relaxation)

    def calculate(self, data):
        '''Calculates ORF coverage for every row in `data`.'''
        data.check_columns(orf_column_names(['length'], self.relaxation))
        data.check_columns(['length']) 
        return (data.df[orf_column_names(['length'],self.relaxation)].values /
                data.df[['length']].values)


class ORFProtein:
    '''Translates ORF of transcript into amino acid sequence.
    
    Attributes
    ----------
    `name`: `str`
        Column name for ORF protein ('ORF protein').'''
    
    def __init__(self):
        '''Initializes `ORFProtein` object.'''
        self.name = 'ORF protein'

    def calculate(self, data):
        '''Calculates ORF protein for every row in `data`.'''
        print("Translating transcript ORFs into amino acids...")
        seqs = []
        data.check_columns(['ORF (start)', 'ORF (end)', 'sequence'])
        for i, row in utils.progress(data.df.iterrows()):
            orf = row['sequence'][row['ORF (start)']:row['ORF (end)']]
            seqs.append(self.calculate_per_sequence(orf))
        return seqs
    
    def calculate_per_sequence(self, sequence):
        '''Translates a given `sequence` into a amino-acid sequence.'''
        return str(Seq(sequence[:-3]).translate()) # Exclude stop codon


class ORFProteinAnalysis:
    '''Calculates features for the protein encoded by the ORF using methods from
    `Bio.SeqUtils.ProtParam.ProteinAnalysis`.

    Attributes
    ----------
    `features`: `dict`
        Dictionary with to-be-calculated features, with names (`str`) as keys,
        and corresponding methods of `ProteinAnalysis` as values.
    `name`: `list[str]`
        Column names for ORF features (inferred from `features`).'''

    def __init__(self, features={
        'pI': ProteinAnalysis.isoelectric_point,
        'MW': ProteinAnalysis.molecular_weight,
        'aromaticity': ProteinAnalysis.aromaticity,
        'instability': ProteinAnalysis.instability_index,
        'gravy': ProteinAnalysis.gravy,
        'helix': lambda x: ProteinAnalysis.secondary_structure_fraction(x)[0],
        'turn': lambda x: ProteinAnalysis.secondary_structure_fraction(x)[1],
        'sheet': lambda x: ProteinAnalysis.secondary_structure_fraction(x)[2],
    }):
        '''Initializes `ORFProteinAnalysis` object. 
        
        Arguments
        ---------
        `features`: `dict`
            Dictionary with to-be-calculated features, with names (`str`) as 
            keys, and corresponding methods of `ProteinAnalysis` as values.
            Default is:
            * Isoelectric point (pI) 
            * Molecular weight (MW)
            * Aromaticity
            * Instability index
            * Gravy
            * Helix (ratio of associated amino acids) 
            * Turn (ratio of associated amino acids) 
            * Sheet (ratio of associated amino acids) '''
        
        self.features = features
        self.name = [f'ORF {feature}' for feature in features]

    def calculate(self, data):
        '''Calculates ORF protein feature(s) for all rows in `data`.'''
        print("Calculating features of ORF protein...")
        results = []
        data.check_columns(['ORF protein'])
        for i, row in utils.progress(data.df.iterrows()):
            results.append(self.calculate_per_sequence(row['ORF protein']))
        return results
    
    def calculate_per_sequence(self, sequence):
        '''Calculates ORF protein feature(s) for a given amino acid 
        `sequence`.'''
        if len(sequence) == 0 or 'X' in sequence:
            return [np.nan for name in self.features]
        else:
            sequence = ProteinAnalysis(sequence)
            return [self.features[name](sequence) for name in self.features]
        

class ORFIsoelectric(ORFProteinAnalysis):
    '''Theoretical isoelectric point of the protein encoded by the ORF.'''

    def __init__(self):
        '''Initializes `IsoelectricPoint` object.'''
        super().__init__(features={'pI':ProteinAnalysis.isoelectric_point})


class ORFAminoAcidFreqs:
    '''Calculates occurrence frequencies in the protein encoded by an ORF 
    for all amino acids.
    
    Attributes
    ----------
    `amino_acids`: `str`
        All 20 possible amino acid symbols.
    `name`: `list[str]`
        List of column names for ORF amino acid frequencies.
        
    References
    ----------
    CONC: Blake et al. (2006) https://doi.org/10.1371/journal.pgen.0020029'''

    def __init__(self):
        '''Initializes `ORFAminoAcidFreqs` object.'''
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.name = [f'{symbol} (ORF,aa)' for symbol in self.amino_acids]

    def calculate(self, data):
        '''Calculates ORF amino acid frequencies for all rows in `data`.'''
        print("Calculating ORF amino acids frequencies...")
        data.check_columns(['ORF protein'])
        freqs = []
        for _, row in utils.progress(data.df.iterrows()):
            freqs.append(self.calculate_per_sequence(row['ORF protein'])) 
        return np.stack(freqs)
    
    def calculate_per_sequence(self, protein):
        '''Calculates ORF amino acid frequencies for a given `protein`.'''
        occ_freqs = np.array(
            [len(re.findall(aa, protein)) for aa in self.amino_acids]
        )
        return occ_freqs / (len(protein))


class FickettTestcode:
    '''Calculates the Fickett TESTCODE statistic as used by CPAT. Summarizes 
    bias in nucleotide position values and frequencies. 

    Attributes
    ----------
    `name`: `str`
        Column name for Fickett score ('Fickett TESTCODE')
    `pos_intervals`: `np.ndarray`
        Thresholds determining which index to use for the position LUT. 
    `cont_intervals`: `np.ndarray`
        Thresholds determining which index to use for the content LUT. 
    `pos_lut`: `np.ndarray`
        Position value look-up table. Percentage of coding fragments in the
        interval (as defined by `pos_interval`) divided by the total number of 
        fragments in the interval.
    `cont_lut`: `np.ndarray`
        Nucleotide content look-up table. Percentage of coding fragments in the
        interval (as defined by `cont_interval`) divided by the total number of 
        fragments in the interval.
    `fickett_weights`: `np.ndarray`
        Parameters used to weigh the four position values and four nucleotide 
        content values. Equals the percentage of time that each value alone can
        successfully distinguish between coding/noncoding sequences.
    
    References
    ----------
    CPAT: Wang et al. (2013) https://doi.org/10.1093/nar/gkt006
    Fickett et al. (1982) https://doi.org/10.1093/nar/10.17.5303'''

    def __init__(self, data, export_path=None):
        '''Initializes `FickettTestcode` object.
        
        Arguments
        ---------
        `data`: `Data` | `str`
            `Data` object used to calculate the look-up tables, or path to file 
            containing them.
        `export_path`: `str`
            Path to save lookup tables and weights to for later use.'''

        self.name = 'Fickett TESTCODE'
        self.pos_intervals = np.arange(1.1, 1.91, 0.1)
        self.cont_intervals = np.arange(0.17, 0.34, 0.02)
        if type(data) == str:
            loaded = self.loadtxt(data)
            self.pos_lut, self.cont_lut, self.fickett_weights = loaded
        else:
            # TODO recalculating LUT tables
            raise NotImplementedError()
            if export_path is not None:
                self.savetxt(export_path)
        
    def loadtxt(self, data):
        '''Extracts look-up tables and weights from txt file.'''
        data = np.loadtxt(data)
        fickett_weights = np.concatenate((data[0],data[1]))
        pos_lut = data[2:12]
        cont_lut = data[12:]
        return pos_lut, cont_lut, fickett_weights

    def savetxt(self, filepath):
        '''Saves look-up tables and weights to txt file.'''
        output = np.zeros((22,4))
        output[0] = self.fickett_weights[:4]
        output[1] = self.fickett_weights[4:]
        output[2:12] = self.pos_lut 
        output[12:] = self.cont_lut
        np.savetxt(filepath, output, fmt="%.6f", header='Fickett TESTCODE ' + 
                   'intervals and look-up tables for FickettTestcode object.\n'+
                   'Load using FickettTestcode(data="<filepath>")')

    def calculate(self, data):
        '''Calculates Fickett score for every row in `data`.'''
        print("Calculating Fickett TESTCODE...")
        scores = []
        for _, row in utils.progress(data.df.iterrows()):
            scores.append(self.calculate_per_sequence(row['sequence']))
        return scores

    def calculate_per_sequence(self, sequence):
        '''Calculates Fickett score of `sequence`.'''
        
        probs = []
        
        # Position scores
        pos_values = self.position_values(sequence)
        for j, value in enumerate(pos_values):
            prob = self.pos_lut[0,j] # Initialize at lowest interval value
            # Loop through intervals and check if upper bound is surpassed
            for i, threshold in enumerate(self.pos_intervals):
                if value >= threshold:
                    prob = self.pos_lut[i+1,j]
            probs.append(prob)
        
        # Nucleotide content scores
        nucl_freqs = self.nucleotide_frequencies(sequence)
        for j, value in enumerate(nucl_freqs):
            prob = self.cont_lut[0,j] # Initialize at lowest interval value
            # Loop through intervals and check if upper bound is surpassed
            for i, threshold in enumerate(self.cont_intervals):
                if value >= threshold:
                    prob = self.cont_lut[i+1,j]
            probs.append(prob)

        return np.dot(self.fickett_weights, np.array(probs))
    
    def position_values(self, sequence):
        '''Calculates position values of `sequence` for every codon, returning a
        list of four values (corresponding to A, C, G, T, respectively). 
        Describes bias of nucleotides occurrence at specific codon positions.'''
        occ_values = [[],[],[],[]]
        for i in range(3):
            subseq = sequence[i::3]
            for i, base in enumerate('ACGT'):
                occ_values[i].append(len(re.findall(base, subseq)))
        occ_values = np.array(occ_values)
        return np.max(occ_values, axis=1)/(np.min(occ_values, axis=1)+1)

    def nucleotide_frequencies(self, sequence):
        '''Calculates nucleotide frequencies of `sequence` for every codon, 
        returning a list of four values (corresponding to A, C, G, T, 
        respectively).'''
        occ_freqs = np.array(
            [len(re.findall(base, sequence)) for base in 'ACGT']
        )
        return occ_freqs / (len(sequence))
    

class MLCDS:
    '''Determines Most-Like Coding Sequence (MLCDS) coordinates based on 
    Adjoined Nucleotide Triplets (ANT), as proposed by CNCI. Calculates six 
    MLCDSs, based on two directions and three reading frames. The MLCDSs are 
    sorted based on score, where MLCDS1 has the highest score.

    Attributes
    ----------
    `name`: `list[str]`
        Column names for MLCDS coordinates and scores.
    `kmers`: `dict[str:int]`
        Dictionary containing 3-mers (keys) and corresponding indices (values).
    `ant_matrix`: `np.ndarray`
        Adjoined Nucleotide Triplet matrix, containing the log-ratios of ANTs
        appearing in coding over non-coding RNA. 

    References
    ----------
    CNCI: Sun et al. (2013) https://doi.org/10.1093/nar/gkt646
    CNIT: Guo et al. (2019) https://doi.org/10.1093/nar/gkz400''' 

    def __init__(self, data, export_path=None):
        '''Initializes `MLCDS` object, calculates ANT matrix.
        
        Arguments
        ---------
        `data`: `Data`
            `Data` object used to calculate the ANT matrix (trainset), or path
            to file containing this matrix.'''
        
        self.name = [f'MLCDS{i} {feature}' for i in range(1,7) for 
                     feature in ['(start)', '(end)', 'score']]
        self.kmers = KmerBase(3).kmers
        if type(data) == str:
            self.ant_matrix = np.loadtxt(data)
        else:
            self.ant_matrix = self.calculate_ant_matrix(data)
            if export_path is not None:
                np.savetxt(export_path, self.ant_matrix, fmt="%.6f", header=
                        "ANT matrix for MLCDS object.\n" + 
                        'Load using MLCDS(data="<filepath>")')

    def calculate_ant_matrix(self, data):
        '''Calculates Adjoined Nucleotide Triplet matrix, containing the
        log-ratios of ANTs appearing in coding over non-coding RNA.'''

        # Initialize matrix
        print("Calculating ANT matrix...")
        ant_matrix = np.zeros((2, len(self.kmers), len(self.kmers)))

        # Loop through coding and non-coding
        all_seqs = data.df.groupby('label')['sequence']
        all_seqs = all_seqs.apply(lambda x: "!".join(x.tolist()))
        for i, label in enumerate(['pcrna', 'ncrna']):
            for p in utils.progress(range(6, len(all_seqs[label])+1)):
                try:
                    i_left_kmer = self.kmers[all_seqs[label][p-6:p-3]]
                    i_right_kmer = self.kmers[all_seqs[label][p-3:p]]
                    ant_matrix[i, i_left_kmer, i_right_kmer] += 1
                except KeyError:
                    continue

        # Calculate log ratio of ANT 
        ant_matrix = np.log2(
            (ant_matrix[0] / ant_matrix[0].sum() + 1e-7) / 
            (ant_matrix[1] / ant_matrix[1].sum() + 1e-7) + 1e-10
        )
        ant_matrix = np.nan_to_num(ant_matrix, nan=0)

        return ant_matrix

    def calculate(self, data):
        '''Calculates MLCDS for every row in `data`.'''
        mlcds = []
        print("Calculating MLCDS...")
        for _, row in utils.progress(data.df.iterrows()):
            mlcds.append(self.calculate_per_sequence(row['sequence']))
        return mlcds

    def calculate_per_sequence(self, sequence):
        '''Calculates MLCDS of `sequence`.'''

        # Calculating MLCDS for six reading frames 
        scores = []
        coordinates = []
        for dir in [1, -1]: # Frame direction
            for offset in range(3): # Frame offset
                frame = self.get_reading_frame(sequence, dir, offset)
                start_, end_, score = self.get_mlcds(frame) 
                # Convert to coordinates of original sequence
                start, end = self.get_abs_coordinates(start_, end_, dir, offset)
                assert sequence[start:end:dir] == frame[start_:end_] # = check
                coordinates.append((start, end))
                scores.append(score)

        # Sorting based on score
        features = []
        sorting = np.argsort(scores)[::-1]
        for i in sorting:
            features.append(coordinates[i][0])
            features.append(coordinates[i][1])
            features.append(scores[i])

        return features

    def get_reading_frame(self, sequence, dir, offset):
        '''Extract reading frame from sequence given direction and offset''' 
        return sequence[::dir][offset:len(sequence)-((len(sequence)-offset)%3)]

    def get_mlcds(self, reading_frame):
        '''Calculates Most-Like Coding Sequence (MLCDS) given reading frame, 
        using the Adjoined Nucleotide Triplet (ANT) matrix. 

        Adapted from `cal_score` function in CNIT's code by Fang Shuangsang.'''

        # Calculating possible optimal cumulative scores
        coden_num = int(len(reading_frame) / 3)
        score = [-1000 for _ in range(coden_num)] # Track cumulative score
        start_list = [0] # Track optimal starting points
        for i in range(1, coden_num): # Loop over codons
            left = reading_frame[(i-1)*3:i*3] # Get hexamer...
            right = reading_frame[i*3:(i+1)*3] # ... & extract ANT matrix score
            try:
                tmp_score = self.ant_matrix[self.kmers[left], self.kmers[right]]
            except KeyError: # In case of non-ACTG bases
                tmp_score = 0
            if score[i - 1] < 0: # If cumulative score < 0
                max_start = (i - 1) # Define new starting point
                start_list.append(max_start)
                score[i] = tmp_score # Start over with cumulative score
            else: # If cumulative score >= 0 
                score[i] = score[i - 1] + tmp_score # Accumulate score
                max_start = start_list[i - 1] # Keep old starting point
                start_list.append(max_start)

        # Find optimal cumulative scores
        max_start, max_end = 0, 0
        max_score = -1000
        for i in range(len(score)): # Loop through scores
            start = start_list[i] * 3
            end = i * 3 + 3
            cur_score = score[i]
            if cur_score > max_score: # If new optimal score is found
                max_score = cur_score # Save as optimal score
                max_start, max_end = start, end # Extract start/end coord.

        return max_start, max_end, max_score
    
    def get_abs_coordinates(self, coord1, coord2, dir, offset):
        '''Transforms a set of coordinates relative to a reading frame to a set
        of coordinates that are defined in relation to the original sequence.'''

        base = -1 if dir == -1 else 0
        coord1 = base + dir*(coord1 + offset)
        coord2 = base + dir*(coord2 + offset)

        return coord1, coord2

    def imshow_ant_matrix(self, filepath=None, **kwargs):
        '''Plots ANT matrix.
        
        Arguments
        ---------
        `filepath`: `str`
            If provided, will export figure to this filepath.
        `kwargs`:
            Any `kwargs` accepted by `matplotlib.pyplot.imshow`.
        '''
        fig, ax = plt.subplots()
        plot = ax.imshow(self.ant_matrix, **kwargs)
        ax.set_xticks(np.arange(len(self.kmers)), self.kmers, fontsize=5, 
                      rotation=90)
        ax.set_yticks(np.arange(len(self.kmers)), self.kmers, fontsize=5)
        fig.colorbar(plot)
        fig.tight_layout()
        if filepath is not None:
            fig.savefig(filepath)
        return fig
    
    
class MLCDSLength:
    '''Lengths of Most-Like Coding Sequences.
    
    Attributes
    ----------
    `name`: `list[str]`
        Column name for MLCDS lengths'''

    def __init__(self):
        '''Initializes `MLCDSLength` object.'''
        self.name = [f'MLCDS{i} length' for i in range(1,7)]

    def calculate(self, data):
        '''Calculates MLCDS lengths for all rows in `data`.'''
        start_columns = [f'MLCDS{i} (start)'  for i in range(1,7)]
        end_columns = [f'MLCDS{i} (end)'  for i in range(1,7)]
        data.check_columns(start_columns)
        data.check_columns(end_columns)
        return np.absolute(data.df[end_columns].values - 
                           data.df[start_columns].values) 
    

class MLCDSLengthPercentage:
    '''Length percentages of Most-Like Coding Sequences. Defined as the length
    of the MLCDS with the highest score, divided by the sum of the lengths of
    the remaining MLCDSs. 
    
    Attributes
    ----------
    `name`: `list[str]`
        Column name for MLCDS length percentage ('MLCDS length-percentage')'''

    def __init__(self):
        '''Initializes `MLCDSLengthPercentage` object.'''
        self.name = 'MLCDS length-percentage'

    def calculate(self, data):
        '''Calculates length percentage for every row in `data`.'''
        data.check_columns([f'MLCDS{i} length'  for i in range(1,7)])
        return (
            data.df['MLCDS1 length'] / 
            np.sum(data.df[[f'MLCDS{i} length'  for i in range(2,7)]], axis=1)
        ) 
    

class MLCDSScoreDistance:
    '''Score distance of Most-Like Coding Sequences. Defined as sum of the 
    differences between the score of the highest MLCDS and that of the the 
    remaining ones. 
    
    Attributes
    ----------
    `name`: `str`
        Column name for MLCDS score distance ('MLCDS score-distance')'''

    def __init__(self):
        '''Initializes `MLCDSScoreDistance` object.'''
        self.name = 'MLCDS score-distance'

    def calculate(self, data):
        '''Calculates score distance for every row in `data`.'''
        data.check_columns([f'MLCDS{i} score'  for i in range(1,7)])
        return (
            np.sum(
                [data.df['MLCDS1 score'] - data.df[f'MLCDS{i} score'] 
                 for i in range(2,7)], axis=0
            ) / 5
        )
    

class MLCDSKmerFreqs(KmerFreqsBase):
    '''K-mer Frequencies of the Most-Like Coding Sequence with the highest
    score. 
    
    Attributes
    ----------
    `name`: `list[str]`
        Column name for MLCDS lengths
    `k`: `int`
        Length of to-be-generated nucleotide combinations in the vocabulary.
    `uncertain`: `str`
        Optional character that indicates any base that falls outside of ACGT.
    `k-mers`: `dict[str:int]`
        Dictionary containing k-mers (keys) and corresponding indices (values).
    `name`: `list[str]`
        Column names for frequency features (= all k-mers).'''

    def __init__(self, k):
        '''Initializes `MLCDSKmerFreqs` object.
        
        Arguments
        ---------
        `k`: `int`
            Length of to-be-generated nucleotide combinations in the vocabulary.
        '''
        super().__init__(k, 'MLCDS1', 1)

    def calculate(self, data):
        '''Calculates MLCDS k-mer frequencies for every row in `data`.'''
        print(f"Calculating MLCDS {self.k}-mer frequencies...")
        cols = [self.apply_to + suffix for suffix in [' (start)', ' (end)']]
        data.check_columns(cols)
        data.df[cols[0]] = pd.to_numeric(data.df[cols[0]], downcast='integer')
        data.df[cols[1]] = pd.to_numeric(data.df[cols[1]], downcast='integer')
        freqs = []
        for _, row in utils.progress(data.df.iterrows()):
            sequence = self.get_sequence(row)
            freqs.append(self.calculate_kmer_freqs(sequence)) 
        return np.stack(freqs)


class MLCDSScoreStd:
    '''Standard deviation of Most-Like Coding Sequences scores.
    
    Attributes
    ----------
    `name`: `str`
        Column name for MLCDS score standard deviation ('MLCDS score (std)')'''

    def __init__(self):
        '''Initializes `MLCDSScoreStd` object.'''
        self.name = 'MLCDS score (std)'

    def calculate(self, data):
        '''Calculates MLCDS score standard deivation for all rows in `data`.'''
        columns = [f'MLCDS{i} score'  for i in range(1,7)]
        data.check_columns(columns)
        return data.df[columns].std(axis=1)
    

class MLCDSLengthStd:
    '''Standard deviation of Most-Like Coding Sequences lengths.
    
    Attributes
    ----------
    `name`: `str`
        Column name for MLCDS length standard deviation ('MLCDS length (std)')
    '''

    def __init__(self):
        '''Initializes `MLCDSScoreStd` object.'''
        self.name = 'MLCDS length (std)'

    def calculate(self, data):
        '''Calculates MLCDS length standard deivation for all rows in `data`.'''
        columns = [f'MLCDS{i} length'  for i in range(1,7)]
        data.check_columns(columns)
        return data.df[columns].std(axis=1)
    

# NOTE Currently not included in unittests because of blast dependency
class BLASTXSearch:
    '''Derives features based on a BLAST database search:
    * BLASTX hits: number of blastx hits found for given transcript.
    * BLASTX hit score: mean of the mean log evalue over each of the three 
    reading frames (as proposed by CPC). 
    * BLASTX frame score: mean of the deviation from the hit score over each of
    the three reading frames (as proposed by CPC). 
    * BLASTX identity: sum of the identity percentage for each of the blast hits
    for a given sequence.
    
    Attributes
    ----------
    `database`: `str`
        Path to local BLAST database or name of official BLAST database (when 
        running remotely).
    `remote`: `bool`
        Whether to run remotely or locally. If False, requires a local
        installation of BLAST with a callable blastx program (default is False).
    `evalue`: `float`
        Cut-off value for statistical significance (default is 1e-10).
    `strand`: ['both'|'plus'|'minus']
        Which reading direction(s) to consider (default is 'plus').
    `threads`: `int`
        Specifies how many threads for BLAST to use (when running locally).
    `name`: `list[str]`
        Column names for MLCDS length standard deviation ('MLCDS length (std)')

    References
    ----------
    CPC: Kong et al. (2007) https://doi.org/10.1093/nar/gkm391'''

    def __init__(self, database, remote=False, evalue=1e-10, strand='plus', 
                 threads=None):
        '''Initializes `BLASTXSearch` object. 
        
        Arguments
        ---------
        `database`: `str`
            Path to local BLAST database or name of official BLAST database 
            (when running remotely).
        `remote`: `bool`
            Whether to run remotely or locally. If False, requires a local
            installation of BLAST with a callable blastx program (default is 
            False).
        `evalue`: `float`
            Cut-off value for statistical significance (default is 1e-10).
        `strand`: ['both'|'plus'|'minus']
            Which reading direction(s) to consider (default is 'plus').
        `threads`: `int`
            Specifies how many threads for BLAST to use (when running locally).
        `name`: `list[str]`
            Column names for MLCDS length standard deviation ('MLCDS length 
            (std)')'''

        self.database = database
        self.remote = remote
        self.evalue = evalue
        self.strand = strand
        self.threads = threads
        self.name = ['BLASTX hits', 'BLASTX hit score', 'BLASTX frame score',
                     'BLASTX identity']

    def calculate(self, data):
        '''Calculates BLASTX database search features for all rows in `data`.'''

        data.to_fasta(filepath='temp.fasta') # Generate FASTA query file

        # Generate command based on object configuration
        command = ['blastx', '-query', 'temp.fasta', '-strand', self.strand, 
                   '-db', self.database, '-out', 'temp.csv', '-outfmt', str(10),
                   '-evalue', str(self.evalue)]
        if self.remote: 
            command += ['-remote']
        elif self.threads is not None:
            command += ['-num_threads', str(self.threads)]

        print("Running blastx...")
        subprocess.run(command, check=True) # Run the command
        output = pd.read_csv('temp.csv', header=0, names=[ # Read dataframe
            'query acc.ver', 'subject acc.ver', 'identity', 'alignment length',
            'mismatches', 'gap opens', 'q. start', 'q. end', 's. start', 
            's. end', 'evalue', 'bit score']
        )
        os.remove('temp.fasta') # Remove FASTA query file
        os.remove('temp.csv') # Remove BLASTX output (is in memory now)

        print("Calculating blastx scores...")
        results = []
        for i, row in utils.progress(data.df.iterrows()):
            # Assumes query ids are same as row ids (error sensitive...?)
            results.append(self.calculate_per_sequence( 
                output[output['query acc.ver'] == row['id']] 
            ))

        return results
    
    def calculate_per_sequence(self, blast_result):
        '''Calculate BLASTX features for given query result'''
        
        # Mean log evalue per reading frame
        S = np.array([np.mean(-np.log(
            blast_result[blast_result['q. start'] % 3 == i]['evalue'] + 1e-250
            )) for i in range(3)])

        if np.isnan(S).sum() == 3: # Can't take mean of empty array
            hit_score, frame_score = np.nan, np.nan
        else:
            hit_score = np.nanmean(S)
            frame_score = np.nanmean((hit_score - np.array(S))**2)
        identity = blast_result['identity'].sum()

        return [len(blast_result), hit_score, frame_score, identity]
    

class Complexity:
    '''Calculates the (local) compositional complexity (entropy) of a transcript
    sequence.'''

    def __init__(self):
        '''Initializes `Complexity` object.'''
        self.name = 'Complexity'

    def calculate(self, data):
        '''Calculates local compositional complexity of all rows in `data`.'''
        print("Calculating local composition complexity of sequences...")
        results = []
        for _, row in utils.progress(data.df.iterrows()):
            results.append(self.calculate_per_sequence(row['sequence']))
        return results

    def calculate_per_sequence(self, sequence):
        '''Calculates the complexity for a given `sequence`.'''
        return lcc_simp(sequence)
    

class FeatureEntropy:
    '''Calculates the entropy of specific features of a sequence.
    
    Attributes
    ----------
    `feature_names`: `list[str]`
        Names of the features for which the entropy should be calculated.
    `name`: `str`
        Name of the combined entropy feature calculated by this class.'''

    def __init__(self, new_feature_name, feature_names):
        '''Initializes `FeatureEntropy` object.
        
        Arguments
        ---------
        `new_feature_name`: `str`
            Name of the combined entropy feature calculated by this class.
        `feature_names`: `list[str]`
            Names of the features for which the entropy should be calculated.'''
        
        self.name = new_feature_name
        self.feature_names = feature_names

    def calculate(self, data):
        '''Calculates the entropy of features for every row in `Data`.'''
        print(f"Calculating {self.name}...")
        entropies = []
        for _, row in utils.progress(data.df.iterrows()):
            entropies.append(entropy(list(row[self.feature_names].values)))
        return entropies


# NOTE currently only used by KmerScore, might move there
def count_kmers(sequence, kmers, k, stride=1):
    '''Returns an array of frequencies k-mer counts in `sequence`. Uses k-mer 
    indices as defined by dictionary.'''

    counts = np.zeros(max(kmers.values())+1)

    for i in utils.progress(range(k,len(sequence)+1,stride)):
        try:
            counts[kmers[sequence[i-k:i]]] += 1
        except KeyError:
            continue

    return counts


def orf_column_names(columns, relaxation):
    '''Generate column names for the ORF features in `columns`, for a specific
    `relaxation` type.
    
    Arguments
    ---------
    `columns`: `list[str]`
        List of ORF feature names (e.g. 'length')
    `relaxation`: `list`|`int`
        Relaxation type(s) for the to-be-generated columns.'''
    
    relaxation = [relaxation] if type(relaxation) == int else relaxation
    names = []
    for r in relaxation:
        suffix = r if r > 0 else ''
        names += [f'ORF{suffix} {name}' for name in columns]
    return names