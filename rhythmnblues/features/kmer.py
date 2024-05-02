'''Feature extractors based on k-mer frequencies.'''

import itertools
import re
import matplotlib.pyplot as plt
import numpy as np
from rhythmnblues import utils
from rhythmnblues.features.general import SequenceFeature


class KmerBase:
    '''Base class for k-mer-based feature extractors.
    
    Attributes
    ----------
    `k`: `int`
        Length of to-be-generated nucleotide combinations in the vocabulary.
    `stride`: `int`
        Step size of sliding window during calculation.
    `alphabet`: `str`
        Alphabet of characters that the k-mers exist of (default is 'ACGT').
    `uncertain`: `str`
        Optional character that indicates any base that falls outside of ACGT.
    `k-mers`: `dict[str:int]`
        Dictionary containing k-mers (keys) and corresponding indices (values).
    '''

    def __init__(self, k, stride=1, alphabet='ACGT', uncertain=''):
        '''Initializes `KmerBase` object. 
        
        Arguments
        ---------
        `k`: `int`
            Length of to-be-generated nucleotide combinations in the vocabulary.
        `stride`: `int`
            Step size of sliding window during calculation.
        `alphabet`: `str`
            Alphabet of characters that the k-mers exist of (default is 'ACGT'). 
        `uncertain`: `str`
            Optional character that indicates any base that falls outside of 
            ACGT (default is `''`).'''
        
        self.k = k
        self.stride = stride
        self.uncertain = uncertain
        self.alphabet = alphabet
        self.kmers = {
            ''.join(list(kmer)):i for i, kmer in enumerate(
                itertools.product(alphabet + uncertain, repeat=self.k)
            )
        }

    # Decide if we wan't to keep this (currently not used)
    def replace_uncertain_bases(self, sequence):
        '''Replaces non-ACGT bases in `sequence` with `self.uncertain`.'''
        
        if self.uncertain != '':
            return re.sub(f'[^{self.alphabet}]', self.uncertain, sequence)
        else: 
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
    

class KmerFreqs(KmerBase, SequenceFeature):
    '''For every k-mer, calculate its occurrence frequency in the sequence
    divided by the total number of k-mers appearing in that sequence.
    
    Attributes
    ----------
    `k`: `int`
        Length of to-be-generated nucleotide combinations in the vocabulary.
    `scaling`: `float`
        Scaling factor applied to every k-mer spectrum. Usually set to 1, unless
        `PLEK` argument was True at initialization.  
    `apply_to`: `str`
        Indicates which column this class extracts its features from.
    `stride`: `int`
        Step size of sliding window during calculation.
    `alphabet`: `str`
        Alphabet of characters that the k-mers exist of (default is 'ACGT').
    `uncertain`: `str`
        Optional character that indicates any base that falls outside of ACGT.
    `k-mers`: `dict[str:int]`
        Dictionary containing k-mers (keys) and corresponding indices (values).
    `name`: `list[str]`
        Column names for frequency features (= all k-mers).'''

    def __init__(self, k, apply_to='sequence', stride=1, alphabet='ACGT', 
                 PLEK=False):
        '''Initializes `KmerFreqs` object.
        
        Arguments
        ---------
        `k`: `int`
            Length of to-be-generated nucleotide combinations in the vocabulary.
        `apply_to`: `str`
            Indicates which column this class extracts its features from.
        `stride`: `int`
            Step size of sliding window during calculation.
        `alphabet`: `str`
            Alphabet of characters that the k-mers exist of (default is 'ACGT').
        `PLEK`: bool
            If True, will scales k-mer frequencies by 1/(4^(5-k)), compensating
            for small k-mers occuring more often than large k-mers (default is 
            False).

        References
        ----------
        PLEK: Li et al. (2014) https://doi.org/10.1186/1471-2105-15-311'''

        KmerBase.__init__(self, k, stride, alphabet)
        SequenceFeature.__init__(self, apply_to)
        self.scaling = 1/(4**(5-self.k)) if PLEK else 1
        suffix = '' if apply_to == 'sequence' else  f' ({apply_to})'
        suffix = suffix if stride == 1 else f'{suffix} s={stride}'
        suffix = suffix if not PLEK else f'{suffix} (PLEK)'
        self.name = [kmer + suffix for kmer in self.kmers]

    def calculate(self, data):
        '''Calculates k-mer frequencies for every row in `data`.'''
        print(f"Calculating {self.k}-mer frequencies...")
        self.check_columns(data)
        freqs = []
        for _, row in utils.progress(data.df.iterrows()):
            sequence = self.get_sequence(row)
            freqs.append(self.scaling*self.calculate_kmer_freqs(sequence)) 
        return np.stack(freqs)
    

class KmerScore(KmerBase, SequenceFeature):
    '''Calculates k-mer score, indicating how likely a sequence is to be 
    protein-coding (the higher, the more likely). Sums the log-ratios of k-mer
    frequencies in protein-coding RNA over non-coding RNA. Introduced by CPAT 
    as hexamer score or hexamer usage bias. 

    Attributes
    ----------
    `k`: `int`
        Length of to-be-generated nucleotide combinations in the vocabulary.
    `kmer_freqs`: `np.ndarray`
        Log-ratios of k-mer frequencies in protein-coding RNA over non-coding 
        RNA. 
    `apply_to`: `str`
        Indicates which column this class extracts its features from.
    `stride`: `int`
        Step size of sliding window during calculation.
    `alphabet`: `str`
        Alphabet of characters that the k-mers exist of (default is 'ACGT').
    `uncertain`: `str`
        Optional character that indicates any base that falls outside of ACGT.
    `k-mers`: `dict[str:int]`
        Dictionary containing k-mers (keys) and corresponding indices (values).
    `name`: `list[str]`
        Column names for frequency features (= all k-mers).
       
    References
    ----------
    CPAT: Wang et al. (2013) https://doi.org/10.1093/nar/gkt006
    FEELnc: Wucher et al. (2017) https://doi.org/10.1093/nar/gkw1306'''

    def __init__(self, data, k, apply_to='sequence', stride=1, alphabet='ACGT',  
                 export_path=None):
        '''Initializes `KmerScore` object, calculates log-ratios of k-mer 
        frequencies in protein-coding RNA over non-coding RNA.
        
        Arguments
        ---------
        `data`: `Data` | `str`
            `Data` object containing sequences for calculating k-mer frequency
            bias (training set), or path to file that contains these values. 
        `k`: `int`
            Length of to-be-generated nucleotide combinations in the vocabulary.
        `apply_to`: `str`
            Indicates which column this class extracts its features from 
            (default is full transcript).
        `stride`: `int`
            Step size of sliding window during calculation (default is 1).
        `alphabet`: `str`
            Alphabet of characters that the k-mers exist of (default is 'ACGT').
        `export_path`: `str`
            Path to save k-mer frequency bias matrix to for later use.'''
        
        # Initializing parent classes
        KmerBase.__init__(self, k, stride, alphabet)
        SequenceFeature.__init__(self, apply_to)

        # Setting an unambiguous proper name
        suffix = '' if apply_to == 'sequence' else  f' ({apply_to})'
        suffix = suffix if stride == 1 else f'{suffix} s={stride}'
        self.name = f'{k}-mer score{suffix}'

        if type(data) == str: # If provided, load kmer frequency bias data
            self.kmer_freqs = np.loadtxt(data)

        else: # If not, calculate from data
            print(f"Initializing {self.k}-mer score...")
            kmer_freqs = np.zeros((len(self.kmers), 2))

            # Count occurrences
            all_seqs = data.df.groupby('label')['sequence']
            all_seqs = all_seqs.apply(lambda x: "!".join(x.tolist()))
            for j, label in enumerate(['pcrna', 'ncrna']):
                kmer_freqs[:,j] = self.count_kmers(all_seqs[label])

            # Convert to log ratio
            kmer_freqs = kmer_freqs / (kmer_freqs.sum(axis=0) + 1e-7)
            kmer_freqs = (kmer_freqs[:,0] + 1e-10) / (kmer_freqs[:,1] + 1e-10)
            kmer_freqs = np.log(kmer_freqs + 1e-10)
            kmer_freqs = np.nan_to_num(kmer_freqs, nan=0)
            self.kmer_freqs = kmer_freqs

            if export_path is not None: # Export if export_path specified
                np.savetxt(
                    export_path, self.kmer_freqs, header=f'{k}-mer ' + 
                    'frequency bias matrix for KmerScore object.\n' +  
                    f'Load using KmerScore(data="<filepath>", k={k}, ' + 
                    f'apply_to="{apply_to}", stride={stride})', 
                    fmt="%.6f"
                )

    def calculate(self, data):
        '''Calculates k-mer score for every row in `data`.'''
        print(f"Calculating {self.k}-mer scores...")
        self.check_columns(data)
        scores = []
        for _, row in utils.progress(data.df.iterrows()):
            scores.append(self.calculate_per_sequence(self.get_sequence(row)))
        return scores
    
    def calculate_per_sequence(self, sequence):
        '''Calculates k-mer score of `sequence`.'''
        sequence = self.replace_uncertain_bases(sequence)
        score = 0
        j = 0
        for i in range(0, len(sequence)-self.k+1, self.stride):
            try: 
                score += self.kmer_freqs[self.kmers[sequence[i:i+self.k]]]
                j += 1
            except KeyError:
                pass
        try:
            return score / j 
        except ZeroDivisionError:
            return np.nan
    
    def count_kmers(self, sequence):
        '''Returns an array of frequencies k-mer counts in `sequence`.'''
        counts = np.zeros(max(self.kmers.values())+1)
        for i in utils.progress(range(self.k,len(sequence)+1,self.stride)):
            try:
                counts[self.kmers[sequence[i-self.k:i]]] += 1
            except KeyError:
                continue
        return counts
    
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
    

class KmerDistance(KmerBase, SequenceFeature):
    '''Calculates distance to average k-mer profiles of coding and non-coding
    RNA transcripts, as introduced by LncFinder. Also calculates the ratio of 
    the two distances. 
    
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
        Indicates which column this class extracts its features from.
    `stride`: `int`
        Step size of sliding window during calculation.
    `alphabet`: `str`
        Alphabet of characters that the k-mers exist of.
    `uncertain`: `str`
        Optional character that indicates any base that falls outside of ACGT.
    `k-mers`: `dict[str:int]`
        Dictionary containing k-mers (keys) and corresponding indices (values).
    `name`: `list[str]`
        Column names for k-mer distance (to protein-/non-coding) (ratio) 
        features.

    References
    ----------
    LncFinder: Han et al. (2019) https://doi.org/10.1093/bib/bby065'''

    def __init__(self, data, k, dist_type, apply_to='sequence', stride=1,
                 alphabet='ACGT', export_path=None):
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
            Indicates which column this class extracts its features from.
        `stride`: `int`
            Step size of sliding window during calculation (default is 1).
        `alphabet`: `str`
            Alphabet of characters that the k-mers exist of (default is 'ACGT').
        `export_path`: `str`
            Path to save calculated k-mer profiles to for later use.'''
        
        KmerBase.__init__(self, k, stride, alphabet)
        SequenceFeature.__init__(self, apply_to)
        
        if dist_type == 'euc': # Define euclidian distance
            self.calculate_distance = lambda a,b: np.sqrt(np.sum((a - b)**2))
        elif dist_type == 'log':
            eps = 1/(100*len(self.kmers))
            self.calculate_distance = lambda a,b: np.sum(np.log((a+eps)/b))
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
        self.name = ([prefix + f'{k}-mer {dist_type}Dist {i} s={stride}'
                      for i in ['pc', 'nc']] + 
                     [prefix + f'{k}-mer {dist_type}DistRatio s={stride}'])
        self.pc_kmer_profile = kmer_freqs[0]
        self.nc_kmer_profile = kmer_freqs[1]

    def calculate(self, data):
        '''Calculates k-mer distance for every row in `data`.'''
        
        print(f"Calculating {self.k}-mer distances...")
        self.check_columns(data)
        distances = []
        for _, row in utils.progress(data.df.iterrows()):
            sequence = self.get_sequence(row)
            kmer_freqs = self.calculate_kmer_freqs(sequence)
            dist_pc = self.calculate_distance(kmer_freqs, self.pc_kmer_profile)
            dist_nc = self.calculate_distance(kmer_freqs, self.nc_kmer_profile)
            distances.append([
                dist_pc,
                dist_nc,
                dist_pc / (dist_nc + 1e-7)
            ])

        return distances