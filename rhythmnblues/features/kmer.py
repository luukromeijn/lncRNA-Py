'''Feature extractors based on k-mer frequencies.'''

import itertools
import re
import matplotlib.pyplot as plt
import numpy as np
from rhythmnblues import utils


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
        suffix = '' if apply_to == 'sequence' else  f' ({apply_to})'
        suffix = suffix if stride == 1 else f'{suffix} s={stride}'
        self.name = [kmer + suffix for kmer in self.kmers]

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
            kmer_freqs = (kmer_freqs[:,0] + 1e-10) / (kmer_freqs[:,1] + 1e-10)
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