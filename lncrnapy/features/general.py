'''Feature extractors for general features.'''

from Bio.SeqUtils.lcc import lcc_simp
from scipy.stats import entropy
from lncrnapy import utils
from lncrnapy.features.sequence_base import SequenceBase
import numpy as np
import re


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


class Complexity:
    '''Calculates the (local) compositional complexity (entropy) of a transcript
    sequence.'''

    def __init__(self):
        '''Initializes `Complexity` object.'''
        self.name = 'complexity'

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
    

class Entropy:
    '''Calculates the shannon entropy of specific features of a sequence.
    
    Attributes
    ----------
    `feature_names`: `list[str]`
        Names of the features for which the entropy should be calculated.
    `name`: `str`
        Name of the combined entropy feature calculated by this class.'''

    def __init__(self, new_feature_name, feature_names):
        '''Initializes `Entropy` object.
        
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
    

class EntropyDensityProfile:
    '''Calculates the Entropy Density Profile (EDP) as utilized by LncADeep.
    
    Attributes
    ----------
    `feature_names`: `list[str]`
        Names of the features for which the entropy should be calculated.
    `name`: `str`
        Names of entropy density columns calculated by this class.'''
    
    def __init__(self, feature_names):
        '''Initializes `Entropy Density Profile` object.'''

        self.name = [f'EDP({name})' for name in feature_names]
        self.feature_names = feature_names

    def calculate(self, data):
        '''Calculates the EDP for every row in `data`.'''
        print("Calculating Entropy Density Profiles...")
        edps = []
        for _, row in utils.progress(data.df.iterrows()):
            edps.append(self.calculate_edp(
                list(row[self.feature_names].values))
            )
        return edps

    def calculate_edp(self, values):
        '''Calculates the EDP for given list of values'''
        values = np.array(values)
        shannon = entropy(values)
        return (-1/(shannon + 1e-7))*(values*np.log10(values + 1e-7))
    

class GCContent(SequenceBase):
    '''Calculates the proportion of bases that are either Guanine or Cytosine.
    
    Attributes
    ----------
    `apply_to`: `str`
        Indicates which column this class extracts its features from.
    `name`: `str`
        Name of the feature calculated by this object ('GC content')'''
    
    def __init__(self, apply_to='sequence'):
        '''Initializes `GCContent` object.
        
        Arguments
        ---------
        `apply_to`: `str`
            Indicates which column this class extracts its features from.'''
        
        super().__init__(apply_to)
        suffix = '' if apply_to == 'sequence' else f' ({apply_to})'
        self.name = 'GC content' + suffix
    
    def calculate(self, data):
        '''Calculates GC content for every row in `data`.'''
        print("Calculating GC content...")
        self.check_columns(data)
        values = []
        for _, row in utils.progress(data.df.iterrows()):
            values.append(self.calculate_per_sequence(self.get_sequence(row)))
        return values
    
    def calculate_per_sequence(self, sequence):
        '''Calculates the GC content for a given sequence.'''
        return len(re.findall('[CG]', sequence)) / (len(sequence) + 1e-7)
    

class StdStopCodons(SequenceBase):
    '''Calculates the standard deviations of stop codon counts between three
    reading frames, as formulated by lncRScan-SVM.
    
    Attributes
    ----------
    `apply_to`: `str`
        Indicates which column this class extracts its features from.
    `name`: `str`
        Column name of feature calculated by this class ('SCS')

    References
    ----------
    lncRScan-SVM: Sun et al. (2015) https://doi.org/10.1371/journal.pone.0139654
    '''

    def __init__(self, apply_to='sequence'):
        '''Initializes `StdStopCodons` object.
        
        Arguments 
        ---------
        `apply_to`: `str`
            Indicates which column this class extracts its features from.'''
        super().__init__(apply_to)
        self.name = 'SCS'

    def calculate(self, data):
        '''Calculates the std of stop codon counts for every row in `data`.'''
        print("Calculating standard deviation of stop codon counts...")
        stds = []
        self.check_columns(data)
        for _, row in utils.progress(data.df.iterrows()):
            stds.append(self.calculate_per_sequence(self.get_sequence(row)))
        return stds
    
    def calculate_per_sequence(self, sequence):
        '''Calculates the std of stop codon counts for a given `sequence`.'''
        counts = np.zeros(3)
        for frame in range(3):
            for i in range(frame, len(sequence)-3+1, 3):
                if sequence[i:i+3] in ['TAA', 'TAG', 'TGA']:
                    counts[frame] += 1
        return np.std(counts)


class SequenceDistribution(SequenceBase):
    '''For every word in a given vocabulary, calculate the percentage that is 
    contained within every quarter of the total length of the sequence. 
    Loosely based on the D (disrtibution) feature of CTD as proposed by CPPred.
    
    Attributes
    ----------
    `vocabulary`: `dict[str:int]`
        Words of equal length `k` for which to determine distribution for. 
    `k`: `int`
        Inferred lenght of the words in the vocabulary.
    `apply_to`: `str`
        Indicates which column this class extracts its features from.
    `name`: `list[str]`
        List of column names of feature calculated by this class.

    References
    ----------
    CPPred: Tong et al. (2019) https://doi.org/10.1093/nar/gkz087'''

    def __init__(self, apply_to='sequence', vocabulary=['A', 'C', 'G', 'T']):
        '''Initializes `SequenceDistribution` object.
        
        Arguments
        ---------
        `apply_to`: `str`
            Indicates which column this class extracts its features from 
            (default is full transcript).
        `vocabulary`: `list[str]`
            Words of equal length for which to determine distribution for 
            (default is `['A','C','G','T']`).'''

        super().__init__(apply_to)
        self.k = len(vocabulary[0])
        for word in vocabulary:
            if self.k != len(word):
                raise ValueError("Please provide words of even length.")
        self.vocabulary = {word:i for i, word in enumerate(vocabulary)}
        suffix = '' if apply_to == 'sequence' else f' ({apply_to})'
        self.name = [f'distQ{i} {w}{suffix}' for w in vocabulary
                     for i in range(1,5)]

    def calculate(self, data):
        '''Calculates the sequence distribution for every row in `data`.'''
        print("Calculating sequence distribution...")
        dists = []
        self.check_columns(data)
        for _, row in utils.progress(data.df.iterrows()):
            dists.append(self.calculate_per_sequence(self.get_sequence(row)))
        return dists
    
    def calculate_per_sequence(self, sequence):
        '''Calculates the distriution of a given `sequence`.'''
        dist = np.zeros((len(self.vocabulary), 4))
        Q4 = len(sequence)
        Q1, Q2, Q3 = Q4/4, Q4/2, 3*Q4/4 
        for i in range(0, Q4-self.k+1):
            try:
                index = self.vocabulary[sequence[i:i+self.k]]
            except KeyError:
                continue
            if i < Q1: 
                dist[index,0] += 1
            elif i < Q2:
                dist[index,1] += 1
            elif i < Q3:
                dist[index,2] += 1
            else:
                dist[index,3] += 1
        return (dist/(Q4-self.k+1+1e-7)).flatten()
    

class Quality:
    '''Calculates the ratio of uncertain bases (bases other than ACGT) per 
    sequence.
    
    Attributes
    ----------
    `name`: `str`
        Name of the feature calculated by this class ('quality').'''

    def __init__(self):
        '''Initializes `Quality` object.'''
        self.name = 'quality'
    
    def calculate(self, data):
        '''Calculates the quality for all sequences in `data`.'''
        print("Calculating sequence quality...")
        data.check_columns(['sequence'])
        certain = data.df['sequence'].str.count("A|C|T|G")
        length = data.df['sequence'].str.len()
        return (length - certain) / length