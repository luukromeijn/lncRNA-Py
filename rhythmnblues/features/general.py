'''Feature extractors for general features.'''

from Bio.SeqUtils.lcc import lcc_simp
from scipy.stats import entropy
from scipy.fft import fft
from rhythmnblues import utils
from rhythmnblues.features.sse import HL_SSE_NAMES, get_hl_sse_sequence
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
    

class SequenceFeature: 
    '''Base class for features that operate on data (sub)sequences, the type of 
    which is specified by the `apply_to` attribute. 
    
    Sequence-based features in rhyhthmnblues are not required to inherit from 
    this class, but it does make them more versatile as it enables a single 
    implementation to operate on multiple sequence types.
    
    Most important methods are `get_sequence` and `check_columns`, which will 
    behave differently for different `apply_to` settings.
    
    Attributes
    ----------
    `apply_to`: `str`
        Indicates which column this class extracts its features from.'''

    def __init__(self, apply_to='sequence'):
        '''Initializes `SequenceFeature` base class.'''
        if apply_to in ['sequence', 'ORF protein']:
            self.get_sequence = self._get_full_sequence
        elif apply_to in HL_SSE_NAMES:
            self.get_sequence = self._get_hl_sse_sequence
        elif apply_to == 'UTR5':
            self.get_sequence = self._get_utr5_sequence
        elif apply_to == 'UTR3':
            self.get_sequence = self._get_utr3_sequence
        elif apply_to.startswith('MLCDS') or apply_to == 'ORF':
            self.get_sequence = self._get_subsequence
        else: 
            print("Sequence type not known by rhythmnblues.")
            print("Assuming its presence as column in future input data.")
            self.get_sequence = self._get_full_sequence
        self.apply_to = apply_to

    def check_columns(self, data):
        '''Checks if the required columns, based on the `apply_to` attribute,
        are present within `data`.'''
        if self.apply_to in HL_SSE_NAMES:
            data.check_columns(['SSE'])
        elif self.apply_to in ['UTR5', 'UTR3']:
            data.check_columns(['ORF (start)', 'ORF (end)'])
        elif self.apply_to.startswith('MLCDS') or self.apply_to == 'ORF':
            data.check_columns([f'{self.apply_to} ({suffix})'
                                for suffix in ['start', 'end']])
        else: 
            data.check_columns([self.apply_to])
            
    def _get_full_sequence(self, data_row):
        ''''Returns full RNA transcript sequence'''
        return data_row[self.apply_to]
    
    def _get_hl_sse_sequence(self, data_row):
        '''Returns high-level secondary strucutre derived sequence, the type
        of which is determined by the `apply_to` attribute.'''
        return get_hl_sse_sequence(data_row, self.apply_to)
    
    def _get_subsequence(self, data_row):
        '''Returns subsequence based on coordinates in `data_row`, as specified 
        by `apply_to` attribute.'''
        start = int(data_row[f'{self.apply_to} (start)'])
        end = int(data_row[f'{self.apply_to} (end)'])
        dir = 1 if start <= end else -1
        return data_row['sequence'][start:end:dir]
    
    def _get_utr5_sequence(self, data_row):
        '''Returns 5' UTR subsequence.'''
        return data_row['sequence'][:data_row['ORF (start)']]
    
    def _get_utr3_sequence(self, data_row):
        '''Returns 3' UTR subsequence.'''
        return data_row['sequence'][data_row['ORF (end)']:]


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
    

class EIIPPhysicoChemical:
    '''EIIP-derived physico-chemical features, as proposed by LNCFinder. Every 
    sequence is converted into an EIIP representation, of which the power
    spectrum is calculated with a Fast Fourier Transform. Several properties 
    are derived from this power spectrum. 

    Attributes
    ----------
    `name`: `str`
        Names of the EIIP-derived physico-chemical features.
    `eiip_map`: `dict[str:float]`
        Mapping to convert nucleotides into EIIP values.

    References
    ----------
    LNCFinder: Han et al. (2018) https://doi.org/10.1093/bib/bby065'''

    def __init__(self, eiip_map={'A':0.126, 'C':0.134, 'G':0.0806, 'T':0.1335}):
        '''Initializes `EIIPPhysicoChemical` object.
        
        Arguments
        ---------
        `eiip_map`: `dict[str:float]`
            Mapping to convert nucleotides into EIIP values.'''
        
        self.name = ['EIIP 1/3', 'EIIP SNR', 'EIIP Q1', 'EIIP Q2', 'EIIP min', 
                     'EIIP max']
        self.eiip_map = eiip_map
        
    def calculate(self, data):
        '''Calculate EIIP physico-chemical features for every row in `data`.'''
        print("Calculating EIIP-derived physico-chemical features...")
        results = []
        for _, row in utils.progress(data.df.iterrows()):
            results.append(self.calculate_per_sequence(row['sequence']))
        return results

    def calculate_per_sequence(self, sequence):
        '''Calculate EIIP physico-chemical features of given `sequence`.'''
        spectrum = self.calculate_power_spectrum(sequence)

        N = len(spectrum)
        EIIP_onethird = spectrum[int(N/3)] # pcRNA often has peak at 1/3
        EIIP_SNR = EIIP_onethird / np.mean(spectrum) # Signal to noise ratio

        # Quantile statistics of top 10%
        sorted = np.sort(spectrum)[::-1][:int(N/10)] # Top 10% (desc.)
        EIIP_Q1, EIIP_Q2 = np.quantile(sorted, [0.25, 0.5]) 
        EIIP_min, EIIP_max = np.min(sorted), np.max(sorted)

        return EIIP_onethird, EIIP_SNR, EIIP_Q1, EIIP_Q2, EIIP_min, EIIP_max
    
    def calculate_power_spectrum(self, sequence):
        '''Given an RNA `sequence`, convert it to EIIP values and calculate 
        its power spectrum.'''

        # Conversion to EIIP values
        EIIP_values = []
        for base in sequence:
            try: 
                EIIP_values.append(self.eiip_map[base])
            except KeyError:
                EIIP_values.append(np.mean(list(self.eiip_map.values())))

        # Fast fourier transform, obtain power spectrum
        N = int(len(sequence)/3)*3 # Cut off at mod 3
        EIIP_values = EIIP_values[:N]
        return np.abs(fft(EIIP_values)) 
    

class GCContent(SequenceFeature):
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
        return len(re.findall('[CG]', sequence)) / len(sequence)
    

class StdStopCodons(SequenceFeature):
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
