'''Feature extractors that identify and rely on the Open Reading Frame (ORF) in
a transcript.'''

import re
import numpy as np
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from rhythmnblues import utils
from rhythmnblues.features import Length


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