'''Fickett TESTCODE statistic (Fickett et al., 1982).'''

import re
import numpy as np
from rhythmnblues import utils


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