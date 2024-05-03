''''Zhang nucleotide bias around start codon, as proposed by DeepCPP (Zhang et 
al. 2020)'''

import numpy as np
from rhythmnblues import utils


class ZhangScore:
    '''Nucleotide bias around the start codon of the ORF, as proposed by
    DeepCPP.

    Attributes
    ----------
    `bias`: `np.ndarray`
        Array containing the nucleotide bias around the start codon.
    `name`:`str`
        Column name of the feature calculated by this class ('Zhang score').
    
    References
    ----------
    DeepCPP: Zhang et al. (2020) https://doi.org/10.1093/bib/bbaa039'''

    def __init__(self, data, export_path=None):
        '''Initializes `ZhangScore` object.
         
        Arguments
        ---------
        `data`: `Data` | `str`
            `Data` object used to calculate nucleotide bias around the start
            codon, or path to file containing this.
        `export_path`: `str`
            Path to save nucleotide bias to for later use (default is None).'''
        
        self._bases = {'A':0, 'C':1, 'G':2, 'T':3}
        labels = {'pcrna': 0, 'ncrna':1}
        self.name = 'Zhang score'

        if type(data) == str:
            self.bias = np.loadtxt(data)
        else:
            print("Initializing Zhang nucleotide bias score...")
            data.check_columns(['ORF (start)'])
            bias = np.zeros((2,4,6))

            # Looping through data
            for _, row in utils.progress(data.df.iterrows()):
                start = row['ORF (start)'] # Start codon coordinate
                if start < 0: # Skip row if no ORF is found
                    continue
                sequence = row['sequence']
                for i, offset in enumerate([-3,-2,-1,3,4,5]):
                    pos = start + offset
                    if pos >= 0 and pos < len(sequence):
                        try: # Increment counter
                            bias[labels[row['label']], 
                                 self._bases[sequence[pos]],i] += 1
                        except KeyError: # For uncertain nucleotides (e.g. N)
                            pass

            # Normalize and take logarithm
            bias[0] = bias[0] / np.sum(bias[0], axis=0)
            bias[1] = bias[1] / np.sum(bias[1], axis=0)
            self.bias = np.log10(bias[0] / (bias[1]+1e-7))

            if export_path is not None:
                np.savetxt(export_path, self.bias, fmt="%.6f", header=
                        "Zhang nucleotide bias for ZhangBias object.\n" + 
                        'Load using ZhangBias(data="<filepath>")')
                
    def calculate(self, data):
        '''Calculates the Zhang nucleotide bias score for every row in `data`.
        '''
        scores = []
        print("Calculating Zhang nucleotide bias score...")
        data.check_columns(['sequence', 'ORF (start)'])
        for _, row in utils.progress(data.df.iterrows()):
            scores.append(self.calculate_per_sequence(row['sequence'], 
                                                      row['ORF (start)']))
        return scores

    def calculate_per_sequence(self, sequence, orf_start):
        '''Calculates the Zhang nucleotide bias score for given `sequence`.'''
        if orf_start < 0:
            return 0 # In case no ORF is found
        score = 0
        for i, offset in enumerate([-3, -2, -1, 3, 4, 5]):
            pos = orf_start + offset
            try:
                score += self.bias[self._bases[sequence[pos]], i]
            except KeyError: # In case of uncertain nucleotides (e.g. N)
                score += 0
        return score