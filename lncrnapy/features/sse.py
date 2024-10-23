'''Feature extractors based on Secondary Structure Elements (SSEs).'''

import re
try:
    import ViennaRNA
except ModuleNotFoundError:
    pass
from lncrnapy import utils

HL_SSE_NAMES = ['acguD', 'acguS', 'acgu-ACGU', 'UP']


def get_hl_sse_sequence(data_row, type):
    '''Returns high-level secondary structure-derived sequence, using the 
    'sequence' and 'SSE' columns of `data_row`. 
    
    Arguments
    ---------
    `data_row`: `pd.Series`
        Row with 'sequence' and 'SSE' columns.
    `type`: 'acguD'|'acguS'|'acgu-ACGU'
        Type of secondary structure-derived sequence.

    References
    ----------
    LNCFinder: Han et al. (2018) https://doi.org/10.1093/bib/bby065'''
    
    if type not in HL_SSE_NAMES:
        raise ValueError()
    
    output = ''
    for i in range(len(data_row['sequence'])):
        if data_row['SSE'][i] == '.':
            if type == 'acguD':
                output += 'D'
            elif type == 'acguS':
                output += data_row['sequence'][i]
            elif type == 'acgu-ACGU':
                output += data_row['sequence'][i].lower()
            elif type == 'UP':
                output += 'U'
        else:
            if type == 'acguD':
                output += data_row['sequence'][i]
            elif type == 'acguS':
                output += 'S'
            elif type == 'acgu-ACGU':
                output += data_row['sequence'][i]
            elif type == 'UP':
                output += 'P'

    return output


class SSE:
    '''Calculates Secondare Elements (SSEs) based on Minimum Free Energy (MFE),
    using the ViennaRNA package, as proposed by LNCFinder.
    
    Attributes
    ----------
    `name`: `list[str]`
        Names features calculated by this object ('MFE', 'SSE').
        
    References
    ----------
    LNCFinder: Han et al. (2018) https://doi.org/10.1093/bib/bby065'''

    def __init__(self):
        '''Initializes `SSE` object.'''
        self.name = ['MFE', 'SSE']

    def calculate(self, data):
        '''Calculates MFE and SSE for every row in `data`.'''
        print("Calculating Secondary Structure Elements...")
        data.check_columns(['sequence'])
        results = []
        for _, row in utils.progress(data.df.iterrows()):
            results.append(self.calculate_per_sequence(row['sequence']))
        return results

    def calculate_per_sequence(self, sequence):
        '''Calculates MFE and SSE for a given `sequence`.'''
        fc = ViennaRNA.fold_compound(sequence)
        sse, mfe = fc.mfe()
        return mfe, str(sse)
    

class UPFrequency:
    '''Calculates the frequency of unpaired nucleotide bases (UP frequency),
    using the SSE, as proposed by LNCFinder.
    
    Attributes
    ----------
    `name`: `str`
        Name of the feature as calculated by this object ('UP freq.')'''

    def __init__(self):
        '''Initializes `UPFrequency` object.'''
        self.name = 'UP freq.'

    def calculate(self, data):
        '''Calculates the UP frequency for every row in `data`.'''
        print("Calculating UP frequencies...")
        results = []
        data.check_columns(['SSE'])
        for _, row in utils.progress(data.df.iterrows()):
            results.append(self.calculate_per_sequence(row))
        return results

    def calculate_per_sequence(self, row):
        '''Calculates the UP frequency of a given data row.'''
        return (len(re.findall('\.', get_hl_sse_sequence(row, 'UP'))) / 
                len(row['sequence']))