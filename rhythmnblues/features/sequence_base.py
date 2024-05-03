'''Contains base class for features that operate on data (sub)sequences.'''

from rhythmnblues.features.sse import HL_SSE_NAMES, get_hl_sse_sequence


class SequenceBase: 
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
        '''Initializes `SequenceBase` base class.'''
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