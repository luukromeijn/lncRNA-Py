'''Feature extractors that perform a BLAST database search.'''

import os
import subprocess
import numpy as np
import pandas as pd
from rhythmnblues import utils


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


class BLASTXBinary:
    '''Calculates whether or not the number of BLASTX hits surpasses a preset 
    threshold.
    
    Attributes
    ----------
    `threshold`: `int`
        Minimum number of BLASTX hits to be surpassed to return True (default is
        0).
    `name`: `str`
        Name of `BLASTXBinary` feature ('BLASTX hits > {threshold}')
    '''

    def __init__(self, threshold=0):
        '''Initializes `BLASTXBinary` object.
        
        Arguments
        ---------
        `threshold`: `int`
            Minimum number of BLASTX hits to be surpassed to return True 
            (default is 0).
        '''
        self.name = f'BLASTX hits > {threshold}'
        self.threshold = threshold

    def calculate(self, data):
        '''Calculates the BLASTX binary feature for every row in `data`'''
        data.check_columns(['BLASTX hits'])
        return (data.df['BLASTX hits'] > self.threshold).astype(int)