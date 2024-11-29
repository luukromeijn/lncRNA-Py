'''Feature extractors that perform a BLAST database search.'''

import os
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import entropy
from lncrnapy import utils


# NOTE Currently not included in unittests because of blast dependency
class BLASTXSearch:
    '''Derives features based on a BLAST database search:
    * BLASTX hits: number of blastx hits found for given transcript.
    * BLASTX hit score: mean of the mean log evalue over each of the three 
    reading frames (as proposed by CPC). 
    * BLASTX frame score: mean of the deviation from the hit score over each of
    the three reading frames (as proposed by CPC). 
    * BLASTX S-score: sum of the logs of the significant scores (as proposed by
    PLncPro)
    * BLASTX bit score: total bit score (as proposed by PLncPro)
    * BLASTX frame entropy: Shannon entropy of probabilities that hits are in
    the ith frame (as proposed by PLncPro)
    * BLASTX identity: sum of the identity percentage for each of the blast hits
    for a given sequence.
    
    Attributes
    ----------
    `reference`: `str`
        Reference to use for BLASTX feature extraction, usually the path to a 
        local BLAST database. When the provided string ends with '.csv', will 
        assume that it refers to a saved BLASTX output from a finished run. Note
        that all other arguments will then be ignored. Alternatively, when
        running remotely (`remote=True`), this argument should correspond to the
        name of an official BLAST database.
    `remote`: `bool`
        Whether to run remotely or locally. If False, requires a local
        installation of BLAST with a callable blastx program (default is False).
    `evalue`: `float`
        Cut-off value for statistical significance (default is 1e-10).
    `strand`: ['both'|'plus'|'minus']
        Which reading direction(s) to consider (default is 'plus').
    `threads`: `int`
        Specifies how many threads for BLAST to use (when running locally).
    `tmp_folder`: `str`
        Path to folder where temporary FASTA and output files will be saved.
    `name`: `list[str]`
        Column names for MLCDS length standard deviation ('MLCDS length (std)')

    References
    ----------
    CPC: Kong et al. (2007) https://doi.org/10.1093/nar/gkm391
    PLncPro: Singh et al. (2017) https://doi.org/10.1093/nar/gkx866'''

    def __init__(self, reference, remote=False, evalue=1e-10, strand='plus', 
                 threads=None, output_dir='', save_results=False):
        '''Initializes `BLASTXSearch` object. 
        
        Arguments
        ---------
        `reference`: `str`
            Reference to use for BLASTX feature extraction, usually the path to
            a local BLAST database. When the provided string ends with '.csv', 
            will assume that it refers to a saved BLASTX output from a finished
            run. Note that all other arguments will then be ignored. 
            Alternatively, when running remotely (`remote=True`), this argument
            should correspond to the name of an official BLAST database.
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
        `output_dir`: `str`
            Path to folder where temporary FASTA and output files will be saved
            (default is '').
        `save_results`: `bool`
            If True, will not delete BLASTX output after calculation (default is
            False)'''

        self.reference = reference
        self.remote = remote
        self.evalue = evalue
        self.strand = strand
        self.threads = threads
        self.output_dir = output_dir
        self.save_results = save_results
        self.name = ['BLASTX hits', 'BLASTX hit score', 'BLASTX frame score',
                     'BLASTX S-score', 'BLASTX bit score',
                     'BLASTX frame entropy', 'BLASTX identity']

    def calculate(self, data):
        '''Calculates BLASTX database search features for all rows in `data`.'''

        if self.reference.endswith('csv'):
            output = self.read_blastx_output(self.reference)
        else:
            output = self.run_blastx(data)
        columns = output.columns

        print("Calculating blastx scores...")
        results = []
        output = output.groupby(by='query acc.ver')
        for i, row in utils.progress(data.df.iterrows()):
            # Assumes query ids are same as row ids (error sensitive...?)
            try:
                group = output.get_group(row['id'])
            except KeyError:
                group = pd.DataFrame(columns=columns)
            results.append(self.calculate_per_sequence(group))

        return results
    
    def calculate_per_sequence(self, blast_result):
        '''Calculate BLASTX features for given query result'''
        
        s_score = np.sum(-np.log10(blast_result['evalue'] + 1e-250))
        bit_score = np.sum(blast_result['bit score'])
        identity = blast_result['identity'].sum()

        # Mean log evalue per reading frame
        S = np.array([np.mean(-np.log10(
            blast_result[blast_result['q. start'] % 3 == i]['evalue'] + 1e-250
            )) for i in range(3)])

        if np.isnan(S).sum() == 3: # Can't take mean of empty array
            hit_score, frame_score = np.nan, np.nan
        else:
            hit_score = np.nanmean(S)
            frame_score = np.nanmean((hit_score - np.array(S))**2)

        frame_entropy = entropy(
            [(blast_result['q. start'] % 3 == i).sum()/(len(blast_result)+1e-7)
             for i in range(3)]
        )

        # If still nan (no hits), set scores to... 
        hit_score = np.nan_to_num(hit_score, nan=0) #... 0 (no hits)
        frame_score = np.nan_to_num(frame_score, nan=0) # ... 0 (no hits)
        # ... maximum entropy value (but this is questionable, hence comment)
        # frame_entropy = np.nan_to_num(frame_entropy, 
        #                               nan=entropy([1/3,1/3,1/3]))

        return [len(blast_result), hit_score, frame_score, s_score, bit_score,
                frame_entropy, identity]
    
    def read_blastx_output(self, out_filepath):
        '''Reads a BLAST .csv output file (outfmt 10) with good column names.'''
        return pd.read_csv(out_filepath, header=0, names=[ # Read dataframe
            'query acc.ver', 'subject acc.ver', 'identity', 'alignment length',
            'mismatches', 'gap opens', 'q. start', 'q. end', 's. start', 
            's. end', 'evalue', 'bit score']
        )

    def run_blastx(self, data):
        '''Runs a BLASTX database search for all rows in `data`.'''

        fasta_filepath = f'{self.output_dir}/temp.fasta'
        out_filepath = f'{self.output_dir}/BLASTX_results.csv'
        data.to_fasta(fasta_filepath) # Generate FASTA query file

        # Generate command based on object configuration
        command = ['blastx', '-query', fasta_filepath, '-strand', self.strand, 
                   '-db', self.reference, '-out', out_filepath, '-outfmt', 
                   str(10), '-evalue', str(self.evalue)]
        if self.remote: 
            command += ['-remote']
        elif self.threads is not None:
            command += ['-num_threads', str(self.threads)]

        print("Running blastx...")
        subprocess.run(command, check=True) # Run the command
        output = self.read_blastx_output(out_filepath)
        os.remove(fasta_filepath) # Remove FASTA query file
        if not self.save_results:
            os.remove(out_filepath) # Remove BLASTX output (is in memory now)

        return output


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
    

# NOTE Currently not included in unittests because of blast dependency
class BLASTNSearch:
    '''Performs BLASTN database search, returns max BLASTN identity per query.
    
    Attributes
    ----------
    `reference`: `str`
        Reference to use for BLASTX feature extraction, usually the path to a 
        local BLAST database. When the provided string ends with '.csv', will 
        assume that it refers to a saved BLASTX output from a finished run. Note
        that all other arguments will then be ignored. Alternatively, when
        running remotely (`remote=True`), this argument should correspond to the
        name of an official BLAST database.
    `remote`: `bool`
        Whether to run remotely or locally. If False, requires a local
        installation of BLAST with a callable blastx program (default is False).
    `strand`: ['both'|'plus'|'minus']
        Which reading direction(s) to consider (default is 'plus').
    `threads`: `int`
        Specifies how many threads for BLAST to use (when running locally).
    `tmp_folder`: `str`
        Path to folder where temporary FASTA and output files will be saved.
    `name`: `list[str]`
        Column names for extracted features.'''

    def __init__(self, reference, remote=False, strand='plus', threads=None, 
                 output_dir='', save_results=False):
        '''Initializes `BLASTNSearch` object. 
        
        Arguments
        ---------
        `reference`: `str`
            Reference to use for BLASTX feature extraction, usually the path to
            a local BLAST database. When the provided string ends with '.csv', 
            will assume that it refers to a saved BLASTX output from a finished
            run. Note that all other arguments will then be ignored. 
            Alternatively, when running remotely (`remote=True`), this argument
            should correspond to the name of an official BLAST database.
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
        `output_dir`: `str`
            Path to folder where temporary FASTA and output files will be saved
            (default is '').
        `save_results`: `bool`
            If True, will not delete BLASTX output after calculation (default is
            False)'''

        self.reference = reference
        self.remote = remote
        self.strand = strand
        self.threads = threads
        self.output_dir = output_dir
        self.save_results = save_results
        self.name = ['BLASTN max identity', 'BLASTN alignment length']

    def calculate(self, data):
        '''Calculates BLASTX database search features for all rows in `data`.'''

        if self.reference.endswith('csv'):
            output = self.read_blastn_output(self.reference)
        else:
            output = self.run_blastn(data)
        columns = output.columns

        print("Calculating blastn scores...")
        results = []
        output = output.groupby(by='query acc.ver')
        for i, row in utils.progress(data.df.iterrows()):
            # Assumes query ids are same as row ids (error sensitive...?)
            try:
                group = output.get_group(row['id'])
            except KeyError:
                group = pd.DataFrame(columns=columns)
            results.append(group.sort_values(by='identity', ascending=False)[
                                ['identity', 'alignment length']
                           ].head(1).max().tolist())

        return np.nan_to_num(results, nan=0)
    
    def read_blastn_output(self, out_filepath):
        '''Reads a BLAST .csv output file (outfmt 10) with good column names.'''
        return pd.read_csv(out_filepath, header=0, names=[ # Read dataframe
            'query acc.ver', 'subject acc.ver', 'identity', 'alignment length',
            'mismatches', 'gap opens', 'q. start', 'q. end', 's. start', 
            's. end', 'evalue', 'bit score']
        )

    def run_blastn(self, data):
        '''Runs a BLASTX database search for all rows in `data`.'''

        fasta_filepath = f'{self.output_dir}/temp.fasta'
        out_filepath = f'{self.output_dir}/BLASTN_results.csv'
        data.to_fasta(fasta_filepath) # Generate FASTA query file

        # Generate command based on object configuration
        command = ['blastn', '-query', fasta_filepath, '-strand', self.strand, 
                   '-db', self.reference, '-out', out_filepath, '-outfmt', 
                   str(10)]
        if self.remote: 
            command += ['-remote']
        elif self.threads is not None:
            command += ['-num_threads', str(self.threads)]

        print("Running blastn...")
        subprocess.run(command, check=True) # Run the command
        output = self.read_blastn_output(out_filepath)
        os.remove(fasta_filepath) # Remove FASTA query file
        if not self.save_results:
            os.remove(out_filepath) # Remove BLASTX output (is in memory now)

        return output
    

class BLASTNCoverage:
    '''Alignment length / sequence length'''

    def __init__(self):
        '''Initializes `BLASTNcoverage` object.'''
        self.name = 'BLASTN coverage'

    def calculate(self, data):
        '''Calculates the BLASTN coverage feature for every row in `data`'''
        data.check_columns(['length', 'BLASTN alignment length'])
        return data.df['BLASTN alignment length'] / data.df['length']