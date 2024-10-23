'''Feature extractors based on the Most-Like Coding Sequence (MLCDS)'''

import matplotlib.pyplot as plt
import numpy as np
from lncrnapy import utils
from lncrnapy.features.kmer import KmerBase


class MLCDS:
    '''Determines Most-Like Coding Sequence (MLCDS) coordinates based on 
    Adjoined Nucleotide Triplets (ANT), as proposed by CNCI. Calculates six 
    MLCDSs, based on two directions and three reading frames. The MLCDSs are 
    sorted based on score, where MLCDS1 has the highest score.

    Attributes
    ----------
    `name`: `list[str]`
        Column names for MLCDS coordinates and scores.
    `kmers`: `dict[str:int]`
        Dictionary containing 3-mers (keys) and corresponding indices (values).
    `ant_matrix`: `np.ndarray`
        Adjoined Nucleotide Triplet matrix, containing the log-ratios of ANTs
        appearing in coding over non-coding RNA. 

    References
    ----------
    CNCI: Sun et al. (2013) https://doi.org/10.1093/nar/gkt646
    CNIT: Guo et al. (2019) https://doi.org/10.1093/nar/gkz400''' 

    def __init__(self, data, export_path=None):
        '''Initializes `MLCDS` object, calculates ANT matrix.
        
        Arguments
        ---------
        `data`: `Data`
            `Data` object used to calculate the ANT matrix (trainset), or path
            to file containing this matrix.'''
        
        self.name = [f'MLCDS{i} {feature}' for i in range(1,7) for 
                     feature in ['(start)', '(end)', 'score']]
        self.kmers = KmerBase(3).kmers
        if type(data) == str:
            self.ant_matrix = np.loadtxt(data)
        else:
            self.ant_matrix = self.calculate_ant_matrix(data)
            if export_path is not None:
                np.savetxt(export_path, self.ant_matrix, fmt="%.6f", header=
                        "ANT matrix for MLCDS object.\n" + 
                        'Load using MLCDS(data="<filepath>")')

    def calculate_ant_matrix(self, data):
        '''Calculates Adjoined Nucleotide Triplet matrix, containing the
        log-ratios of ANTs appearing in coding over non-coding RNA.'''

        # Initialize matrix
        print("Calculating ANT matrix...")
        ant_matrix = np.zeros((2, len(self.kmers), len(self.kmers)))

        # Loop through coding and non-coding
        all_seqs = data.df.groupby('label')['sequence']
        all_seqs = all_seqs.apply(lambda x: "!".join(x.tolist()))
        for i, label in enumerate(['pcRNA', 'ncRNA']):
            for p in utils.progress(range(6, len(all_seqs[label])+1)):
                try:
                    i_left_kmer = self.kmers[all_seqs[label][p-6:p-3]]
                    i_right_kmer = self.kmers[all_seqs[label][p-3:p]]
                    ant_matrix[i, i_left_kmer, i_right_kmer] += 1
                except KeyError:
                    continue

        # Calculate log ratio of ANT 
        ant_matrix = np.log2(
            (ant_matrix[0] / ant_matrix[0].sum() + 1e-7) / 
            (ant_matrix[1] / ant_matrix[1].sum() + 1e-7) + 1e-10
        )
        ant_matrix = np.nan_to_num(ant_matrix, nan=0)

        return ant_matrix

    def calculate(self, data):
        '''Calculates MLCDS for every row in `data`.'''
        mlcds = []
        print("Calculating MLCDS...")
        for _, row in utils.progress(data.df.iterrows()):
            mlcds.append(self.calculate_per_sequence(row['sequence']))
        return mlcds

    def calculate_per_sequence(self, sequence):
        '''Calculates MLCDS of `sequence`.'''

        # Calculating MLCDS for six reading frames 
        scores = []
        coordinates = []
        for dir in [1, -1]: # Frame direction
            for offset in range(3): # Frame offset
                frame = self.get_reading_frame(sequence, dir, offset)
                start_, end_, score = self.get_mlcds(frame) 
                # Convert to coordinates of original sequence
                start, end = self.get_abs_coordinates(start_, end_, dir, offset)
                assert sequence[start:end:dir] == frame[start_:end_] # = check
                coordinates.append((start, end))
                scores.append(score)

        # Sorting based on score
        features = []
        sorting = np.argsort(scores)[::-1]
        for i in sorting:
            features.append(coordinates[i][0])
            features.append(coordinates[i][1])
            features.append(scores[i])

        return features

    def get_reading_frame(self, sequence, dir, offset):
        '''Extract reading frame from sequence given direction and offset''' 
        return sequence[::dir][offset:len(sequence)-((len(sequence)-offset)%3)]

    def get_mlcds(self, reading_frame):
        '''Calculates Most-Like Coding Sequence (MLCDS) given reading frame, 
        using the Adjoined Nucleotide Triplet (ANT) matrix. 

        Adapted from `cal_score` function in CNIT's code by Fang Shuangsang.'''

        # Calculating possible optimal cumulative scores
        coden_num = int(len(reading_frame) / 3)
        score = [-1000 for _ in range(coden_num)] # Track cumulative score
        start_list = [0] # Track optimal starting points
        for i in range(1, coden_num): # Loop over codons
            left = reading_frame[(i-1)*3:i*3] # Get hexamer...
            right = reading_frame[i*3:(i+1)*3] # ... & extract ANT matrix score
            try:
                tmp_score = self.ant_matrix[self.kmers[left], self.kmers[right]]
            except KeyError: # In case of non-ACTG bases
                tmp_score = 0
            if score[i - 1] < 0: # If cumulative score < 0
                max_start = (i - 1) # Define new starting point
                start_list.append(max_start)
                score[i] = tmp_score # Start over with cumulative score
            else: # If cumulative score >= 0 
                score[i] = score[i - 1] + tmp_score # Accumulate score
                max_start = start_list[i - 1] # Keep old starting point
                start_list.append(max_start)

        # Find optimal cumulative scores
        max_start, max_end = 0, 0
        max_score = -1000
        for i in range(len(score)): # Loop through scores
            start = start_list[i] * 3
            end = i * 3 + 3
            cur_score = score[i]
            if cur_score > max_score: # If new optimal score is found
                max_score = cur_score # Save as optimal score
                max_start, max_end = start, end # Extract start/end coord.

        return max_start, max_end, max_score
    
    def get_abs_coordinates(self, coord1, coord2, dir, offset):
        '''Transforms a set of coordinates relative to a reading frame to a set
        of coordinates that are defined in relation to the original sequence.'''

        base = -1 if dir == -1 else 0
        coord1 = base + dir*(coord1 + offset)
        coord2 = base + dir*(coord2 + offset)

        return coord1, coord2

    def imshow_ant_matrix(self, filepath=None, figsize=None, **kwargs):
        '''Plots ANT matrix.
        
        Arguments
        ---------
        `filepath`: `str`
            If provided, will export figure to this filepath.
        `kwargs`:
            Any `kwargs` accepted by `matplotlib.pyplot.imshow`.
        '''
        fig, ax = plt.subplots(figsize=figsize)
        plot = ax.imshow(self.ant_matrix, **kwargs)
        ax.set_xticks(np.arange(len(self.kmers)), self.kmers, fontsize=5, 
                      rotation=90)
        ax.set_yticks(np.arange(len(self.kmers)), self.kmers, fontsize=5)
        fig.colorbar(plot)
        fig.tight_layout()
        if filepath is not None:
            fig.savefig(filepath)
        return fig
    
    
class MLCDSLength:
    '''Lengths of Most-Like Coding Sequences.
    
    Attributes
    ----------
    `name`: `list[str]`
        Column name for MLCDS lengths'''

    def __init__(self):
        '''Initializes `MLCDSLength` object.'''
        self.name = [f'MLCDS{i} length' for i in range(1,7)]

    def calculate(self, data):
        '''Calculates MLCDS lengths for all rows in `data`.'''
        start_columns = [f'MLCDS{i} (start)'  for i in range(1,7)]
        end_columns = [f'MLCDS{i} (end)'  for i in range(1,7)]
        data.check_columns(start_columns)
        data.check_columns(end_columns)
        return np.absolute(data.df[end_columns].values - 
                           data.df[start_columns].values) 
    

class MLCDSLengthPercentage:
    '''Length percentages of Most-Like Coding Sequences. Defined as the length
    of the MLCDS with the highest score, divided by the sum of the lengths of
    the remaining MLCDSs. 
    
    Attributes
    ----------
    `name`: `list[str]`
        Column name for MLCDS length percentage ('MLCDS length-percentage')'''

    def __init__(self):
        '''Initializes `MLCDSLengthPercentage` object.'''
        self.name = 'MLCDS length-percentage'

    def calculate(self, data):
        '''Calculates length percentage for every row in `data`.'''
        data.check_columns([f'MLCDS{i} length'  for i in range(1,7)])
        return (
            data.df['MLCDS1 length'] / 
            np.sum(data.df[[f'MLCDS{i} length'  for i in range(2,7)]], axis=1)
        ) 
    

class MLCDSScoreDistance:
    '''Score distance of Most-Like Coding Sequences. Defined as sum of the 
    differences between the score of the highest MLCDS and that of the the 
    remaining ones. 
    
    Attributes
    ----------
    `name`: `str`
        Column name for MLCDS score distance ('MLCDS score-distance')'''

    def __init__(self):
        '''Initializes `MLCDSScoreDistance` object.'''
        self.name = 'MLCDS score-distance'

    def calculate(self, data):
        '''Calculates score distance for every row in `data`.'''
        data.check_columns([f'MLCDS{i} score'  for i in range(1,7)])
        return (
            np.sum(
                [data.df['MLCDS1 score'] - data.df[f'MLCDS{i} score'] 
                 for i in range(2,7)], axis=0
            ) / 5
        )


class MLCDSScoreStd:
    '''Standard deviation of Most-Like Coding Sequences scores.
    
    Attributes
    ----------
    `name`: `str`
        Column name for MLCDS score standard deviation ('MLCDS score (std)')'''

    def __init__(self):
        '''Initializes `MLCDSScoreStd` object.'''
        self.name = 'MLCDS score (std)'

    def calculate(self, data):
        '''Calculates MLCDS score standard deivation for all rows in `data`.'''
        columns = [f'MLCDS{i} score'  for i in range(1,7)]
        data.check_columns(columns)
        return data.df[columns].std(axis=1)
    

class MLCDSLengthStd:
    '''Standard deviation of Most-Like Coding Sequences lengths.
    
    Attributes
    ----------
    `name`: `str`
        Column name for MLCDS length standard deviation ('MLCDS length (std)')
    '''

    def __init__(self):
        '''Initializes `MLCDSScoreStd` object.'''
        self.name = 'MLCDS length (std)'

    def calculate(self, data):
        '''Calculates MLCDS length standard deivation for all rows in `data`.'''
        columns = [f'MLCDS{i} length'  for i in range(1,7)]
        data.check_columns(columns)
        return data.df[columns].std(axis=1)