'''Contains feature extractor classes that can calculate several features of RNA
sequences, such as Most-Like Coding Sequences and nucleotide frequencies. 

Every feature extractor class contains:
* A `name` attribute of type `str`, indicating what name a `Data` column for
this feature will have.
* A `calculate` method with a `Data` object as argument, returning a list or
array of the same length as the `Data` object.'''

import itertools
import matplotlib.pyplot as plt
import numpy as np
import re
from tqdm import tqdm


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


class KmerBase:
    '''Base class for k-mer-based feature extractors.
    
    Attributes
    ----------
    `k`: `int`
        Length of to-be-generated nucleotide combinations in the vocabulary.
    `uncertain`: `str`
        Optional character that indicates any base that falls outside of ACGT.
    `k-mers`: `dict[str:int]`
        Dictionary containing k-mers (keys) and corresponding indices (values).
    '''

    def __init__(self, k, uncertain=''):
        '''Initializes `KmerBase` object. 
        
        Arguments
        ---------
        `k`: `int`
            Length of to-be-generated nucleotide combinations in the vocabulary.
        `uncertain`: `str`
            Optional character that indicates any base that falls outside of 
            ACGT (default is `''`).'''
        
        self.k = k
        self.uncertain = uncertain
        self.kmers = {
            ''.join(list(kmer)):i for i, kmer in enumerate(
                itertools.product('ACGT' + uncertain, repeat=self.k)
            )
        }

    # Decide if we wan't to keep this (currently not used)
    def replace_uncertain_bases(self, sequence):
        '''Replaces non-ACGT bases in `sequence` with `self.uncertain`.'''
        
        if self.uncertain != '':
            return re.sub('[^ACGT]', self.uncertain, sequence)
        else: 
            return sequence
    

class KmerFreqs(KmerBase):
    '''For every k-mer, calculate its occurrence frequency in the sequence
    divided by the total number of k-mers appearing in that sequence.
    
    Attributes
    ----------
    `k`: `int`
        Length of to-be-generated nucleotide combinations in the vocabulary.
    `uncertain`: `str`
        Optional character that indicates any base that falls outside of ACGT.
    `k-mers`: `dict[str:int]`
        Dictionary containing k-mers (keys) and corresponding indices (values).
    `name`: `list[str]`
        Column names for frequency features (= all k-mers).'''

    def __init__(self, k):
        '''Initializes `KmerFreqs` object.
        
        Arguments
        ---------
        `k`: `int`
            Length of to-be-generated nucleotide combinations in the vocabulary.
        '''
        super().__init__(k)
        self.name = list(self.kmers.keys())

    def calculate(self, data):
        '''Calculates k-mer frequencies for every row in `data`.'''
        print(f"Calculating {self.k}-mer frequencies...")
        freqs = []
        for _, row in tqdm(data.df.iterrows()):
            freqs.append(self.calculate_per_sequence(row['sequence'])) 
        return np.stack(freqs)
    
    def calculate_per_sequence(self, sequence):
        '''Calculates k-mer frequencies of `sequence`.'''
        sequence = self.replace_uncertain_bases(sequence)
        freqs = np.zeros(len(self.kmers))
        for i in range(len(sequence)-self.k+1): 
            try:
                freqs[self.kmers[sequence[i:i+self.k]]] += 1
            except KeyError:
                pass
        freqs = freqs/(freqs.sum()+1e-7)
        return freqs


class KmerFreqsPLEK:
    '''Scales k-mer frequencies by 1/(4^(5-k)), compensating for small k-mers 
    occuring more often than large k-mers. Calculates based on original k-mer 
    frequencies (does not recalculate k-mer frequencies.)
    
    Attributes
    ----------
    `k`: `int`
        Length of to-be-generated nucleotide combinations in the vocabulary.
    `name`: `list[str]`
        Column names for frequency features (= all k-mers).
        
    References
    ----------
    PLEK: Li et al. (2014) https://doi.org/10.1186/1471-2105-15-311'''

    def __init__(self, k):
        '''Initializes `KmerFreqsPLEK` object.
        
        Arguments
        ---------
        `k`: `int`
            Length of nucleotide combinations in the vocabulary.'''
        self.k = k 
        self.name = [f'{kmer} (PLEK)' for kmer in KmerBase(k).kmers]

    def calculate(self, data):
        '''Scales k-mer frequencies for every row in `data`.'''
        kmers = list(KmerBase(self.k).kmers)
        data.check_columns(kmers)
        return 1/(4**(5-self.k)) * data.df[kmers]
    

class KmerScore(KmerBase):
    '''Calculates k-mer score, indicating how likely a sequence is to be 
    protein-coding (the higher, the more likely). Sums the log-ratios of k-mer
    frequencies in protein-coding RNA over non-coding RNA. Introduced by CPAT 
    as hexamer score or hexamer usage bias. 

    Attributes
    ----------
    `k`: `int`
        Length of to-be-generated nucleotide combinations in the vocabulary.
    `uncertain`: `str`
        Optional character that indicates any base that falls outside of ACGT.
    `k-mers`: `dict[str:int]`
        Dictionary containing k-mers (keys) and corresponding indices (values).
    `name`: `list[str]`
        Column names for frequency features (= all k-mers).
    `kmer_freqs`: `np.ndarray`
        Log-ratios of k-mer frequencies in protein-coding RNA over non-coding 
        RNA. 
       
    References
    ----------
    CPAT: Wang et al. (2013) https://doi.org/10.1093/nar/gkt006'''

    def __init__(self, data, k, export_path=None):
        '''Initializes `KmerScore` object, calculates log-ratios of k-mer 
        frequencies in protein-coding RNA over non-coding RNA.
        
        Arguments
        ---------
        `data`: `Data` | `str`
            `Data` object containing sequences for calculating k-mer frequency
            bias (training set), or path to file that contains these values. 
        `k`: `int`
            Length of to-be-generated nucleotide combinations in the vocabulary.
        `export_path`: `str`
            Path to save k-mer frequency bias matrix to for later use.'''
        
        super().__init__(k)
        self.name = f'{k}-mer score'
        if type(data) == str:
            self.kmer_freqs = np.loadtxt(data)
        else:
            print(f"Initializing {self.k}-mer score...")
            kmer_freqs = np.zeros((len(self.kmers), 2))
            all_seqs = data.df.groupby('label')['sequence']
            all_seqs = all_seqs.apply(lambda x: "!".join(x.tolist()))
            for j, label in enumerate(['pcrna', 'ncrna']):
                kmer_freqs[:,j] = count_kmers(all_seqs[label],self.kmers,self.k)
            kmer_freqs = kmer_freqs / kmer_freqs.sum(axis=0) + 1e-7
            kmer_freqs = kmer_freqs[:,0] / kmer_freqs[:,1] + 1e-10
            kmer_freqs = np.log(kmer_freqs)
            kmer_freqs = np.nan_to_num(kmer_freqs, nan=0)
            self.kmer_freqs = kmer_freqs
            if export_path is not None:
                np.savetxt(
                    export_path, self.kmer_freqs, header=f'{k}-mer ' + 
                    'frequency bias matrix for KmerScore object.\n' +  
                    f'Load using KmerScore(data="<filepath>", k={k})', 
                    fmt="%.6f"
                )

    def calculate(self, data):
        '''Calculates k-mer score for every row in `data`.'''
        print(f"Calculating {self.k}-mer scores...")
        scores = []
        for _, row in tqdm(data.df.iterrows()):
            scores.append(self.calculate_per_sequence(row['sequence']))
        return scores
    
    def calculate_per_sequence(self, sequence):
        '''Calculates k-mer score of `sequence`.'''
        sequence = self.replace_uncertain_bases(sequence)
        score = 0
        for i in range(len(sequence)-self.k+1):
            try: 
                score += self.kmer_freqs[self.kmers[sequence[i:i+self.k]]]
            except KeyError:
                pass
        return score / (len(sequence)-self.k+1)
    
    def plot_bias(self, filepath=None):
        '''Plots log ratio of usage frequency of k-mers in pcRNA/ncRNA.'''
        fig, ax = plt.subplots()
        plot = ax.bar(np.arange(len(self.kmers)), self.kmer_freqs)
        ax.set_xticks(np.arange(len(self.kmers)), self.kmers, fontsize=5, 
                      rotation=90)
        ax.set_ylabel('Frequency coding/non-coding (log ratio)')
        fig.tight_layout()
        if filepath is not None:
            fig.savefig(filepath)
        return fig
    

class ORFCoordinates:
    '''Determines Open Reading Frame (ORF) coordinates.
    
    Attributes
    ----------
    `name`: `list[str]`
        Column names for ORF coordinates ('ORF (start)', 'ORF (end)').
    `min_length`: `int`
        Minimum required length for an ORF.'''

    def __init__(self, min_length=30):
        '''Initializes `ORFCoordinates` object.'''
        self.min_length = min_length
        self.name = ['ORF (start)', 'ORF (end)']

    def calculate(self, data):
        '''Calculates ORF for every row in `data`.'''
        print("Finding Open Reading Frames...")
        orfs = []
        for _, row in tqdm(data.df.iterrows()):
            orfs.append(self.calculate_per_sequence(row['sequence']))
        return orfs
    
    def calculate_per_sequence(self, sequence): 
        '''Returns start (incl.) and stop (excl.) position of longest ORF in
        `sequence`.'''

        start_codons, stop_codons = [], []
        for i in range(len(sequence)-2): # Loop through sequence
            codon = sequence[i:i+3]
            if codon == 'ATG': # Store positions of start/stop codons
                start_codons.append(i)
            elif codon in ['TAA', 'TAG', 'TGA']:
                stop_codons.append(i+3)

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
    `name`: `list[str]`
            Column names for ORF length ('ORF length').'''

    def __init__(self):
        '''Initializes `ORFLength` object.'''
        self.name = 'ORF length'

    def calculate(self, data):
        '''Calculates ORF length for every row in `data`.'''
        data.check_columns(['ORF (end)', 'ORF (start)'])  
        return data.df['ORF (end)'] - data.df['ORF (start)'] 
    

class ORFCoverage:
    '''Calculates ORF coverage (ORF length / sequence length).

    Attributes
    ----------
    `name`: `list[str]`
            Column names for ORF length ('ORF length').'''    

    def __init__(self):
        '''Initializes `ORFCoverage` object.'''
        self.name = 'ORF coverage'

    def calculate(self, data):
        '''Calculates ORF coverage for every row in `data`.'''
        data.check_columns(['ORF length', 'length']) 
        return data.df['ORF length'] / data.df['length']
    

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
        for _, row in tqdm(data.df.iterrows()):
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
    

# NOTE: should the ANT matrix also be calculated using the different orientation?
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
        for i, label in enumerate(['pcrna', 'ncrna']):
            for p in tqdm(range(6, len(all_seqs[label])+1)):
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
        for _, row in tqdm(data.df.iterrows()):
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

    def imshow_ant_matrix(self, filepath=None, **kwargs):
        '''Plots ANT matrix.
        
        Arguments
        ---------
        `filepath`: `str`
            If provided, will export figure to this filepath.
        `kwargs`:
            Any `kwargs` accepted by `matplotlib.pyplot.imshow`.
        '''
        fig, ax = plt.subplots()
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
    

class MLCDSKmerFreqs(KmerFreqs):
    '''K-mer Frequencies of the Most-Like Coding Sequence with the highest
    score. 
    
    Attributes
    ----------
    `name`: `list[str]`
        Column name for MLCDS lengths
    `k`: `int`
        Length of to-be-generated nucleotide combinations in the vocabulary.
    `uncertain`: `str`
        Optional character that indicates any base that falls outside of ACGT.
    `k-mers`: `dict[str:int]`
        Dictionary containing k-mers (keys) and corresponding indices (values).
    `name`: `list[str]`
        Column names for frequency features (= all k-mers).'''

    def __init__(self, k):
        '''Initializes `MLCDSKmerFreqs` object.
        
        Arguments
        ---------
        `k`: `int`
            Length of to-be-generated nucleotide combinations in the vocabulary.
        '''
        super().__init__(k)
        self.name = [f'{kmer} (MLCDS)' for kmer in self.name]

    def calculate(self, data):
        '''Calculates MLCDS k-mer frequencies for every row in `data`.'''
        print(f"Calculating MLCDS {self.k}-mer frequencies...")
        data.check_columns(['MLCDS1 (start)', 'MLCDS1 (end)'])
        freqs = []
        for _, row in tqdm(data.df.iterrows()):
            start, end = int(row['MLCDS1 (start)']), int(row['MLCDS1 (end)'])
            dir = 1 if start < end else -1
            mlcds = row['sequence'][start:end:dir]
            freqs.append(self.calculate_per_sequence(mlcds)) 
        return np.stack(freqs)


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
    

def count_kmers(sequence, kmers, k):
    '''Returns an array of frequencies k-mer counts in `sequence`. Uses k-mer 
    indices as defined by dictionary.'''

    counts = np.zeros(max(kmers.values())+1)

    for i in tqdm(range(k,len(sequence)+1)):
        try:
            counts[kmers[sequence[i-k:i]]] += 1
        except KeyError:
            continue

    return counts