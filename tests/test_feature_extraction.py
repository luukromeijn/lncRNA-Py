import pytest
from rhythmnblues.feature_extraction import *

class TestNoError:
    '''For every feature, simply check whether the calculate method can be run
    without causing an error.'''

    def test_length(self, data):
        feature = Length()
        feature.calculate(data)
    
    @pytest.mark.parametrize('k', range(1,7))
    def test_kmer_freqs(self, data, k):
        feature = KmerFreqs(k)
        feature.calculate(data)

    @pytest.mark.parametrize('k', range(1,7))
    def test_kmer_freqs_plek(self, data, k):
        data.calculate_feature(KmerFreqs(k))
        feature = KmerFreqsPLEK(k)
        feature.calculate(data)

    @pytest.mark.parametrize('k', range(1,7))
    def test_kmer_score(self, data, k):
        feature = KmerScore(data, k)
        feature.calculate(data)

    @pytest.mark.parametrize('dist_type,apply_to,stride',[
        ['euc', 'sequence', 1],
        ['log', 'sequence', 1],
        ['euc', 'ORF', 3],
        ['log', 'ORF', 3],
    ])
    def test_kmer_distance_and_ratio(self, data, dist_type, apply_to, stride):
        data.calculate_feature(ORFCoordinates())
        feature = KmerDistance(data, 3, dist_type, apply_to, stride)
        data.calculate_feature(feature)
        feature = KmerDistanceRatio(3, dist_type, apply_to, stride)
        data.calculate_feature(feature)

    def test_orf_coordinates(self, data):
        feature = ORFCoordinates()
        feature.calculate(data)

    def test_orf_length(self, data):
        data.df[["ORF (start)", 
                 "ORF (end)"]] = np.random.randint(1,99,(len(data.df), 2))
        feature = ORFLength()
        feature.calculate(data)

    def test_orf_coverage(self, data_hdf):
        data_hdf.df['ORF length'] = np.random.randint(1,20, len(data_hdf.df))
        feature = ORFCoverage()
        feature.calculate(data_hdf)

    def test_orf_protein(self, data):
        data.calculate_feature(ORFCoordinates())
        feature = ORFProtein()
        feature.calculate(data)

    def test_orf_protein_analysis(self, data):
        data.calculate_feature(ORFCoordinates())
        data.calculate_feature(ORFProtein())
        feature = ORFProteinAnalysis()
        feature.calculate(data)

    def test_orf_isoelectric(self, data):
        data.calculate_feature(ORFCoordinates())
        data.calculate_feature(ORFProtein())
        feature = ORFIsoelectric()
        feature.calculate(data)

    def test_orf_amino_acids_freqs(self, data):
        data.calculate_feature(ORFCoordinates())
        data.calculate_feature(ORFProtein())
        data.calculate_feature(ORFAminoAcidFreqs())

    def test_fickett(self, data):
        feature = FickettTestcode('tests/data/fickett_paper.txt')
        feature.calculate(data)

    def test_mlcds(self, data):
        feature = MLCDS(data)
        feature.calculate(data)

    def test_mlcds_length(self, data):
        cols = [name for i in range(1,7) for name in 
                [f"MLCDS{i} (start)", f"MLCDS{i} (end)"]]
        data.df[cols] = np.random.randint(1,99,(len(data.df), 12))
        feature = MLCDSLength()
        feature.calculate(data)

    def test_mlcds_length_percentage(self, data):
        cols = [f"MLCDS{i} length" for i in range(1,7)]
        data.df[cols] = np.random.randint(1,99,(len(data.df), 6))
        feature = MLCDSLengthPercentage()
        feature.calculate(data)

    def test_mlcds_score_distance(self, data):
        cols = [f"MLCDS{i} score" for i in range(1,7)]
        data.df[cols] = np.random.random((len(data.df), 6))
        feature = MLCDSScoreDistance()
        feature.calculate(data)

    @pytest.mark.parametrize('k', range(1,7))
    def test_mlcds_kmer_freqs(self, data, k):
        data.calculate_feature(MLCDS(data))
        feature = MLCDSKmerFreqs(k)
        feature.calculate(data)

    def test_mlcds_score_std(self, data):
        cols = [f"MLCDS{i} score" for i in range(1,7)]
        data.df[cols] = np.random.random((len(data.df), 6))
        feature = MLCDSScoreStd()
        feature.calculate(data)

    def test_mlcds_length_std(self, data):
        cols = [f"MLCDS{i} length" for i in range(1,7)]
        data.df[cols] = np.random.randint(1,99,(len(data.df), 6))
        feature = MLCDSLengthStd()
        feature.calculate(data)

    def test_complexity(self, data):
        feature = Complexity()
        feature.calculate(data)

    def test_entropy(self, data):
        cols = [str(i) for i in range(10)]
        data.df[cols] = np.random.random((len(data), 10))
        feature = FeatureEntropy('Test Entropy', cols)
        feature.calculate(data)


def test_kmer_base():
    for i in range(6):
        assert len(KmerBase(i).kmers) == 4**i

@pytest.mark.parametrize('sequence,answers',[
    # ORF subsequence
    ("AATGATGTGAC",     [[ 1,10],[ 1,10],[ 1,10]]), 
    # Double start 
    ("ATGATGTGA",       [[ 0, 9],[ 0, 9],[ 0, 9]]), 
    # No triplets between start and stop
    ("ATGATTGAC",       [[-1,-1],[ 0, 9],[ 2, 8]]), 
    # No stop codon
    ("ATGCCGAG",        [[-1,-1],[ 0, 6],[-1,-1]]),
    # No start codon & double stop
    ("GACGTGACTGA",     [[-1,-1],[-1,-1],[ 2,11]]), 
    # Double start & double stop codon
    ("ATGATGACCTGATGA", [[ 0,12],[ 0,12],[ 0,12]]), 
])
def test_orf_coordinates_special_cases(sequence, answers):
    for relaxation, answer in enumerate(answers):
        feature = ORFCoordinates(min_length=1, relaxation=relaxation)
        p_start, p_end = feature.calculate_per_sequence(sequence)
        assert p_start == answer[0]
        assert p_end == answer[1]
        assert (p_end - p_start) % 3 == 0 # Must always be mutiple of triplet

def test_orf_coordinates_relaxation_3_4(data):
    for r in range(5):
        data.calculate_feature(ORFCoordinates(min_length=1, relaxation=r))
    for i, row in data.df.iterrows():
        length3 = (row['ORF3 (end)'] - row['ORF3 (start)'])
        length4 = (row['ORF4 (end)'] - row['ORF4 (start)'])
        assert length3 % 3 == 0
        assert length4 % 3 == 0
        assert length4 >= length3

def test_orf_lengths_codons(data):
    '''ORF lengths must be divisor of three (because of codons)'''
    for feature in [ORFCoordinates(), ORFLength()]:
        data.calculate_feature(feature)
    for i, row in data.df.iterrows():
        assert row['ORF length'] % 3 == 0

@pytest.mark.parametrize('dir,offset',[(1,0),(1,1),(1,2),(-1,0),(-1,1),(-1,2)])
def test_mlcds_reading_frames(data, dir, offset):
    feature = MLCDS(data)
    for i, row in data.df.iterrows():
        assert (len(feature.get_reading_frame(row['sequence'], dir, offset)) % 3
                == 0)
        
@pytest.mark.parametrize('sequence,truth',[
    ('ACTG', np.array([1,1])),
    ('ACTNG', np.array([1,0])),
    ('ACGACG', np.array([0,0])),
])
def test_count_kmers(sequence, truth):
    assert (count_kmers(sequence, {'ACT':0, 'CTG':1}, k=3) == truth).all()

@pytest.mark.parametrize('nt_seq,aa_seq',[
    ('', '') # Emtpy ORF
])
def test_orf_protein_edge_cases(nt_seq, aa_seq):
    assert ORFProtein().calculate_per_sequence(nt_seq) == aa_seq

@pytest.mark.parametrize('aa_seq',[
    '',  # Empty
    'X', # Unknown amino acid
])
def test_orf_protein_analysis_edge_cases(aa_seq):
    analysis = ORFProteinAnalysis()
    for feature in analysis.calculate_per_sequence(aa_seq):
        assert np.isnan(feature)

def test_orf_column_names():
    assert orf_column_names(['length'], 0)[0] == 'ORF length'
    assert orf_column_names(['length'], 1)[0] == 'ORF1 length'
    assert orf_column_names(['length'], [1,2])[1] == 'ORF2 length'
    assert orf_column_names(['length', 'coverage'],[1,2])[-1] == 'ORF2 coverage'

def test_kmer_freqs_base():
    feature = KmerFreqsBase(3, 'sequence', 1)
    freqs = feature.calculate_kmer_freqs('ACTG')
    assert freqs[feature.kmers['ACT']]*(2+1e-7) == 1
    assert freqs[feature.kmers['AAA']]*(2+1e-7) == 0
    assert freqs[feature.kmers['CTG']]*(2+1e-7) == 1
    feature = KmerFreqsBase(3, 'sequence', 3)
    freqs = feature.calculate_kmer_freqs('ACTG')
    assert freqs[feature.kmers['ACT']]*(1+1e-7) == 1
    assert freqs[feature.kmers['AAA']]*(1+1e-7) == 0
    assert freqs[feature.kmers['CTG']]*(1+1e-7) == 0