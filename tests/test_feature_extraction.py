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

def test_kmer_base():
    for i in range(6):
        assert len(KmerBase(i).kmers) == 4**i

@pytest.mark.parametrize('sequence,start,end',[
    ("AATGATGTGAC", 1, 10), # ORF subsequence
    ("ATGATGTGA", 0, 9), # Double start codon presence
    ("ATGATTGA", -1, -1), # No triplets between start and stop
    ("ATGATGAGA", -1, -1), # No stop codon
    ("ATGATGACCTGATGA", 0, 12), # Double start & double stop codon
])
def test_orf_coordinates_special_cases(sequence, start, end):
    feature = ORFCoordinates(min_length=1)
    p_start, p_end = feature.calculate_per_sequence(sequence)
    assert p_start == start
    assert p_end == end

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