from rhythmnblues.data import Data
from rhythmnblues.feature_extraction import *

data = Data('data/sequences/gencode.v45.pc_transcripts.fa',
            'data/sequences/gencode.v45.lncRNA_transcripts.fa',
            'data/tables/gencode.v45.test.h5')

train, test = data.train_test_split(test_size=0.2, train_size=0.8)
datasets = {'train': train, 'test':test}

features = [
    Length(),
    KmerFreqs(1),
    KmerFreqs(2),
    KmerFreqs(3),
    KmerFreqs(4),
    KmerFreqs(5),
    KmerScore(train, 6, 'data/features/6mer_bias.txt'),
    ORFCoordinates(),
    ORFLength(),
    ORFCoverage(),
    FickettTestcode('data/features/fickett_paper.txt'),
    MLCDS(train, 'data/features/ANT_matrix.txt'),
    MLCDSLength(),
    MLCDSLengthPercentage(),
    MLCDSScoreDistance(),
    MLCDSKmerFreqs(3),
    MLCDSScoreStd(),
    MLCDSLengthStd()
]

for name in datasets:

    dataset = datasets[name]

    for feature in features:

       dataset.calculate_feature(feature)
       dataset.to_hdf(f'data/tables/gencode.v45.{name}.h5')