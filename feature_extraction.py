import argparse
import numpy as np
from rhythmnblues.features import *
from rhythmnblues.data import Data
from rhythmnblues import utils

utils.watch_progress(on=False) # For SHARK

def feature_extraction(
        dataset_name, pcrna_filepath, ncrna_filepath, tables_folder, 
        features_folder, blastdbs_path='data/blastdbs/uniref90', tmp_folder='', 
        test_mode=False
):
    data = Data([pcrna_filepath, ncrna_filepath])
    data.calculate_feature(Length())
    data.filter_outliers('length', [100, np.inf])
    if test_mode:
        data = data.sample(100, 100)

    features = [
        lambda data: FickettScore(f'{features_folder}/fickett_paper.txt'),
        lambda data: Complexity(),
        lambda data: Quality(),
        lambda data: ORFCoordinates(relaxation=0),
        lambda data: ORFCoordinates(relaxation=1),
        lambda data: ORFCoordinates(relaxation=2),
        lambda data: ORFCoordinates(relaxation=3),
        lambda data: ORFCoordinates(relaxation=4),
        lambda data: ORFLength(relaxation=0),
        lambda data: ORFLength(relaxation=1),
        lambda data: ORFLength(relaxation=2),
        lambda data: ORFLength(relaxation=3),
        lambda data: ORFLength(relaxation=4),
        lambda data: ORFCoverage(relaxation=0),
        lambda data: ORFCoverage(relaxation=1),
        lambda data: ORFCoverage(relaxation=2),
        lambda data: ORFCoverage(relaxation=3),
        lambda data: ORFCoverage(relaxation=4),
        lambda data: ORFProtein(),
        lambda data: ORFProteinAnalysis(),
        lambda data: ORFAminoAcidFreqs(),
        lambda data: UTRLength(),
        lambda data: UTRCoverage(),
        lambda data: GCContent('sequence'),
        lambda data: GCContent('ORF'),
        lambda data: SequenceDistribution('sequence'),
        lambda data: SequenceDistribution('ORF'),
        lambda data: StdStopCodons(),
        lambda data: EIIPPhysicoChemical(),
        lambda data: KmerFreqs(k=1, apply_to='sequence', stride=1),
        lambda data: KmerFreqs(k=1, apply_to='ORF', stride=1),
        lambda data: KmerFreqs(k=2, apply_to='sequence', stride=1),
        lambda data: KmerFreqs(k=2, apply_to='sequence', stride=1, gap_length=1, gap_pos=1),
        lambda data: KmerFreqs(k=2, apply_to='sequence', stride=1, gap_length=2, gap_pos=1),
        lambda data: KmerFreqs(k=2, apply_to='ORF', stride=1),
        lambda data: KmerFreqs(k=3, apply_to='sequence', stride=1),
        lambda data: KmerFreqs(k=3, apply_to='sequence', stride=1, PLEK=True),
        lambda data: KmerFreqs(k=3, apply_to='ORF', stride=3),
        lambda data: KmerFreqs(k=6, apply_to='sequence', stride=1),
        lambda data: KmerFreqs(k=6, apply_to='ORF', stride=3),
        lambda data: Entropy('3-mer entropy', KmerFreqs(k=3, apply_to='sequence', stride=1).name),
        lambda data: Entropy('3-mer ORF entropy', KmerFreqs(k=3, apply_to='ORF', stride=3).name),
        lambda data: EntropyDensityProfile(KmerFreqs(k=2, apply_to='ORF', stride=1).name), 
        lambda data: ZhangScore(data, export_path=f'{features_folder}/{dataset_name}/zhang_ref.txt'), 
        lambda data: KmerScore(data, k=6, apply_to='sequence', export_path=f'{features_folder}/{dataset_name}/6mer_ref.txt'),
        lambda data: MLCDS(data, export_path=f'{features_folder}/{dataset_name}/mlcds_ref.txt'), 
        lambda data: MLCDSLength(),
        lambda data: MLCDSLengthPercentage(),
        lambda data: MLCDSLengthStd(),
        lambda data: MLCDSScoreDistance(),
        lambda data: MLCDSScoreStd(),
        lambda data: BLASTXSearch(blastdbs_path, tmp_folder=f'{tmp_folder}/{dataset_name}', threads=12),
        lambda data: BLASTXBinary(),
        lambda data: SSE(),
        lambda data: UPFrequency(),
        lambda data: KmerDistance(data, 6, 'log', 'ORF', stride=3, export_path=f'{features_folder}/{dataset_name}/6mer_orf_dist_ref.txt'),
        lambda data: KmerDistance(data, 4, 'log', 'acguD', stride=1, alphabet='ACGTD', export_path=f'{features_folder}/{dataset_name}/4mer_acgtd_dist_ref.txt'), 
        lambda data: KmerDistance(data, 3, 'log', 'acgu-ACGU', stride=1, alphabet='ACGTacgt', export_path=f'{features_folder}/{dataset_name}/3mer_acgu_dist_ref.txt'),
    ]

    for feature in features:
        feature = feature(data)
        data.calculate_feature(feature)
        data.to_hdf(f'{tables_folder}/{dataset_name}.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name')
    parser.add_argument('pcrna_filepath')
    parser.add_argument('ncrna_filepath')
    parser.add_argument('tables_folder')
    parser.add_argument('features_folder')
    parser.add_argument('--blastdbs_path', default='data/blastdbs/uniref90')
    parser.add_argument('--tmp_folder', default='')
    parser.add_argument('--test_mode', type=bool, default=False)
    args = parser.parse_args()
    feature_extraction(
        args.dataset_name, args.pcrna_filepath, args.ncrna_filepath, 
        args.tables_folder, args.features_folder, args.blastdbs_path, 
        args.tmp_folder, args.test_mode
    )