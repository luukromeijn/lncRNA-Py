import argparse
import numpy as np
from lncrnapy.features import *
from lncrnapy.data import Data
from lncrnapy import utils

utils.watch_progress(on=False) # For SHARK

def feature_extraction_fit(
        dataset_name, pcrna_filepath, ncrna_filepath, tables_folder, 
        features_folder, blast_folder, test_mode=False
):
    data = Data([pcrna_filepath, ncrna_filepath])
    if test_mode:
        data = data.sample(100, 100)

    features = [
        lambda data: Length(),
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
        lambda data: KmerDistance(data, 6, 'euc', 'ORF', stride=3, export_path=f'{features_folder}/{dataset_name}/6mer_orf_dist_ref_euc.txt'),
        lambda data: BLASTXSearch(f'{blast_folder}/uniref90', output_dir='/tmp'),
        lambda data: BLASTXBinary()
    ]

    for feature in features:
        feature = feature(data)
        data.calculate_feature(feature)
        data.to_hdf(f'{tables_folder}/{dataset_name}.h5')


def feature_extraction_predict(
        dataset_name, pcrna_filepath, ncrna_filepath, tables_folder, 
        features_folder, blast_folder, test_mode=False
):
    data = Data([pcrna_filepath, ncrna_filepath])
    data.calculate_feature(Length())
    # data.filter_outliers('length', [100, 10000])
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
        lambda data: ZhangScore(f'{features_folder}/finetune_gencode/zhang_ref.txt'), 
        lambda data: KmerScore(f'{features_folder}/finetune_gencode/6mer_ref.txt', k=6, apply_to='sequence'),
        lambda data: MLCDS(f'{features_folder}/finetune_gencode/mlcds_ref.txt'), 
        lambda data: MLCDSLength(),
        lambda data: MLCDSLengthPercentage(),
        lambda data: MLCDSLengthStd(),
        lambda data: MLCDSScoreDistance(),
        lambda data: MLCDSScoreStd(),
        lambda data: KmerDistance(f'{features_folder}/finetune_gencode/6mer_orf_dist_ref_euc.txt', 6, 'euc', 'ORF', stride=3),
        lambda data: BLASTXSearch(f'{blast_folder}/uniref90', output_dir='/tmp'),
        lambda data: BLASTXBinary()
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
    parser.add_argument('blast_folder')
    parser.add_argument('--test_mode', type=bool, default=False)
    parser.add_argument('--predict', action='store_true', default=False)
    args = parser.parse_args()
    if parser.predict:
        feature_extraction = feature_extraction_predict
    else:
        feature_extraction = feature_extraction_fit
    feature_extraction(
        args.dataset_name, args.pcrna_filepath, args.ncrna_filepath, 
        args.tables_folder, args.features_folder, args.blast_folder,
        args.test_mode
    )