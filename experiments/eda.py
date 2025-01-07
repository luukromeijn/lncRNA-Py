'''Feature-based Exploratory Data Analysis.'''

import numpy as np
import matplotlib.pyplot as plt
from lncrnapy.data import Data, reduce_dimensionality
from lncrnapy.features import Length, Entropy, KmerFreqs, KmerDistance
from lncrnapy.features import ORFCoordinates, ORFLength, ORFCoverage
from lncrnapy.features import EIIPPhysicoChemical
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Device-specific constants
data_dir = '/data/s2592800/data'

# Loading the data
train = Data([f'{data_dir}/sequences/finetune_gencode_pcrna.fasta',
              f'{data_dir}/sequences/finetune_gencode_ncrna.fasta',])
valid = Data([f'{data_dir}/sequences/valid_gencode_pcrna.fasta',
              f'{data_dir}/sequences/valid_gencode_ncrna.fasta',])


# ORF --------------------------------------------------------------------------
for feature in [ORFCoordinates(), ORFLength(), Length(), ORFCoverage()]:
    train.calculate_feature(feature)

print("No ORFs found in:")
print(len(train.df[(train.df['ORF length']==0) & (train.df['label']=='pcRNA')]), 
      'pcRNAs and',
      len(train.df[(train.df['ORF length']==0) & (train.df['label']=='ncRNA')])
      , 'lncRNAs.')
print(train.test_features(['length', 'ORF length', 'ORF coverage']))

train.plot_feature_density('ORF length','orf_length_density.pdf', figsize=(4,3))
train.plot_feature_density('ORF coverage','orf_coverage_density.pdf', 
                           figsize=(4,3))
train.plot_feature_scatter('length', 'ORF length', xlim=[0,5000], ylim=[0,5000],
                           figsize=(4,3), filepath='orf_length_scatter.pdf')


# K-mer patterns ---------------------------------------------------------------
for k in [3,6]:
    
    # Intializing
    kmer_freqs = KmerFreqs(k)
    
    # Calculating
    for data in [train, valid]:
        data.calculate_feature(kmer_freqs)
    
    # Dimensionality reduction
    valid.df[['L0', 'L1']] = reduce_dimensionality(valid.df[kmer_freqs.name])
    valid.plot_feature_scatter('L0', 'L1', axis_labels=False, 
                               filepath=f'{k}mer_dimred.pdf', figsize=(4,3))
    
    # Fitting
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(class_weight='balanced', max_features='log2',
                               random_state=42)
    )
    model.fit(train.df[kmer_freqs.name], train.df['label'])
    y_pred = model.predict(valid.df[kmer_freqs.name])
    print(classification_report(valid.df['label'], y_pred))

    # Plot density of most important feature
    kmer = model.feature_names_in_[
                             np.argsort(model[1].feature_importances_)[::-1][0]]
    train.plot_feature_density(kmer, f'{kmer}_density.pdf', figsize=(4,3))

    # Significance testing
    print(train.test_features([kmer]))
    print((train.test_features(kmer_freqs.name)['P value']<(0.05/(4**k))).sum())


# Sequence Entropy & Organization ----------------------------------------------
orf_kmer_dist = KmerDistance(train, 6, 'euc', 'ORF', 3)
orf_3mer_freqs = KmerFreqs(3, 'ORF')
orf_3mer_entropy = Entropy('ORF 3-mer entropy', orf_3mer_freqs.name)
eiip = EIIPPhysicoChemical()

for feature in [orf_kmer_dist, orf_3mer_freqs, orf_3mer_entropy, eiip]:
    train.calculate_feature(feature)

train.plot_feature_density('ORF 6-mer eucDist nc s=3', 
                           'ORF_6mer_dist_density.pdf', figsize=(4,3))
train.plot_feature_density('ORF 3-mer entropy', 
                           'ORF_3mer_ent_density.pdf', figsize=(4,3))
train.plot_feature_scatter('ORF 6-mer eucDist nc s=3', 'ORF 3-mer entropy', 
                           filepath='6mer_dist_entropy.pdf', figsize=(4,3),
                           xlim=[0,0.3])

print(train.feature_correlation(['length', 'ORF length', 'ORF 3-mer entropy',
                                 'ORF 6-mer eucDist pc s=3', 
                                 'ORF 6-mer eucDist nc s=3',]))

train.plot_feature_density('EIIP SNR', 'eiip_snr_density.pdf', figsize=(4,3))

for label, idx in zip(['pcRNA', 'ncRNA'], [0,-1]):
    ps = eiip.calculate_power_spectrum(valid.df.iloc[idx]['sequence'])[1:]
    plt.figure(figsize=(4,3))
    plt.plot(np.arange(len(ps)), ps)
    plt.ylim([0,10])
    plt.xlabel('Position')
    plt.ylabel('Power')
    plt.tight_layout()
    plt.savefig(f'eiip_{label}.pdf')
    plt.close()