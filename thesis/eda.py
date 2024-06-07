'''Exploratory Data Analysis'''

from rhythmnblues.data import Data, plot_cross_dataset_violins
from sklearn.decomposition import PCA
from rhythmnblues.features import KmerFreqs, MLCDS


tables_dir = '/exports/sascstudent/lromeijn/data/tables/copy'
results_dir = 'results/eda'
D = results_dir # Abbreviation
P = None # Prefix


# ANT matrix ###################################################################
mlcds = MLCDS(
    '/exports/sascstudent/lromeijn/data/features/gencode/mlcds_ref.txt'
)
mlcds.imshow_ant_matrix(f'{D}/ant_matrix.png')

# CPAT (train)-only plots ######################################################
data = Data(hdf_filepath=f'{tables_dir}/cpat_train.h5')
print(data.plot_feature_density('MFE', f'{D}/density_mfe.png'))

# GENCODE-only plots ###########################################################
data = Data(hdf_filepath=f'{tables_dir}/gencode.h5')

# # K-mer feature spaces
kmer_feats = [
    KmerFreqs(k=3,apply_to='sequence',stride=1).name,
    KmerFreqs(k=3,apply_to='ORF',stride=3).name,
    KmerFreqs(k=6,apply_to='sequence',stride=1).name,
    KmerFreqs(k=1,apply_to='sequence',stride=1).name +
    KmerFreqs(k=2,apply_to='sequence',stride=1).name +
    KmerFreqs(k=2,apply_to='sequence',stride=1,gap_length=1,gap_pos=1).name +
    KmerFreqs(k=2,apply_to='sequence',stride=1,gap_length=2,gap_pos=1).name +
    KmerFreqs(k=3,apply_to='sequence',stride=1).name +
    KmerFreqs(k=6,apply_to='sequence',stride=1).name,
]
kmer_names = ['3mers', '3mers_ORF', '6mers', 'all']

for i, kmers in enumerate(kmer_feats):
    data.plot_feature_space(kmers, 
                            filepath=f'{D}/space_{kmer_names[i]}.png')

# Others
corr_features = [
    'length', 'Fickett score', 'ORF length', 'ORF coverage', 'ORF pI', 
    'GC content', 'A', 'C', 'G', 'T', '3-mer entropy', '3-mer ORF entropy', 
    'EIIP 1_3','Zhang score', '6-mer score', 'MLCDS1 score', 'BLASTX hits', 
]

data.plot_feature_correlation(corr_features, f'{D}/correlation.png')
data.plot_feature_correlation(
    ['Fickett score', '6-mer score', 'Zhang score', 'MLCDS1 score'], 
    f'{D}/correlation_scores.png', figsize=(3.2,2.4)
)

P = D + '/density_'
data.plot_feature_density('Fickett score', f'{P}fickettscore.png')
data.plot_feature_density('Zhang score', f'{P}zhangscore.png')
data.plot_feature_density('6-mer score', f'{P}6merscore.png')
data.plot_feature_density('MLCDS1 score', f'{P}mlcds1score.png')
data.plot_feature_density('BLASTX hits', f'{P}blastx.png')
data.plot_feature_density('ORF 6-mer eucDist pc s=3', f'{P}eucdist_pc.png')
data.plot_feature_density('ORF 6-mer eucDist nc s=3', f'{P}eucdist_nc.png')
data.plot_feature_density('ORF1 coverage', f'{P}orf1_coverage.png')
data.plot_feature_density('3-mer ORF entropy', f'{P}orf_entropy.png')
data.plot_feature_density('TGA (ORF) s=3', f'{P}tga_orf.png')
data.plot_feature_density('BLASTX identity', f'{P}blastx_identity.png')
data.plot_feature_density('BLASTX hit score', f'{P}blastx_identity.png')
data.plot_feature_density('UTR5 coverage', f'{P}utr5_coverage.png')
data.plot_feature_density('G (ORF)', f'{P}G_ORF.png')
data.plot_feature_density('GAAGAG (ORF) s=3', f'{P}GAAGAG_ORF.png')
data.plot_feature_density('MLCDS4 score', f'{P}mlcds4.png')
data.plot_feature_density('TTTGCC (ORF) s=3', f'{P}tttgcc.png')
data.plot_feature_density('GATGTG (ORF) s=3', f'{P}gatgtg.png')
data.plot_feature_density('ATCAGC (ORF) s=3', f'{P}atcatg.png')

P = D + '/scatter_'
data.plot_feature_scatter('BLASTX identity', 'BLASTX hit score', 
                          f'{P}blastx_scatter.png')
data.plot_feature_scatter('6-mer score', 'GC content', f'{P}hexamer_gc.png')
data.plot_feature_scatter('Zhang score', 'Fickett score', 
                          f'{P}zhang_fickett.png')
data.plot_feature_scatter('BLASTX hits', 'ORF1 length', f'{P}blast_orf.png')
data.plot_feature_scatter('ORF length', 'length', f'{P}orfvslength.png')
data.plot_feature_scatter('ORF 6-mer eucDist pc s=3', '3-mer ORF entropy',
                          f'{P}entropystuff.png')
data.plot_feature_scatter('BLASTX hits', 'BLASTX hit score',
                          f'{P}blastx_stuff.png')


# Cross-dataset plots ##########################################################
data = [
    Data(hdf_filepath=f'{tables_dir}/gencode.h5'),
    Data(hdf_filepath=f'{tables_dir}/refseq.h5'),
    Data(hdf_filepath=f'{tables_dir}/noncode-refseq.h5'),
    Data(hdf_filepath=f'{tables_dir}/cpat_train.h5'),
    Data(hdf_filepath=f'{tables_dir}/cpat_test.h5'),
]

for i, dname in enumerate(['gencode', 'refseq', 'noncode']):
    data[i].plot_feature_density('ORF length', f'{D}/density_orf_{dname}.png')
    data[i].plot_feature_density('ORF1 length', f'{D}/density_orf1_{dname}.png')

names = [
    'GENCODE', 
    'RefSeq', 
    'NONCODE', 
    'CPAT (train)', 
    'CPAT (test)',
]

# data[2].df = data[2].df[data[2].df['label'] != 'pcrna'] # For NONCODE

cross_dataset_plot = lambda name, fname: plot_cross_dataset_violins(
    data, names, name, showmeans=True, showextrema=False, 
    filepath=f'{D}/cross_dataset_{fname}.png', figsize=(10,4.8)
)

cross_dataset_plot('length', 'length')
cross_dataset_plot('ORF length', 'orf_length')
cross_dataset_plot('UTR5 length', 'utr5_length')
cross_dataset_plot('UTR3 length', 'utr3_length')
cross_dataset_plot('GC content', 'gc_content')
cross_dataset_plot('CGA (PLEK)', 'cga_plek')
cross_dataset_plot('AAG (ORF) s=3', 'aag_orf')
cross_dataset_plot('BLASTX hit score', 'blastx_hitscore')
cross_dataset_plot('ORF length', 'orf_length')
cross_dataset_plot('ORF1 length', 'orf1_length')
cross_dataset_plot('BLASTX hits', 'blastx_hits')
cross_dataset_plot('EIIP 1_3', 'eiip_1_3')
cross_dataset_plot('EIIP SNR', 'eiip_snr')
cross_dataset_plot('CGA (ORF) s=3', 'cga_orf')
cross_dataset_plot('GATGAA (ORF) s=3', 'gatgaa_orf')
cross_dataset_plot('AATAAA', 'aataaa')

# CPAT validation ##############################################################
from rhythmnblues.algorithms.traditional import CPAT
from sklearn.metrics import classification_report

data = Data(hdf_filepath=f'{tables_dir}/cpat_train.h5')
train_data = data.sample(10000,10000)
test_data = Data(hdf_filepath=f'{tables_dir}/cpat_test.h5')

alg = CPAT(
    '/exports/sascstudent/lromeijn/data/features/fickett_paper.txt',
    '/exports/sascstudent/lromeijn/data/features/cpat_train/6mer_ref.txt'
)

alg.fit(train_data)
pred = alg.predict(test_data)

print(classification_report(test_data.df['label'], pred))

train_data.plot_feature_density('ORF length', f'{D}/cpat_orf_length.png')
train_data.plot_feature_density('ORF coverage', f'{D}/cpat_orf_coverage.png')
train_data.plot_feature_density('Fickett score', f'{D}/cpat_fickett.png')
train_data.plot_feature_density('6-mer score', f'{D}/cpat_hexamer.png')