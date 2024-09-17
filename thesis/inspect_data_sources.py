'''Produces figures about the data resources uses in this study.'''

from rhythmnblues.data import Data, plot_cross_dataset_violins
from rhythmnblues.features import Length

seq_dir = '/data/s2592800/data/sequences'
datafiles = {
    'GENCODE':      ['gencode.v46.pc_transcripts.fa', 
                     'gencode.v46.lncRNA_transcripts.fa'], 
    'NONCODE':      ['NONCODE.lncRNA.fa'],
    'RefSeq':       ['refseq225_human_pcrna.fasta',
                     'refseq225_human_ncrna.fasta'],
    'RNAcentral':   ['rnacentral_species_specific_ids.fasta'],
    'CPAT':         ['cpat_test_pcrna.fa',
                     'cpat_test_ncrna.fa'],
    'RNAChallenge': ['RNAChallenge_mRNAs.fa', 
                     'RNAChallenge_ncRNAs.fa'], 
}

datasets = {}
for name in datafiles:

    print(name)
    filepaths = [f'{seq_dir}/{filename}' for filename in datafiles[name]]
    
    if name == 'NONCODE' or name == 'RNAcentral':
        data = Data(filepaths[0])
        data.df['label'] = 'ncRNA'
    else:
        data = Data(filepaths)

    data.calculate_feature(Length())
    datasets[name] = data

plot_cross_dataset_violins(list(datasets.values()), list(datasets.keys()), 
                           'length', 'lengths_data_sources.pdf', figsize=(10,4),
                           showextrema=False, showmedians=True)