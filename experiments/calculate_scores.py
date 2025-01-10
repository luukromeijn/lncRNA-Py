'''Calculates classification metrics (F1, precision, etc.) from output files of
classifiers.'''

import pandas as pd
from lncrnapy.data import Data
from lncrnapy.evaluate import lncRNA_classification_report

results_dir = 'results'
seq_dir = 'data/sequences'
pred_dir = f'{results_dir}/predictions'
scores_dir = f'{results_dir}/scores'

# RNAsamba ---------------------------------------------------------------------
method = 'RNAsamba'
m_pred_dir = f'{pred_dir}/{method.lower()}'

testsets = {
    'GENCODE-RefSeq': ['test_human_pcrna.tsv', 'test_human_ncrna.tsv'],
    'CPAT': ['cpat_test_pcrna.tsv', 'cpat_test_ncrna.tsv'],
    'RNAChallenge': ['RNAChallenge_mRNAs.tsv', 'RNAChallenge_ncRNAs.tsv']
}

for testset in testsets:

    pcrna_file, ncrna_file = testsets[testset]

    rnasamba_ncrna = pd.read_csv(f"{m_pred_dir}/{ncrna_file}", delimiter='\t')
    rnasamba_pcrna = pd.read_csv(f"{m_pred_dir}/{pcrna_file}", delimiter='\t')

    rnasamba_pcrna = rnasamba_pcrna
    rnasamba_pcrna['label'] = 'pcRNA'
    rnasamba_ncrna['label'] = 'ncRNA'

    rnasamba = pd.concat([rnasamba_pcrna, rnasamba_ncrna]).replace(
                    {'coding':'pcRNA', 'noncoding':'ncRNA'})

    print(
        lncRNA_classification_report(
            rnasamba['label'], rnasamba['classification'], method, testset, 
            f'{scores_dir}/{method.lower()}_{testset.lower()}.csv'
        )
    )

# LncFinder --------------------------------------------------------------------
method = 'LncFinder'
m_pred_dir = f'{pred_dir}/{method.lower()}'    

testsets = {
    "GENCODE-RefSeq": ("test_human_pcrna.fasta", "test_human_ncrna.fasta"),
    "CPAT":           ("cpat_test_pcrna.fa", "cpat_test_ncrna.fa"),
    "RNAChallenge":   ("RNAChallenge_mRNAs.fa", "RNAChallenge_ncRNAs.fa")
}

for testset in testsets:
    pcrna, ncrna = testsets[testset]
    data = Data([f'{seq_dir}/{pcrna}', f'{seq_dir}/{ncrna}'])
    pred = pd.concat(
        (pd.read_csv(f'{m_pred_dir}/{pcrna}.csv'),
         pd.read_csv(f'{m_pred_dir}/{ncrna}.csv'))
    ).replace({'Coding': 'pcRNA', "NonCoding": 'ncRNA'})
    assert len(pred) == len(data)
    print(lncRNA_classification_report(
        data.df['label'], pred['Pred'], 'lncFinder', testset, 
        f'{scores_dir}/{method.lower()}_{testset.lower()}.csv'
    ))

# CPAT -------------------------------------------------------------------------
method = 'CPAT'
m_pred_dir = f'{pred_dir}/{method.lower()}'

cutoff = 0.364
testsets = {
    'GENCODE-RefSeq': ['test_human_pcrna.fasta', 'test_human_ncrna.fasta'],
    'CPAT': ['cpat_test_pcrna.fa', 'cpat_test_ncrna.fa'],
    'RNAChallenge': ['RNAChallenge_mRNAs.fa', 'RNAChallenge_ncRNAs.fa']
}

for testset in testsets:

    pcrna_file, ncrna_file = testsets[testset]

    full_pcrna = Data(f'data/sequences/{pcrna_file}').df[['id']]
    full_ncrna = Data(f'data/sequences/{ncrna_file}').df[['id']]
    full_pcrna['id'] = full_pcrna['id'].str.upper()
    full_ncrna['id'] = full_ncrna['id'].str.upper()

    cpat_pcrna = pd.read_csv(
        f'{m_pred_dir}/{testset.lower()}_pcrna.ORF_prob.best.tsv', 
        delimiter='\t')
    cpat_ncrna = pd.read_csv(
        f'{m_pred_dir}/{testset.lower()}_ncrna.ORF_prob.best.tsv', 
        delimiter='\t')

    cpat_pcrna = pd.merge(full_pcrna, cpat_pcrna, how='left', left_on='id', 
                          right_on='seq_ID').fillna(0)
    cpat_ncrna = pd.merge(full_ncrna, cpat_ncrna, how='left', left_on='id', 
                          right_on='seq_ID').fillna(0)

    cpat_pcrna['label'] = 'pcRNA'
    cpat_ncrna['label'] = 'ncRNA'

    prediction = pd.concat([cpat_pcrna, cpat_ncrna])
    prediction['prediction'] = 'pcRNA'
    prediction['prediction'] = prediction['prediction'].where(
        prediction['Coding_prob'] > cutoff, 'ncRNA')

    print(
        lncRNA_classification_report(
            prediction['label'], prediction['prediction'], method, testset, 
            f'{scores_dir}/cpat_{testset.lower()}.csv'
        )
    )

# mRNN -------------------------------------------------------------------------
method = 'mRNN'
m_pred_dir = f'{pred_dir}/{method.lower()}'

testsets = {
    "GENCODE-RefSeq": ("test_human_pcrna.fasta", "test_human_ncrna.fasta"),
    "CPAT":           ("cpat_test_pcrna.fa", "cpat_test_ncrna.fa"),
    "RNAChallenge":   ("RNAChallenge_mRNAs.fa", "RNAChallenge_ncRNAs.fa")
}

for testset in testsets:
    pcrna, ncrna = testsets[testset]
    data = Data([f'{seq_dir}/{pcrna}', f'{seq_dir}/{ncrna}'])
    pred = pd.concat(
        (pd.read_csv(f'{m_pred_dir}/{testset.lower()}_pcrna', sep='\t',
                     header=None, names=['id', 'P(pcRNA)', 'log?']),
         pd.read_csv(f'{m_pred_dir}/{testset.lower()}_ncrna', sep='\t',
                     header=None, names=['id', 'P(pcRNA)', 'log?']))            
    )
    print(pred)
    pred['Class'] = 'pcRNA'
    pred['Class'] = pred['Class'].where(pred['P(pcRNA)']  >= 0.5, 'ncRNA')
    assert len(pred) == len(data)
    print(lncRNA_classification_report(
        data.df['label'], pred['Class'], method, testset, 
        f'{scores_dir}/{method.lower()}_{testset.lower()}.csv'
    ))

# PredLnc-GFStack --------------------------------------------------------------
method = 'PredLnc-GFStack'
m_pred_dir = f'{pred_dir}/{method.lower()}'

testsets = {
    "GENCODE-RefSeq": ("test_human_pcrna.fasta", "test_human_ncrna.fasta"),
    "CPAT":           ("cpat_test_pcrna.fa", "cpat_test_ncrna.fa"),
    "RNAChallenge":   ("RNAChallenge_mRNAs.fa", "RNAChallenge_ncRNAs.fa")
}

for testset in testsets:
    pcrna, ncrna = testsets[testset]
    data = Data([f'{seq_dir}/{pcrna}', f'{seq_dir}/{ncrna}'])
    pred = pd.concat(
        (pd.read_csv(f'{m_pred_dir}/{testset.lower()}_pcrna', sep='\t'),
         pd.read_csv(f'{m_pred_dir}/{testset.lower()}_ncrna', sep='\t'))
    ).replace({'pct':'pcRNA', 'lncRNA':'ncRNA'}) 
    assert len(pred) == len(data)
    print(lncRNA_classification_report(
        data.df['label'], pred['Class'], 'PredLnc-GFStack', testset, 
        f'{scores_dir}/{method.lower()}_{testset.lower()}.csv'
    ))

# LncADeep ---------------------------------------------------------------------
method = 'LncADeep'
m_pred_dir = f'{pred_dir}/{method.lower()}'

testsets = {
    "GENCODE-RefSeq": ("test_human_pcrna.fasta", "test_human_ncrna.fasta"),
    "CPAT":           ("cpat_test_pcrna.fa", "cpat_test_ncrna.fa"),
    "RNAChallenge":   ("RNAChallenge_mRNAs.fa", "RNAChallenge_ncRNAs.fa")
}

for testset in testsets:
    pcrna, ncrna = testsets[testset]
    data = Data([f'{seq_dir}/{pcrna}', f'{seq_dir}/{ncrna}'])
    pred = pd.concat(
        (pd.read_csv(f'{m_pred_dir}/{testset.lower()}_pcrna_LncADeep.results', 
                    sep='\t', index_col=False),
         pd.read_csv(f'{m_pred_dir}/{testset.lower()}_ncrna_LncADeep.results', 
            sep='\t', index_col=False))            
    ).replace({'Coding':'pcRNA', 'Noncoding':'ncRNA'}) 
    assert len(pred) == len(data)
    print(lncRNA_classification_report(
        data.df['label'], pred['Index'], 'LncADeep', testset, 
        f'{scores_dir}/lncadeep_{testset.lower()}.csv'
    ))

# lncRNA-BERT ------------------------------------------------------------------
testsets = ['GENCODE-RefSeq', 'CPAT', 'RNAChallenge']
methods = {
    'lncRNA-BERT': 'kmer_k3',
}

for testset in testsets:
    for method in methods:
        m_pred_dir = f'{pred_dir}/{method.lower()}'
        data = Data(hdf_filepath=f'{m_pred_dir}/' + 
                   f'CLSv2_long_test_{testset.lower()}_{methods[method]}.h5').df
        data['pred'] = 'ncRNA' 
        data['pred'] = data['pred'].where(data['P(pcRNA)'] < 0.5, 'pcRNA')
        print(lncRNA_classification_report(
            data['label'], data['pred'], method, testset, 
            f'results/scores/{method.lower()}_{testset.lower()}.csv'
        ))