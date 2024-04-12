from rhythmnblues.algorithms import CPAT, CNCI, PLEK, CNIT
from rhythmnblues.data import Data
from rhythmnblues.feature_extraction import Length
from functools import partial
from sklearn.metrics import classification_report, accuracy_score

datasets = {
    'CPAT (train)': [
        'cpat_train_pcrna.fa',
        'cpat_train_ncrna.fa',
    ],
    'CPAT (test)': [
        'cpat_test_pcrna.fa',
        'cpat_test_ncrna.fa',
    ],
    'CNCI (mouse)': [
        'cnci_mouse_test_pcrna.fa',
        'cnci_mouse_test_ncrna.fa',
    ],
    'Gencode': [
        'gencode.v45.pc_transcripts.fa',
        'gencode.v45.lncRNA_transcripts.fa',
    ],
    'RefSeq': [
        'refseq223_pcrna.fasta',
        'refseq223_ncrna.fasta',
    ],
    'Ensembl': [
        'ensembl.GRCh38.cdna.all.fa',
        'ensembl.GRCh38.ncrna.fa',
    ]
}


# Initializing datasets
for name in datasets:
    pf = 'data/sequences/'
    data = Data(pf+datasets[name][0], pf+datasets[name][1]) # Load
    data.calculate_feature(Length()) # Calculate length of sequences
    data.filter_outliers('length', [100,10000]) # Set 100 as min length
    data.filter_outliers('length', 4) # Allow 4 standard deviations from mean
    N = min(data.num_coding_noncoding()) 
    data = data.sample(N,N) # Force majority/minority class to equal #samples
    datasets[name] = data


models = {
    'CPAT': partial(CPAT, 'data/features/fickett_paper.txt'),
    'CNCI': CNCI,
    'PLEK': PLEK,
    'CNIT': CNIT
}


for train_name in datasets:

    train_data = datasets[train_name]

    for model_name in models:

        model = models[model_name](train_data)
        model.fit(train_data)

        for test_name in datasets:
            test_data = datasets[test_name]
            classification = model.predict(test_data)
            print(train_name, model_name, test_name, 
                  accuracy_score(test_data.df['label'], classification))