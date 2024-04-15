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

models = {
    'CPAT': lambda dat: CPAT('data/features/fickett_paper.txt', dat),
    'CNCI': lambda dat: CNCI(dat),
    'PLEK': lambda dat: PLEK(),
    'CNIT': lambda dat: CNIT(dat),
}

def get_dataset(name):
    pf = 'data/sequences/'
    pc_filepath, nc_filepath = datasets[name]
    data = Data(pf+pc_filepath, pf+nc_filepath) # Load
    data.calculate_feature(Length()) # Calculate length of sequences
    data.filter_outliers('length', [100,10000]) # Set 100 as min length
    data.filter_outliers('length', 4) # Allow 4 stds from mean
    N = min(data.num_coding_noncoding()) 
    return data.sample(N,N,seed=42) # Force class balance

for train_name in datasets:

    train_data = get_dataset(train_name)

    for model_name in models:

        model = models[model_name](train_data)
        model.fit(train_data)

        for test_name in datasets:

            test_data = get_dataset(test_name)
            classification = model.predict(test_data)
            score = accuracy_score(test_data.df['label'], classification)
            print(train_name, model_name, test_name, score)
            file = open('output.txt', 'a')
            file.writelines([f'{train_name} {model_name} {test_name} {score}\n'])
            file.close()