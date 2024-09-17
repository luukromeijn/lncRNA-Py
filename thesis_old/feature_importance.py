from rhythmnblues.selection import feature_importance_analysis
from rhythmnblues.selection import *


importances, results = feature_importance_analysis(
    trainsets = ['gencode_train', 'refseq_train', 'noncode-refseq_train', 'cpat_train_train'],
    testsets = ['gencode_test', 'refseq_test', 'noncode-refseq_test', 'cpat_train_test', 'cpat_test'],
    k = 10,
    tables_folder = '/exports/sascstudent/lromeijn/data/tables/copy',
    methods = [
        NoSelection,
        TTestSelection,
        RegressionSelection,
        ForestSelection,
        RFESelection,
        MDSSelection,
    ],
    excluded_features = [
        'id', 'label', 'sequence', 'ORF protein', 'SSE', 'quality', 
        'ORF 6-mer logDist pc s=3', 'ORF 6-mer logDist nc s=3', 
        'ORF 6-mer logDistRatio s=3'
    ],
    test = False,
)

results_folder = '/exports/sascstudent/lromeijn/results'
importances.to_csv(f'{results_folder}/importances.csv')
results.to_csv(f'{results_folder}/results.csv')