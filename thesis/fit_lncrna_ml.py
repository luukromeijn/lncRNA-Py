'''Script for fitting feature-based lncRNA machine learning variants lncRNA-LR
(logistic regression) and lncRNA-RF (random forest)'''

import numpy as np
import pandas as pd
import time
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from lncrnapy.data import Data

data_dir = '/exports/sascstudent/lromeijn/data'
table_file = 'finetune_gencode.h5' 

excluded_features = [
    'id', 'label', 'sequence', 'ORF protein', 'SSE', 'quality', 
    'ORF 6-mer logDist pc s=3', 'ORF 6-mer logDist nc s=3', 
    'ORF 6-mer logDistRatio s=3'
]

# Pre-calculated file with GENCODE data and features
train = Data(hdf_filepath=f'{data_dir}/tables/{table_file}')
train.df['label'] = train.df['label'].replace({'pcrna':'pcRNA','ncrna':'ncRNA'})

features1 = np.array([f for f in train.df.columns if f not in excluded_features])
features2 = np.array([f for f in train.df.columns if f not in excluded_features])


# Recursive Feature Selection --------------------------------------------------
class RFE:
    '''Relative Recurisve Feature Selection, selecting a pre-specified amount
    percentage (instead of fixed number of) features at each iteration.'''

    def __init__(self, model_type, reduction_factor=0.75, k_features=10):
        '''Initializes `RFE` object for specified `model_type` ('LR' or 'RF'), 
        `reduction_factor`, and final (target) number of `k_features`.'''

        if model_type not in ['LR', 'RF']:
            raise NotImplementedError("Unsupported model type.")
        self.model_type = model_type
        self.r_f = reduction_factor
        self.k = k_features

    def get_model(self, type):
        '''Returns untrained model (including data imputing/standardization).'''
        return make_pipeline(
            SimpleImputer(missing_values=np.nan), 
            StandardScaler(),
            {'LR': LogisticRegression(class_weight='balanced', max_iter=1000),
             'RF': RandomForestClassifier(max_features='log2', 
                                          class_weight='balanced'),
            }[type]
        )

    def select(self, X, y):
        '''Selects `self.k_features` from columns in `X` through RFE.'''
        features = X.columns
        model = self.get_model(self.model_type)
        model.fit(X,y)
        if self.model_type == 'LR':
            importances = model[2].coef_[0]
        else: # self.model_type == 'RF'
            importances = model[2].feature_importances_
        order = np.argsort(np.abs(importances))[::-1]
        features, importances = features[order], importances[order]
        n = max(int(np.ceil(self.r_f*len(features))), self.k)
        if len(features) > n:
            return self.select(X[features[:n]], y)
        else:
            return model, features, importances

print("Fitting Logistic Regrssion for 10 features...")
rfe = RFE('LR', 0.75, 10)
t0 = time.time()
model, features, importances = rfe.select(train.df[features1], train.df['label'])
print(f"Fitting completed in {time.time()-t0} seconds")
results = pd.DataFrame({'features': features, 'importances': importances})
results.to_csv('lr_importances.csv')
torch.save(model, f'{data_dir}/models/lncRNA-LR.pt')
        
print("Fitting Random Forest for 100 features...")
rfe = RFE('RF', 0.75, 100)
t0 = time.time()
model, features, importances = rfe.select(train.df[features2], train.df['label'])
print(f"Fitting completed in {time.time()-t0} seconds.:")
results = pd.DataFrame({'features': features, 'importances': importances})
results.to_csv('rf_importances.csv')
torch.save(model, f'{data_dir}/models/lncRNA-RF.pt')

# # Random forest only based on 6-mer data
# from lncrnapy.features import KmerFreqs

# data_dir = '/exports/sascstudent/lromeijn/data'
# train = Data([f'{data_dir}/sequences/finetune_gencode_pcrna.fasta',
#               f'{data_dir}/sequences/finetune_gencode_ncrna.fasta',])
# valid = Data([f'{data_dir}/sequences/valid_gencode_pcrna.fasta',
#               f'{data_dir}/sequences/valid_gencode_ncrna.fasta',])

# kmers = KmerFreqs(6)
# train.calculate_feature(kmers)
# valid.calculate_feature(kmers)
# model = make_pipeline(
#     StandardScaler(),
#     RandomForestClassifier(class_weight='balanced', max_features='log2')
# )
# model.fit(train.df[kmers.name], train.df['label'])
# y_pred = model.predict(valid.df[kmers.name])

# print(classification_report(valid.df['label'], y_pred))