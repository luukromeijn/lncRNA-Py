'''Trains lncRNA-Py re-implementation of CPAT on the CPAT train set and 
evaluates on CPAT's test set.'''

from lncrnapy.algorithms.traditional import CPAT
from lncrnapy.data import Data
from sklearn.metrics import classification_report

data_dir = '/data/s2592800/data'
data_train = Data([f'{data_dir}/sequences/cpat_train_pcrna.fa',
                   f'{data_dir}/sequences/cpat_train_ncrna.fa',])
data_valid = Data([f'{data_dir}/sequences/cpat_test_pcrna.fa',
                   f'{data_dir}/sequences/cpat_test_ncrna.fa',])

model = CPAT(f'{data_dir}/features/fickett_paper.txt', data_train)
model.fit(data_train)
y_pred = model.predict(data_valid)

print(classification_report(data_valid.df['label'], y_pred))