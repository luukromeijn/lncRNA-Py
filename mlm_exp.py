# wd d s ls
# default = (0, 0, 0, 0.1)

# wd, d, s, ls = default
# for wd in (wd, 0.0001, 0.00001):
#     print(wd, d, s, ls)

# wd, d, s, ls = default
# for d in (d, 0.1, 0.25):
#     print(wd, d, s, ls)

# wd, d, s, ls = default
# for s in (s, 0.5, 0.75, 1.0):
#     print(wd, d, s, ls)

# wd, d, s, ls = default
# for ls in (ls, 0.1, 0.25):
#     print(wd, d, s, ls)

# exit()

import argparse
import torch
from rhythmnblues.features import BytePairEncoding
from rhythmnblues.data import Data
from rhythmnblues import utils
from rhythmnblues.train import train_mlm
from rhythmnblues.train.loggers import LoggerList, LoggerPlot, LoggerPrint, LoggerMLMCounts
from rhythmnblues.modules import MLM, BERT

print(utils.DEVICE)
data_dir = '/exports/sascstudent/lromeijn/data'
results_dir = '/exports/sascstudent/lromeijn/results'

# FOR DURANIUM
data_dir = '/data/s2592800/data'
results_dir = 'results'

# CONSTANTS
lr = 0.0001
vs = 512
N = 6

def mlm_exp(wd, d, s, ls):

    exp_name = f'MLM_wd{wd}_d{d}_s{s}_ls{ls}'
    print(exp_name)

    # Loading data and setting tensor features
    bpe = BytePairEncoding(f'{data_dir}/features/gencode/{vs}.bpe')
    data = Data(hdf_filepath=f'{data_dir}/tables/gencode_bpe_{vs}.h5')
    data.set_tensor_features(bpe.name, torch.long)
    train, valid = data.train_test_split(test_size=0.05, random_state=42)

    # Defining and training model
    model = MLM(BERT(bpe.vocab_size), dropout=d, pred_batch_size=64)
    model = model.to(utils.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loggers = LoggerList(
        LoggerPrint(['Accuracy']),
        LoggerPlot(
          f'{results_dir}/{exp_name}', 
          ['Loss','Accuracy','Precision (macro)','Recall (macro)','F1 (macro)']
        ),
        LoggerMLMCounts(bpe.vocab_size, f'{results_dir}/{exp_name}/counts.png'),
    )
    model, history = train_mlm(model, train, valid, 500, batch_size=32,
                                optimizer=optimizer, n_samples_per_epoch=10000,
                                logger=loggers, s=s, ls=ls)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wd', type=float)
    parser.add_argument('d', type=float)
    parser.add_argument('s', type=float)
    parser.add_argument('ls', type=float)
    args = parser.parse_args()
    mlm_exp(args.wd, args.d, args.s, args.ls)