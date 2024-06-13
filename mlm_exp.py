import argparse
import torch
from rhythmnblues.features import BytePairEncoding
from rhythmnblues.data import Data
from rhythmnblues import utils
from rhythmnblues.train import train_mlm
from rhythmnblues.train.loggers import LoggerList, LoggerPlot, EarlyStopping, LoggerMLMCounts
from rhythmnblues.modules import MLM, BERT

utils.DEVICE = torch.device("cuda:2")
data_dir = '/data/s2592800/data'

lr = 0.0001
print(utils.DEVICE)

# Loading data and setting tensor features
bpe = BytePairEncoding(f'{data_dir}/features/gencode/4096.bpe')
data = Data(hdf_filepath=f'{data_dir}/tables/gencode_bpe_4096.h5')
data.set_tensor_features(bpe.name, torch.long)
train, valid = data.train_test_split(test_size=0.05, random_state=42)

# Defining and training model
model = MLM(BERT(bpe.vocab_size), pred_batch_size=64).to(utils.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loggers = LoggerList(
  LoggerPlot(f'results/BERT_MLM_exp', ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'Loss']),
  LoggerMLMCounts(bpe.vocab_size, f'results/BERT_MLM_exp/counts.png') 
  # LoggerExperimental(bpe.vocab_size, f'results/BERT_MLM_lr{lr}_vs4096.png'), 
  # EarlyStopping('Accuracy|valid', f'/exports/sascstudent/lromeijn/models/sade.pt'),
)
model, history = train_mlm(model, train, valid, 500, batch_size=16, p_mask=0,
                           p_random=0,
                            optimizer=optimizer, n_samples_per_epoch=5000,
                            logger=loggers)