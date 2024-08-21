import argparse
import torch
from rhythmnblues.features import BytePairEncoding
from rhythmnblues.data import Data
from rhythmnblues import utils
from rhythmnblues.train import train_masked_token_modeling
from rhythmnblues.train.loggers import LoggerList, LoggerPlot, LoggerPrint, LoggerTokenCounts, LoggerWrite
from rhythmnblues.modules import MaskedTokenModel, BERT

epochs = 100
samples_per_epoch = 10000
data_dir = '/exports/sascstudent/lromeijn/data'
results_dir = '/exports/sascstudent/lromeijn/results'
print(utils.DEVICE)

# For duranium
data_dir = '/data/s2592800/data'
results_dir = '/data/s2592800/results'
utils.DEVICE = torch.device('cuda:2')

# CONSTANTS
N = 6
wd = 0
s = 0 # NOTE: doesn't do anything as of current version
ls = 0.1
d = 0
ws = 8000

metric_names = ['Loss','Accuracy','Precision (macro)','Recall (macro)','F1 (macro)']

def mlm_tune(vocab_size, context_length):

  exp_name = f'MLM_vs{vocab_size}_cl{context_length}'

  # Loading data and setting tensor features
  bpe = BytePairEncoding(f'{data_dir}/features/{vocab_size}.bpe', 
                          context_length)
  train = Data([f'{data_dir}/sequences/train_pcrna.fasta', 
                f'{data_dir}/sequences/train_ncrna.fasta'])
  valid = Data([f'{data_dir}/sequences/valid_pcrna.fasta', 
                f'{data_dir}/sequences/valid_ncrna.fasta']).sample(N=2500)
  train.calculate_feature(bpe)
  valid.calculate_feature(bpe)
  train.set_tensor_features(bpe.name, torch.long)
  valid.set_tensor_features(bpe.name, torch.long)

  # Defining and training model
  model = MaskedTokenModel(BERT(bpe.vocab_size), dropout=d, pred_batch_size=8)
  model = model.to(utils.DEVICE)
  loggers = LoggerList(
      LoggerPrint(['Accuracy']),
      LoggerPlot(f'{results_dir}/{exp_name}', metric_names),
      LoggerTokenCounts(bpe.vocab_size, f'{results_dir}/{exp_name}/counts.png'),
      LoggerWrite(f'{results_dir}/{exp_name}/history.csv', metric_names)
  )
  model, history = train_masked_token_modeling(
    model, train, valid, 100, batch_size=8, warmup_steps=ws, 
    n_samples_per_epoch=10000, logger=loggers, s=s, ls=ls
  )

  torch.save(
    model, 
    f'{data_dir}/models/billie_preston_vs{vocab_size}_cl{context_length}.pt'\
  )
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('vocab_size', type=int) # 256, 512, 1024, 2048, 4096
  parser.add_argument('context_length', type=int) # 1024 for now
  args = parser.parse_args()
  mlm_tune(args.vocab_size, args.context_length)