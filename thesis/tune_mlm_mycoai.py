import argparse
import torch
from rhythmnblues.features import BytePairEncoding
from rhythmnblues.data import Data
from rhythmnblues import utils
from rhythmnblues.train import train_masked_token_modeling
from rhythmnblues.train.loggers import LoggerList, LoggerPlot, LoggerPrint, LoggerTokenCounts, LoggerWrite
from rhythmnblues.modules import MaskedTokenModel, BERT

print(utils.DEVICE)
data_dir = '/exports/sascstudent/lromeijn/data'
results_dir = '/exports/sascstudent/lromeijn/results'

# FOR DURANIUM
utils.DEVICE = torch.device('cuda:2')
data_dir = '/data/s2592800/data'
results_dir = '/data/s2592800/results'

# CONSTANTS
vs = 512
N = 6
wd = 0
s = 0
ls = 0.1
d = 0

metric_names = ['Loss','Accuracy','Precision (macro)','Recall (macro)','F1 (macro)']

def mlm_mycoai(warmup_steps):

  exp_name = f'MLM_mycoai_ws{warmup_steps}'

  # Loading data and setting tensor features
  data = Data(f'{data_dir}/sequences/mycoai_trainset.fasta')
  bpe = BytePairEncoding('mycoai.bpe', 256)
  print(bpe.vocab_size)
  data.calculate_feature(bpe)
  data.set_tensor_features(bpe.name, torch.long)
  train, valid = data.train_test_split(test_size=0.001, random_state=42)

  # Defining and training model
  model = MaskedTokenModel(BERT(bpe.vocab_size), dropout=d, pred_batch_size=64)
  model = model.to(utils.DEVICE)
  loggers = LoggerList(
      LoggerPrint(['Accuracy']),
      LoggerPlot(f'{results_dir}/{exp_name}', metric_names),
      LoggerTokenCounts(bpe.vocab_size, f'{results_dir}/{exp_name}/counts.png'),
      LoggerWrite(f'{results_dir}/{exp_name}/history.csv', metric_names)
  )
  model, history = train_masked_token_modeling(model, train, valid, 500, batch_size=16,
                              warmup_steps=warmup_steps, n_samples_per_epoch=10000,
                              logger=loggers, s=s, ls=ls)
  torch.save(model, f'{data_dir}/models/ne-yo.pt')
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('warmup_steps', type=int) # 4000, 8000, 16000, 32000
  args = parser.parse_args()
  mlm_mycoai(args.warmup_steps)