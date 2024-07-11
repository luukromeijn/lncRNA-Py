import argparse
import torch
from rhythmnblues.data import Data
from rhythmnblues.features import BytePairEncoding
from rhythmnblues.modules import BERT, MLM
from rhythmnblues.train import train_mlm
from rhythmnblues.train.loggers import LoggerList, LoggerPlot, LoggerMLMCounts, LoggerWrite
from rhythmnblues import utils

# DEVICE-SPECIFIC
user_dir = '/exports/sascstudent/lromeijn' # SHARK
# user_dir = '/data1/s2592800' # ALICE
data_dir = f'{user_dir}/data'
results_dir = f'{user_dir}/results'
print(utils.DEVICE)


# CONSTANTS
epochs = 1 # TODO! 250
samples_per_epoch = 1000 # TODO! 10000
context_length = 1024
warmup_steps = 8000
metric_names = ['Loss', 'Accuracy', 'Precision (macro)', 'Recall (macro)', 
                'F1 (macro)']


def tune_mlm_sizes(N, d_model, d_ff, vocab_size, batch_size):

    exp_name = f'MLM_vs{vocab_size}_dm{d_model}_dff{d_ff}_N{N}'

    # Loading the data
    train = Data(hdf_filepath=
        f'{data_dir}/tables/train_vs{vocab_size}_cl{context_length}.h5')
    valid = Data(hdf_filepath=
        f'{data_dir}/tables/valid_vs{vocab_size}_cl{context_length}.h5')
    valid = valid.sample(N=2500)
    
    # Preparing data objects for deep learning
    bpe = BytePairEncoding(f'{data_dir}/features/{vocab_size}.bpe', 
                           context_length)
    train.set_tensor_features(bpe.name, torch.long)
    valid.set_tensor_features(bpe.name, torch.long)
    
    # Initializing a model
    base_arch = BERT(vocab_size, d_model=d_model, d_ff=d_ff, h=8, N=N)
    model = MLM(base_arch, pred_batch_size=batch_size).to(utils.DEVICE)

    # Preparing the loggers
    logger_list = LoggerList(
        LoggerPlot(f'{results_dir}/{exp_name}', metric_names),
        LoggerMLMCounts(bpe.vocab_size, f'{results_dir}/{exp_name}/counts.png'),
        LoggerWrite(f'{results_dir}/{exp_name}/history.csv', metric_names)
    )

    # Training
    model, _ = train_mlm(
        model, train, valid, epochs, batch_size, warmup_steps=warmup_steps,
        n_samples_per_epoch=samples_per_epoch, logger=logger_list
    )

    # Saving the model
    torch.save(model, f'{data_dir}/models/{exp_name}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int) 
    parser.add_argument('d_model', type=int) 
    parser.add_argument('d_ff', type=int) 
    parser.add_argument('vocab_size', type=int)
    parser.add_argument('batch_size', type=int)
    args = parser.parse_args()
    tune_mlm_sizes(args.N, args.d_model, args.d_ff, args.vocab_size, 
                   args.batch_size)