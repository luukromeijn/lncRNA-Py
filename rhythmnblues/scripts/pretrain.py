'''Pre-training script for a Nucleotide Language Model. Several encoding methods
and hyperparameter settings are supported.

Please call `python -m rhythmnblues.scripts.pretrain --help` for usage info.'''

import argparse
import torch
from functools import partial
from rhythmnblues import utils
from rhythmnblues.data import Data
from rhythmnblues.features import KmerTokenizer, BytePairEncoding
from rhythmnblues.train.loggers import EarlyStopping
from rhythmnblues.train.loggers import LoggerList, LoggerPlot, LoggerWrite
from rhythmnblues.modules import BERT, MotifBERT
from rhythmnblues.modules import MaskedTokenModel, MaskedMotifModel
from rhythmnblues.train import train_masked_token_modeling
from rhythmnblues.train import train_masked_motif_modeling


# TODO: unittests for this function
def pretrain(
        fasta_train, fasta_valid, exp_prefix, encoding_method, epochs, 
        n_samples_per_epoch, batch_size, warmup_steps, d_model, N, d_ff, h, 
        dropout, n_motifs, motif_size, bpe_file, k, p_mlm, p_mask, p_random, 
        context_length, data_dir, results_dir, model_dir, mask_size, 
        random_reading_frame, freeze_motifs, fixed_motifs, project_motifs, 
        activate_motifs, project_embeddings, activate_embeddings, 
    ):
    '''Pre-training function as used in pre-training script. Run 
    `rhythmnblues.scripts.pretrain --help` for usage info.'''

    exp_name = f'{exp_prefix}_{encoding_method}'

    # Import data, subsample valid dataset to save time and resources
    train_data = Data(f'{data_dir}/{fasta_train}') 
    valid_data = Data(f'{data_dir}/{fasta_valid}').sample(N=2500,
                                                          random_state=42)

    # Encoding the data
    if encoding_method in ['nuc', 'kmer', 'bpe']:
        if encoding_method == 'nuc':
            tokenizer = KmerTokenizer(1, context_length)
        elif encoding_method == 'kmer':
            tokenizer = KmerTokenizer(k, context_length)
            exp_name += f'_k{k}'
        else:
            tokenizer = BytePairEncoding(f'{data_dir}/{bpe_file}', 
                                         context_length)
            exp_name += f'_vs{tokenizer.vocab_size}'
        for dataset in [train_data, valid_data]:
            dataset.calculate_feature(tokenizer)
            dataset.set_tensor_features(tokenizer.name, torch.long)
    elif encoding_method == 'motif':
        len_4d_dna = (context_length-1)*motif_size
        for dataset in [train_data, valid_data]:
            dataset.set_tensor_features('4D-DNA', len_4d_dna=len_4d_dna)
        exp_name += f'_nm{n_motifs}_sm{motif_size}'

    # Initializing the model
    if encoding_method in ['nuc', 'kmer', 'bpe']:
        base_arch = BERT(tokenizer.vocab_size, d_model, N, d_ff, h)
        model = MaskedTokenModel(base_arch, dropout, batch_size)
        pretrain_function = train_masked_token_modeling
    elif encoding_method == 'motif':
        base_arch = MotifBERT(
            n_motifs, motif_size, d_model, N, d_ff, h, 
            project_motifs=project_motifs, activate_motifs=activate_motifs, 
            fixed_motifs=fixed_motifs
        )
        # If freeze_motifs: learn the identity function, then freeze weights
        if freeze_motifs: 
            # Initialize model without transformer blocks (N=0)
            model = MotifBERT(n_motifs, motif_size, d_model, N=0, 
                 project_motifs=project_motifs, activate_motifs=activate_motifs)
            model = MaskedMotifModel(
                model, dropout, activate_embeddings=activate_embeddings,
                project_embeddings=False, pred_batch_size=batch_size,
            ).to(utils.DEVICE)
            # Train for 1 epoch on identity function (p_mask=p_random=0)
            print("Initializing motifs by learning the identity function...")
            model, history = train_masked_motif_modeling(
                model, train_data, valid_data, 1, n_samples_per_epoch, 
                batch_size, p_mask=0, p_random=0, warmup_steps=8000
            )
            base_arch.motif_embedder = model.base_arch.motif_embedder
            base_arch.freeze_motifs()
        model = MaskedMotifModel(base_arch, dropout, pred_batch_size=batch_size,
                                 project_embeddings=project_embeddings, 
                                 activate_embeddings=activate_embeddings)
        pretrain_function = partial(train_masked_motif_modeling, 
                                    mask_size=mask_size, 
                                    random_reading_frame=random_reading_frame) 

    # Model/experiment name processing
    exp_name += f'_dm{d_model}_N{N}'
    exp_name = f'{exp_name}_dff{d_ff}' if d_ff is not None else exp_name
    exp_name = f'{exp_name}_h{h}' if h is not None else exp_name
    exp_name += f'_bs{batch_size}_ws{warmup_steps}'
    exp_name += f'_cl{context_length}_d{dropout}'
    exp_name = f'{exp_name}_ms{mask_size}' if mask_size != 1 else exp_name
    exp_name = f'{exp_name}--no_rrf' if not random_reading_frame else exp_name
    exp_name = f'{exp_name}--freeze_motifs' if freeze_motifs else exp_name
    exp_name = f'{exp_name}--fixed_motifs' if fixed_motifs else exp_name
    exp_name = f'{exp_name}--motif_lin' if project_motifs else exp_name
    exp_name = f'{exp_name}--no_motif_relu' if not activate_motifs else exp_name
    exp_name = f'{exp_name}--no_emb_lin' if not project_embeddings else exp_name
    exp_name = f'{exp_name}--emb_relu' if activate_embeddings else exp_name

    # Pre-training the model
    model = model.to(utils.DEVICE) # Send model to GPU
    model, history = pretrain_function(
        model, train_data, valid_data, epochs, n_samples_per_epoch, batch_size,
        p_mlm, p_mask, p_random, warmup_steps, logger=LoggerList(
            LoggerPlot(f'{results_dir}/{exp_name}', ['Loss', 'Accuracy']),
            LoggerWrite(f'{results_dir}/{exp_name}/history.csv', 
                        ['Loss', 'Accuracy']),
            EarlyStopping('Accuracy|valid', 
                          filepath=f'{model_dir}/{exp_name}.pt')
        )
    )

# Arguments
args = {
    'fasta_train': {
        'type': str, 
        'help': 'Path to FASTA file with pre-training sequences. (str)',
    },
    'fasta_valid': {
        'type': str, 
        'help': 'Path to FASTA file with sequences to use for validating model '
                'performance after every epoch. (str)',
    },
    '--exp_prefix': {
        'type': str, 
        'default': 'MLM',
        'help': 'Added prefix to model/experiment name. (str="MLM")',
    },
    '--encoding_method': {
        'type': str,
        'choices': ['motif', 'bpe', 'kmer', 'nuc'],
        'default': 'motif',
        'help': 'Sequence encoding method. (str="motif")'
    },
    '--epochs': {
        'type': int,
        'default': 500,
        'help': 'Number of epochs to pre-train for. (int=500)'
    },
    '--n_samples_per_epoch': {
        'type': int,
        'default': 10000,
        'help': 'Number of training samples per epoch. (int=10000)'
    },
    '--batch_size': {
        'type': int,
        'default': 8,
        'help': 'Number of samples per optimization step. (int=8)'
    },
    '--warmup_steps': {
        'type': int,
        'default': 32000,
        'help': 'Number of optimization steps in which learning rate increases '
                'linearly. After this amount of steps, the learning rate '
                'decreases proportional to the inverse square root of the '
                'step number. (int=8)'
    },
    '--d_model': {
        'type': int,
        'default': 768,
        'help': 'BERT embedding dimensionality. (int=768)'
    },
    '--N': {
        'type': int,
        'default': 12,
        'help': 'Number of BERT transformer blocks. (int=12)'
    },
    '--d_ff': {
        'type': int,
        'default': None,
        'help': 'Number of nodes in BERT FFN sublayers (int=4*d_model)'
    },
    '--h': {
        'type': int,
        'default': None,
        'help': 'Number of BERT self-attention heads (int=int(d_model/12))'
    },
    '--dropout': {                                                              # TODO if this works (or has limited influence, also use this dropout argument in the BERT and MotifBERT's inits)
        'type': float,
        'default': 0,
        'help': 'Dropout probability in BERT model (float=0)'
    },
    '--n_motifs': {
        'type': int,
        'default': 768,
        'help': 'Specifies number of motifs when motif encoding is used. '
                '(int=768)'
    },
    '--motif_size': {
        'type': int,
        'default': 10,
        'help': 'Specifies motif size when motif encoding is used. (int=10)'
    },
    '--bpe_file': {
        'type': str,
        'default': "",
        'help': 'Filepath to BPE model generated with bpe script. Required when'# TODO: update script reference when bpe script is completed.
                ' BPE encoding is used. (str="")'
    },
    '--k': {
        'type': int,
        'default': 6,
        'help': 'Specifies k when k-mer encoding is used. (int=6)'
    },
    '--p_mlm': {
        'type': float,
        'default': 0.15,
        'help': 'Selection probability per token/nucleotide in MLM. '
                '(float=0.15)'
    },
    '--p_mask': {
        'type': float,
        'default': 0.8,
        'help': 'Mask probability for selected token/nucleotide. (float=0.8)'
    },
    '--p_random': {
        'type': float,
        'default': 0.1,
        'help': 'Random replacement chance per token/nucleotide. (float=0.1)'
    },
    '--context_length': {
        'type': int,
        'default': 768,
        'help': 'Number of input positions. For motif/k-mer encoding, this '
                'translates to a maximum of (768-1)*k input nucleotides. '
                '(int=768)'
    },
    '--data_dir': {
        'type': str,
        'default': '.',
        'help': 'Parent directory to use for any of the paths specified in '
                'these arguments. (str="")'
    }, 
    '--results_dir': {
        'type': str,
        'default': '.',
        'help': 'Parent directory to use for the results folder of this script.'
                ' (str="")'
    }, 
    '--model_dir': {
        'type': str,
        'default': '.',
        'help': 'Directory where to save pre-trained model to. Model with the '
                'highest accuracy on the validation dataset is saved. '
                '(str=f"{data_dir}/models")'
    }, 
    '--mask_size': {
        'type': int,
        'default': 1,
        'help': 'Number of contiguous nucleotides that make up a mask. (int=1)'
    }, 
    '--no_random_reading_frame': {
        'action': 'store_true',
        'default': False,
        'help': 'Turns off sampling in random reading frame for motif encoding '
                ' (bool)'
    }, 
    '--freeze_motifs': {
        'action': 'store_true',
        'default': False,
        'help': 'Runs a single epoch of modeling the identity function, then '
                'freezes the motif encoding parameters. (bool)'
    }, 
    '--fixed_motifs': {
        'action': 'store_true',
        'default': False,
        'help': 'Uses a set of predefined kernels. (bool)'                      # TODO improve this documentation
    }, 
    '--project_motifs': {
        'action': 'store_true',
        'default': None,
        'help': 'Forces linear projection of motifs onto d_model dimensions in '
                'motif encoding. (bool)'
    }, 
    '--no_activate_motifs': {
        'action': 'store_true',
        'default': False,
        'help': 'Turns off ReLU activation of motifs in motif encoding. (bool)'
    }, 
    '--no_project_embeddings': {
        'action': 'store_true',
        'default': False,
        'help': 'Forces linear projection of embeddings onto n_motifs ' 
                'dimensions before masked motif output layer. (bool)'
    }, 
    '--activate_embeddings': {
        'action': 'store_true',
        'default': False,
        'help': 'Forces ReLU activation of embeddings before masked motif ' 
                'output layer. (bool)'
    }, 
}


if __name__ == '__main__':

    # Parsing arguments
    p = argparse.ArgumentParser()
    for arg in args:
        p.add_argument(arg, **args[arg])
    p = p.parse_args()
    
    # Argument checks and preprocessing
    if p.encoding_method == 'bpe' and len(p.bpe_file) == 0:
        raise ValueError(
            "Please use --bpe_file flag to specify BPE model file."
        )
    p.model_dir = f'{p.data_dir}/models' if p.model_dir=='.' else p.model_dir
    
    pretrain( # Call
        fasta_train=p.fasta_train, fasta_valid=p.fasta_valid, 
        exp_prefix=p.exp_prefix, encoding_method=p.encoding_method, 
        epochs=p.epochs, n_samples_per_epoch=p.n_samples_per_epoch, 
        batch_size=p.batch_size, warmup_steps=p.warmup_steps, d_model=p.d_model,
        N=p.N, d_ff=p.d_ff, h=p.h, dropout=p.dropout, n_motifs=p.n_motifs, 
        motif_size=p.motif_size, bpe_file=p.bpe_file, k=p.k, p_mlm=p.p_mlm, 
        p_mask=p.p_mask, p_random=p.p_random, context_length=p.context_length, 
        data_dir=p.data_dir, results_dir=p.results_dir, model_dir=p.model_dir,
        mask_size=p.mask_size, 
        random_reading_frame=(not p.no_random_reading_frame),  
        freeze_motifs=p.freeze_motifs, fixed_motifs=p.fixed_motifs, 
        project_motifs=p.project_motifs, activate_motifs=(not p.no_activate_motifs), 
        project_embeddings=(not p.no_project_embeddings), 
        activate_embeddings=p.activate_embeddings, 
    )