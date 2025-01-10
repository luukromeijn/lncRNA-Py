Trains (or fine-tunes) a model (optionally pre-trained) for lncRNA classification.

::

	python -m lncnrapy.scripts.train [-h] [--exp_prefix EXP_PREFIX] [--pretrained_model PRETRAINED_MODEL] [--encoding_method {cse,bpe,kmer,nuc}] [--epochs EPOCHS] [--n_samples_per_epoch N_SAMPLES_PER_EPOCH] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--d_model D_MODEL] [--N N] [--d_ff D_FF] [--h H] [--dropout DROPOUT] [--hidden_cls_layers HIDDEN_CLS_LAYERS [HIDDEN_CLS_LAYERS ...]] [--n_kernels N_KERNELS] [--kernel_size KERNEL_SIZE] [--bpe_file BPE_FILE] [--k K] [--context_length CONTEXT_LENGTH] [--data_dir DATA_DIR] [--results_dir RESULTS_DIR] [--model_dir MODEL_DIR] [--no_weighted_loss] [--no_random_reading_frame] [--freeze_network] [--freeze_kernels] [--input_linear] [--no_input_relu] fasta_pcrna_train fasta_ncrna_train fasta_pcrna_valid fasta_ncrna_valid


**Positional arguments:**
  `fasta_pcrna_train`
    Path to FASTA file with pcRNA training sequences. (str)
  `fasta_ncrna_train`
    Path to FASTA file with ncRNA training sequences. (str)
  `fasta_pcrna_valid`
    Path to FASTA file with pcRNA sequences used for validating the model after every epoch. (str)
  `fasta_ncrna_valid`
    Path to FASTA file with ncRNA sequences used for validating the model after every epoch. (str)

**Optional arguments**
  `-h, \-\-help`
    Show help message.
  `\-\-exp_prefix` EXP_PREFIX
    Added prefix to model/experiment name. (str)
  `\-\-pretrained_model` PRETRAINED_MODEL
    If specified, fine-tunes this pre-trained model instead of training one from scratch. Note that this causes model-related hyperparameters, such as d_model and N, to be ignored. Specified by id of a model hosted on the HuggingFace Hub, or a path to a local directory containing model weights. (str)=""
  `\-\-encoding_method` {cse,bpe,kmer,nuc}
    Sequence encoding method. (str="cse")
  `\-\-epochs` EPOCHS
    Number of epochs to train for. (int=100)
  `\-\-n_samples_per_epoch` N_SAMPLES_PER_EPOCH
    Number of training samples per epoch. (int=10000)
  `\-\-batch_size` BATCH_SIZE
    Number of samples per optimization step. (int=8)
  `\-\-learning_rate` LEARNING_RATE
    Learning rate used by Adam optimizer. (float=1e-5)
  `\-\-weight_decay` WEIGHT_DECAY
    Weight decay used by Adam optimizer. (float=0.0)
  `\-\-d_model` D_MODEL
    BERT embedding dimensionality. (int=768)
  `\-\-N` N
    Number of BERT transformer blocks. (int=12)
  `\-\-d_ff` D_FF
    Number of nodes in BERT FFN sublayers (int=4*d_model)
  `\-\-h` H
    Number of BERT self-attention heads (int=int(d_model/64))
  `\-\-dropout` DROPOUT
    Dropout probability in CLS output head. (float=0)
  `\-\-hidden_cls_layers` HIDDEN_CLS_LAYERS [HIDDEN_CLS_LAYERS ...]
    Space-separated list with number of hidden nodes in ReLU-activated classification head layers. (int=[])
  `\-\-n_kernels` N_KERNELS
    Specifies number of kernels when convolutional sequence encoding is used. (int=768)
  `\-\-kernel_size` KERNEL_SIZE
    Specifies kernel size when convolutional sequence encoding is used. (int=9)
  `\-\-bpe_file` BPE_FILE
    Filepath to BPE model generated with BPE script. Required when Byte Pair Encoding is used. (str="")
  `\-\-k` K
    Specifies k when k-mer encoding is used. (int=6)
  `\-\-context_length` CONTEXT_LENGTH
    Number of input positions. For cse/k-mer encoding, this translates to a maximum of (768-1)*k input nucleotides. (int=768)
  `\-\-data_dir` DATA_DIR
    Parent directory to use for any of the paths specified in these arguments. (str="")
  `\-\-results_dir` RESULTS_DIR
    Parent directory to use for the results folder of this script. (str="")
  `\-\-model_dir` MODEL_DIR
    Directory where to save the trained model to. Model with highest macro F1-score on the validation dataset is saved. (str=f"{data_dir}/models")
  `\-\-no_weighted_loss` 
    Applies correction to pcRNA/ncRNA class imbalance. (bool)
  `\-\-no_random_reading_frame`
    Turns off sampling in random reading frame for convolutional sequence encoding. (bool)
  `\-\-freeze_network` 
    Freezes all weights from the pre-trained model and bases the clasification on the mean embeddings of this model. This only works with the --pretrained_model flag. (bool)
  `\-\-freeze_kernels` 
    Freezes all convolutional sequence encoding weights from the pre-trained model. Only works with the --pretrained_model flag. (bool)
  `\-\-input_linear` 
    Forces linear projection of kernels onto d_model dimensions in convolutional sequence encoding. (bool)
  `\-\-no_input_relu` 
    Turns off ReLU activation of kernels in convolutional sequence encoding. (bool)