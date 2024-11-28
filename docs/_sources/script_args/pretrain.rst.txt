Pre-training script for a Nucleotide Language Model. Several encoding methods and hyperparameter settings are supported.

::

	python -m lncnrapy.scripts.pretrain [-h] [--exp_prefix EXP_PREFIX] [--encoding_method {conv,bpe,kmer,nuc}] [--epochs EPOCHS] [--n_samples_per_epoch N_SAMPLES_PER_EPOCH] [--batch_size BATCH_SIZE] [--warmup_steps WARMUP_STEPS] [--d_model D_MODEL] [--N N] [--d_ff D_FF] [--h H] [--dropout DROPOUT] [--n_kernels N_KERNELS] [--kernel_size KERNEL_SIZE] [--bpe_file BPE_FILE] [--k K] [--p_mlm P_MLM] [--p_mask P_MASK] [--p_random P_RANDOM]
                          [--context_length CONTEXT_LENGTH] [--data_dir DATA_DIR] [--results_dir RESULTS_DIR] [--model_dir MODEL_DIR] [--mask_size MASK_SIZE] [--no_random_reading_frame] [--input_linear] [--no_input_relu] [--no_output_linear] [--output_relu]
                          fasta_train fasta_valid


**Positional arguments:**
  `fasta_train`
          Path to FASTA file with pre-training sequences. (str)
  `fasta_valid`
          Path to FASTA file with sequences to use for validating model performance after every epoch. (str)

**Optional arguments**
  `-h, \-\-help`
    Show help message.
  `\-\-exp_prefix` EXP_PREFIX
    Added prefix to model/experiment name. (str="MLM")
  `\-\-encoding_method` {conv,bpe,kmer,nuc}
    Sequence encoding method. (str="conv")
  `\-\-epochs` EPOCHS
    Number of epochs to pre-train for. (int=500)
  `\-\-n_samples_per_epoch` N_SAMPLES_PER_EPOCH
    Number of training samples per epoch. (int=10000)
  `\-\-batch_size` BATCH_SIZE
    Number of samples per optimization step. (int=8)
  `\-\-warmup_steps` WARMUP_STEPS
    Number of optimization steps in which learning rate increases linearly. After this amount of steps, the learning rate decreases proportional to the inverse square root of the step number. (int=8)
  `\-\-d_model` D_MODEL
    BERT embedding dimensionality. (int=768)
  `\-\-N` N
    Number of BERT transformer blocks. (int=12)
  `\-\-d_ff` D_FF
    Number of nodes in BERT FFN sublayers (int=4*d_model)
  `\-\-h` H
    Number of BERT self-attention heads (int=int(d_model/64))
  `\-\-dropout` DROPOUT
    Dropout probability in MLM output head. (float=0)
  `\-\-n_kernels` N_KERNELS
    Specifies number of kernels when convolutional sequence encoding is used. (int=768)
  `\-\-kernel_size` KERNEL_SIZE
    Specifies kernel size when convolutional sequence encoding is used. (int=10)
  `\-\-bpe_file` BPE_FILE
    Filepath to BPE model generated with BPE script. Required when Byte Pair Encoding is used. (str="")
  `\-\-k` K
    Specifies k when k-mer encoding is used. (int=6)
  `\-\-p_mlm` P_MLM
    Selection probability per token/nucleotide in MLM. (float=0.15)
  `\-\-p_mask` P_MASK
    Mask probability for selected token/nucleotide. (float=0.8)
  `\-\-p_random` P_RANDOM
    Random replacement chance per token/nucleotide. (float=0.1)
  `\-\-context_length` CONTEXT_LENGTH
    Number of input positions. For cse/k-mer encoding, this translates to a maximum of (768-1)*k input nucleotides. (int=768)
  `\-\-data_dir` DATA_DIR
    Parent directory to use for any of the paths specified in these arguments. (str="")
  `\-\-results_dir` RESULTS_DIR
    Parent directory to use for the results folder of this script. (str="")
  `\-\-model_dir` MODEL_DIR
    Directory where to save pre-trained model to. Model with the highest accuracy on the validation dataset is saved. (str=f"{data_dir}/models")
  `\-\-mask_size` MASK_SIZE
    Number of contiguous nucleotides that make up a mask. (int=1)
  `\-\-no_random_reading_frame`
    Turns off sampling in random reading frame for convolutional sequence encoding (bool)
  `\-\-input_linear` 
    Forces linear projection of kernels onto d_model dimensions in convolutional sequence encoding. (bool)
  `\-\-no_input_relu` 
    Turns off ReLU activation of kernels in convolutional sequence encoding. (bool)
  `\-\-no_output_linear` 
    Forces linear projection of embeddings onto n_kernels dimensions before masked convolution output layer. (bool)
  `\-\-output_relu` 
    Forces ReLU activation of embeddings before masked convolution output layer. (bool)