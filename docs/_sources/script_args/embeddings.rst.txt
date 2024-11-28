Retrieves sequence embeddings by specified model for input dataset.

::

	python -m lncnrapy.scripts.embeddings [-h] [--output_file OUTPUT_FILE] [--output_plot_file OUTPUT_PLOT_FILE] [--encoding_method {conv,bpe,kmer,nuc}] [--bpe_file BPE_FILE] [--k K] [--pooling {CLS,mean,max}] [--dim_red {tsne,pca,umap,None}] [--batch_size BATCH_SIZE] [--context_length CONTEXT_LENGTH] [--data_dir DATA_DIR] [--results_dir RESULTS_DIR] [--model_dir MODEL_DIR] fasta_file [fasta_file ...] model_file


**Positional arguments:**
  `fasta_file`
           Path to FASTA file of RNA sequences or pair of paths to two FASTA files containing protein- and non-coding RNAs, respectively. (str)
  `model_file`
           (Pre-)trained model to get sequence embeddings from. (str)

**Optional arguments**
  `-h, \-\-help`
    Show help message.
  `\-\-output_file` OUTPUT_FILE
    Name of hdf output file. (str)
  `\-\-output_plot_file` OUTPUT_PLOT_FILE
    If specified, plots the first two dimensions of the (reduced) sequence embeddings and saves them to this file. (str)
  `\-\-encoding_method` {conv,bpe,kmer,nuc}
    Sequence encoding method. (str="conv")
  `\-\-bpe_file` BPE_FILE
    Filepath to BPE model generated with BPE script. Required when Byte Pair Encoding is used. (str="")
  `\-\-k` K
    Specifies k when k-mer encoding is used. (int=6)
  `\-\-pooling` {CLS,mean,max}
    Type of pooling to apply. If "CLS", will extract embeddings from CLS token. (str="mean")
  `\-\-dim_red` {tsne,pca,umap,None}
    Type of dimensionality reduction to apply to retrieved embeddings. If None, will not reduce dimensions. (str=None)
  `\-\-batch_size` BATCH_SIZE
    Number of samples per prediction step. (int=8)
  `\-\-context_length` CONTEXT_LENGTH
    Number of input positions. For cse/k-mer encoding, this translates to a maximum of (768-1)*k input nucleotides. (int=768)
  `\-\-data_dir` DATA_DIR
    Parent directory to use for any of the paths specified in these arguments. (str="")
  `\-\-results_dir` RESULTS_DIR
    Parent directory to use for the results folder of this script. (str="")
  `\-\-model_dir` MODEL_DIR
    Directory where to and load the (pre-)trained model from. (str=f"{data_dir}/models")