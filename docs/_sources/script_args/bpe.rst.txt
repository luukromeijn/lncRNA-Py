Fits a Byte Pair Encoding (BPE) model to a dataset.

::

	python -m lncnrapy.scripts.bpe [-h] [--bpe_file BPE_FILE] [--data_dir DATA_DIR] fasta_train vocab_size


**Positional arguments:**
  `fasta_train`
         Path to FASTA file of RNA sequences to be used for fitting the BPE model. (str)
  `vocab_size`
          Pre-defined number of tokens in vocabulary. (str)

**Optional arguments**
  `-h, \-\-help`
    Show help message.
  `\-\-bpe_file` BPE_FILE
    Name of BPE output file. (str=f"features/{vocab_size}.bpe")
  `\-\-data_dir` DATA_DIR
    Parent directory to use for any of the paths specified in these arguments. (str="")