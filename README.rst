.. |graph_abstr| image:: experiments/figures/graphical_abstract.jpg
    :width: 100%

lncRNA-Py
=========

.. introduction start

lncRNA-Py is a development package for applying machine learning and deep 
learning to the problem of lncRNA classification, i.e. predicting whether a 
novel RNA transcript is coding (mRNA) or long non-coding (lncRNA).

The main functionalities of the package are: 

* **lncRNA-BERT**: A Nucleotide Language Model (NLM) pre-trained on human mRNA
  and lncRNA data, achieving state-of-the-art performance when fine-tuned for 
  lncRNA classification.
* **Convolutional Sequence Encoding**: A novel encoding technique that enables
  a more efficient and effective accomodation of long nucleotide sequences in 
  comparison to K-mer Tokenization and Byte Pair Encoding. Alternative encoders
  (NUC, K-mer, BPE) are also implemented.
* **Library of predictor features**: Re-implementations of a large collection of
  coding potential predictors (e.g. ORF length, pI, Fickett score, k-mer 
  frequencies) used by over 40 existing lncRNA classifiers.

.. introduction end

Usage
-----

.. usage intro start

There are two ways in which lncRNA-Py can be used: 1) running pre-written
scripts; or 2) via the object-oriented API. The scripts are dedicated to
(pre-)training NLMs, the API provides access to the extensive library of 
re-implemented features.

.. usage intro end

We refer to the `official documentation page 
<https://luukromeijn.github.io/lncRNA-Py/>`_ for detailed usage information 
and also provide two example notebooks, one for using our 
`pre-written scripts <https://colab.research.google.com/drive/1NSsFYvQQbwhH0yf7wEVfjxyvqG-bUrUS?usp=sharing>`_, 
and a second one that introduces the `API <https://colab.research.google.com/drive/17yX2LYX5ohe2_dFd1OQi29FjeyeqyzdR?usp=sharing>`_.

.. publication start

Publication: lncRNA-BERT
------------------------
|graph_abstr|

LncRNA-Py accompanies the paper: `LncRNA-BERT: An RNA Language Model for
Classifying Coding and Long Non-Coding RNA 
<https://doi.org/10.1101/2025.01.09.632168>`_. Scripts and 
descriptions of additional experiments that relate to this study are
provided in the `experiments 
<https://github.com/luukromeijn/lncRNA-Py/tree/master/experiments>`_ folder.

.. publication end

.. license start

License
-------
LncRNA-Py is available under the MIT license.

.. license end
