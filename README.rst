lncRNA-Py
=========

.. introduction start

lncRNA-Py is a development package for applying machine learning and deep 
learning to the problem of lncRNA classification, i.e. predicting whether a 
novel RNA transcript is coding (mRNA) or long non-coding (lncRNA). 

The main functionalities of the package are: 

* **Library of predictor features**: Re-implementations of a large collection of
  coding potential predictors (e.g. ORF length, pI, Fickett score, k-mer 
  frequencies) used by over 40 existing lncRNA classifiers.
* **lncRNA-BERT**: A Nucleotide Language Model (NLM) pre-trained on human mRNA
  and lncRNA data, achieving state-of-the-art performance when fine-tuned for 
  lncRNA classification.
* **Convolutional Sequence Encoding**: A novel encoding technique that enables
  a more efficient and effective accomodation of long nucleotide sequences in 
  comparison to K-mer Tokenization and Byte Pair Encoding. Alternative encoders
  (NUC, K-mer, BPE) are also implemented.

.. introduction end

Usage
-----

.. usage intro start

There are two ways in which lncRNA-Py can be used: 1) running pre-written
scripts; or 2) via the object-oriented API. The scripts are dedicated to
(pre-)training NLMs, the API provides access to the extensive library of 
re-implemented features.

.. usage intro end

We refer to the `official documentation page <todo.com>`_ for detailed usage 
information and also provide an `example notebook <example.ipynb>`_.

.. about start

About
-----
The lncRNA-Py package was developed by Luuk Romeijn as part of an MSc thesis 
internship project at the Sequencing Analysis Support Core (SASC) at Leiden 
University Medical Center (LUMC), as part of the Computer Science master's 
program at the Leiden Insitute of Advanced Computer Science (LIACS). The 
package accompanies the thesis: `LncRNA-BERT: An RNA Language Model for
Classifying Coding and Long Non-Coding RNA <https://theses.liacs.nl/cs>`_. The
project was supervised by Katy Wolstencroft (LIACS) and Hailiang Mei (LUMC).

.. about end

License
-------
LncRNA-Py is available under the MIT license.