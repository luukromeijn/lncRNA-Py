Installation 
============

LncRNA-Py can be installed via GitHub and 
`conda <https://www.anaconda.com/download/success>`_. Alternatively, one can use
our Google Colab example notebooks [`1 <https://colab.research.google.com/drive/1NSsFYvQQbwhH0yf7wEVfjxyvqG-bUrUS?usp=sharing>`_, 
`2 <https://colab.research.google.com/drive/17yX2LYX5ohe2_dFd1OQi29FjeyeqyzdR?usp=sharing>`_] 
as starting points.

:: 
    
    git clone https://github.com/luukromeijn/lncRNA-Py.git

Required Dependencies
---------------------

The required conda environment can be installed using the provided
:code:`environment.yml` file. 

:: 
    
    conda env create -f environment.yml

Alternatively, one can install the required packages manually. We find that the
following order of installation will lead to the least tedious and cumbersome 
virtual environment setup.

::

    conda create -n lncrna -y
    conda activate lncrna

PyTorch for CPU only: 

::

    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y 

PyTorch for GPU: 

::

    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

We refer to the `PyTorch <https://pytorch.org/get-started/locally/>`_ website 
for the most up-to-date installation procedure. 

Others: 

::

    conda install pytables -c conda-forge -y
    conda install numpy pandas scikit-learn -c conda-forge -y
    conda install matplotlib biopython huggingface_hub -c conda-forge -y
    conda install anaconda::safetensors
    conda install hugging
    pip install umap-learn sentencepiece

Optional Dependencies
---------------------
Some features require the installation of the following dependencies:

::

    pip install viennarna
    conda install tqdm xgboost pytest -c conda-forge -y
    conda install bioconda::blast