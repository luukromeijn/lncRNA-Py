Installation 
============

LncRNA-Py can be installed via GitHub and conda.

:: 
    
    git clone https://github.com/luukromeijn/rhythmnblues.git


Required Dependencies
---------------------

Following the following order of installation will lead to the least tedious 
and cumbersome virtual environment setup.

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
    conda install matplotlib biopython -c conda-forge -y
    pip install umap-learn sentencepiece

Optional Dependencies
---------------------
Some features require the installation of the following dependencies:

::

    pip install viennarna
    conda install tqdm xgboost pytest -c conda-forge -y
    conda install bioconda::blast