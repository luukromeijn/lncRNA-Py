Development package lncRNA-Py for investigating and classifying
(long non-coding) RNA.

## Conda setup
Following the following order of installation will lead to the least tedious 
and cumbersome virtual environment setup.

    conda create -n lncrna -y
    conda activate lncrna

PyTorch for CPU only: 

    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y 

PyTorch for GPU: 

    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

Others: 

    conda install pytables -c conda-forge -y
    conda install numpy pandas scikit-learn -c conda-forge -y
    conda install matplotlib biopython -c conda-forge -y
    pip install umap-learn sentencepiece

Optional:
    
    pip install viennarna
    conda install tqdm xgboost pytest -c conda-forge -y
    conda install bioconda::blast