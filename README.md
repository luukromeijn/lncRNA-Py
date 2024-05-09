Development package (codename 'rhyhtmnblues') for investigating and classifying
(long non-coding) RNA.

## Conda setup
Following the following order of installation will lead to the least tedious 
and cumbersome virtual environment setup.

    conda create -n rnb -y
    conda activate rnb
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    conda install pytables -c conda-forge -y
    conda install numpy pandas scikit-learn -c conda-forge -y
    conda install matplotlib biopython -c conda-forge -y
    pip install viennarna

Optional:
    
    conda install tqdm xgboost pytest -c conda-forge -y