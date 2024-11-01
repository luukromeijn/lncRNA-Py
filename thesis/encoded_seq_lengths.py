'''A comparison of different encoding methods and the achieved compression.'''

import numpy as np
import matplotlib.pyplot as plt
from lncrnapy.data import Data
from lncrnapy.features.tokenizers import BPELength
from lncrnapy.features import Length

seq_dir = '/data/s2592800/data/sequences'
features_dir = '/data/s2592800/data/features'

# Encoding ---------------------------------------------------------------------
dataset = Data([f'{seq_dir}/pretrain_human_pcrna.fasta',
                f'{seq_dir}/pretrain_human_ncrna.fasta'])

dataset.calculate_feature(Length())
dataset.df['length/3'] = dataset.df['length'] / 3
dataset.df['length/4'] = dataset.df['length'] / 4
dataset.df['length/6'] = dataset.df['length'] / 6
dataset.df['length/9'] = dataset.df['length'] / 9
dataset.df['length/10'] = dataset.df['length'] / 10
dataset.df['length/13'] = dataset.df['length'] / 13
for vocab_size in [256, 512, 768, 1024, 2048, 4096, 8192]:
    print(vocab_size)
    dataset.calculate_feature(BPELength(f'{features_dir}/{vocab_size}.bpe', 
                                        vocab_size))

dataset.to_hdf('f{features_dir}/lengths.h5')

# Visualization ----------------------------------------------------------------
dataset = Data(hdf_filepath=f'{features_dir}/lengths.h5')

to_plot = {
    'NUC (4)': 'length',
    '3-mer (256)': 'length/3', 
    '6-mer (4096)': 'length/6', 
    '9-mer (4^9)': 'length/9',
    'BPE (256)': 'BPE length (vs=256)', 
    'BPE (1024)': 'BPE length (vs=1024)',
    'BPE (4096)': 'BPE length (vs=4096)',
    'CSE k=10': 'length/10',
}

fig, ax = plt.subplots(nrows=len(to_plot), ncols=2, figsize=(9,4))
for j, label in enumerate(['pcRNA', 'ncRNA']):
    for i, length in enumerate(to_plot):
        if length.startswith('NUC'):
            color = '#1F77B4'
        elif length[0].isnumeric(): 
            color = '#FF7F0E'
        elif length.startswith('BPE'):
            color = '#2CA02C'
        elif length.startswith('CSE'):
            color = '#D62728'
        else:
            raise RuntimeError()
        feature = to_plot[length]
        data = dataset.df[dataset.df['label'] == label][feature]
        left_text = round((data <= 768).sum() / len(data)*100)
        right_text = round((data > 768).sum() / len(data)*100)
        plot1 = data.plot.density(ind=np.arange(0,2112), ax=ax[i,j], alpha=0)
        ax[i,j].axvline(x=768, c='black', linestyle='--', linewidth=1)

        # grabbing x and y data from the kde plot
        x = plot1.get_children()[0]._x
        y = plot1.get_children()[0]._y

        # filling the space beneath the distribution
        ax[i,j].fill_between(x,y, alpha=0.5, color=color)

        # ax[i,j].axvline(x=1024)
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
        ax[i,j].set_xlim(0, 2112)
        ax[i,j].set_ylim(0)
        ax[i,j].text(0.5*768, 0.5, f'{left_text}%', va='center', ha='center',
                     transform=ax[i,j].get_xaxis_transform())
        ax[i,j].text(768+0.5*(2048-768), 0.5, f'{right_text}%', va='center', 
                     transform=ax[i,j].get_xaxis_transform(), ha='center')
        ylabel = length if j == 0 else None 
        ax[i,j].set_ylabel(ylabel, rotation=0, size='large', ha='right', 
                           va='center')

ax[0,0].set_title('pcRNA')
ax[0,1].set_title('ncRNA')
ax[len(to_plot)-1,0].set_xlabel('encoded sequence length')
ax[len(to_plot)-1,1].set_xlabel('encoded sequence length')
ax[len(to_plot)-1,0].set_xticks(np.arange(0,2049,256))
ax[len(to_plot)-1,1].set_xticks(np.arange(0,2049,256))

plt.tight_layout()
plt.subplots_adjust(left=0.14, hspace=0)
plt.savefig('encoded_seq_lengths.pdf')