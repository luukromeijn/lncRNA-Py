from lncrnapy.data import Data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig, axs = plt.subplots(1,2,figsize=(4,2))

colors = {
    'pcRNA': '#1F77B4',
    'ncRNA': '#FF7F0E',
}

methods = ['k-mer (k=3)', 'CSE (k=9)']    
mapping = {'NUC': '', 'k-mer': 'kmer_k', 'BPE': 'bpe_vs', 'CSE': 'conv_sm'}

for i, method in enumerate(methods):
    fn = f"{mapping[method.split(' ')[0]]}{method.split('=')[-1].strip(')')}.h5"
    data = Data(hdf_filepath=f'results/thesis/spaces/EM_{fn.lower()}').df
    for label in ['pcRNA', 'ncRNA']:
        axs[i].scatter('L0', 'L1', data=data[data['label']==label], s=1, 
                       alpha=0.25, rasterized=True, color=colors[label], label=label)
    axs[i].set_xticks([])
    axs[i].set_yticks([])

plt.tight_layout()
plt.savefig('ga_latent.png')
plt.close()

fig, ax = plt.subplots(1,1,figsize=(4,1))
for label in ['pcRNA', 'ncRNA']:
    ax.scatter(-1,-1, label=label, color=colors[label])
ax.legend(ncols=2, loc='center')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.axis('off')
plt.tight_layout()
plt.savefig('ga_legend.png')
plt.close()

results_dir = 'results'

methods = [
    'lncRNA-BERT (3-mer)', 
    'lncRNA-BERT (CSE k=9)', 
    'CPAT', 
    'LncFinder',
    'PredLnc-GFStack',
    'LncADeep', 
    'RNAsamba',
    'mRNN',
]

datasets = [
    'GENCODE-RefSeq',
    'CPAT',
    'RNAChallenge'
]

lims = {
    'GENCODE/RefSeq': (0.7, 1.0),
    'CPAT': (0.9, 1.0),
    'RNAChallenge': (0.0, 0.5),
}

results = pd.concat(
    [pd.read_csv(f'{results_dir}/scores/{method.lower()}_{dataset.lower()}.csv') 
     for method in methods for dataset in datasets], ignore_index=True
)

results = results.replace({'lncRNA-BERT (3-mer)': 'Ours (3-mer)',
                           'lncRNA-BERT (CSE k=9)': 'Ours (CSE, k=9)',
                           'PredLnc-GFStack': 'PredLnc',
                           'lncFinder': 'LncFinder',
                           'GENCODE-RefSeq':'GENCODE/RefSeq'})

fig, ax = plt.subplots(1,3, figsize=(5,2))
for i, method in enumerate(results['Method'].unique()):
    for j, dataset in enumerate(results['Dataset'].unique()):
        ax[j].bar(i, results[(results['Method']==method) & 
                             (results['Dataset']==dataset)]['F1 (macro)'])
        ax[j].set_ylim(lims[dataset])
        ax[j].set_xticks([])
        ax[j].set_title(dataset, size=10)
ax[0].set_ylabel('F1')

plt.tight_layout()
plt.savefig('ga_comparison.png')
plt.close()

fig, ax = plt.subplots(1,1,figsize=(4,2))
for method in results['Method'].unique():
    ax.bar(-1, -1, label=method)
ax.legend(ncols=3, columnspacing=0.25, handletextpad=0.25, prop={'size':10}, 
          loc='center')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.axis('off')
plt.savefig('ga_comp_legend.png')
plt.show()