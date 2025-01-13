import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lncrnapy.data import Data
from scipy.stats import gaussian_kde
from lncrnapy.data import get_gencode_gene_names
from lncrnapy.features import Length, KmerFreqs
from matplotlib import colormaps
from matplotlib.patches import Rectangle
import copy


# PERFORMANCE COMPARISON TABLE -------------------------------------------------

results_dir = 'results/report'

metric_layout = [['F1 (macro)', 'Precision (pcRNA)', 'Precision (ncRNA)'],
                 ['Accuracy', 'Recall (pcRNA)', 'Recall (ncRNA)']]

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
    'GENCODE/Refseq': (0.7, 1.0),
    'CPAT': (0.9, 1.0),
    'RNAChallenge': (0.0, 0.5),
}

results = pd.concat(
    [pd.read_csv(f'{results_dir}/scores/{method.lower()}_{dataset.lower()}.csv') 
     for method in methods for dataset in datasets], ignore_index=True
)


results = results.replace({'PredLnc-GFStack': 'PredLnc',
                           'lncFinder': 'LncFinder',
                           'GENCODE-RefSeq':'GENCODE/RefSeq'})

print(results.groupby('Dataset')[['F1 (macro)']].mean())

t_metrics = ['F1 (macro)', 'Precision (pcRNA)', 'Recall (pcRNA)', 
             'Precision (ncRNA)', 'Recall (ncRNA)']
table = results[['Method', 'Dataset'] + t_metrics]
all_data = []
for method in table['Method'].unique():
    subtable = table[table['Method']==method]
    method_data = []
    for dataset in table['Dataset'].unique():
        method_data.append(
            subtable[subtable['Dataset']==dataset][t_metrics].values
        )
    method_data = pd.DataFrame(np.hstack(method_data).T, columns=[method], 
                        index=pd.MultiIndex.from_product((datasets, t_metrics)))
    all_data.append(method_data)
all_data = pd.concat(all_data, axis=1)
all_data.to_csv('table.csv', float_format="%.3f")

# SOME GENERAL DATA PREP -------------------------------------------------------

results_df = pd.read_csv('results/report/validation_results/cls_results.csv')


def get_em_name(exp_name):
    '''Extracts encoding method name from experiment name'''
    vars = exp_name.split('_')
    if 'nuc' in vars:
        return 'NUC'
    if 'kmer' in vars:
        k = None
        for var in vars:
            if var.startswith('k') and var != 'kmer':
                k = int(var[1:])
        return f'k-mer (k={k})'
    if 'bpe' in vars:
        vs = None
        for var in vars:
            if var.startswith('vs'):
                vs = int(var[2:])
        return f'BPE (vs={vs})'
    if 'conv' in vars:
        sm = None
        for var in vars:
            if var.startswith('sm'):
                sm = int(var[2:]) 
        return f'CSE (k={sm})'

# PRE-TRAINING DATA ------------------------------------------------------------
fig = plt.figure(figsize=(12,5))
gs_left = fig.add_gridspec(4, 3, width_ratios=[0.15,2,2])
gs_left.update(right=0.5)
gs_right = fig.add_gridspec(4, 2, width_ratios=[2,1.1])
gs_right.update(left=0.5)
ax_leg_hum = fig.add_subplot(gs_left[0:2,0])
ax_hum_hum = fig.add_subplot(gs_left[0:2,1])
ax_arc_hum = fig.add_subplot(gs_left[0:2,2])
ax_leg_arc = fig.add_subplot(gs_left[2:4,0])
ax_hum_arc = fig.add_subplot(gs_left[2:4,1])
ax_arc_arc = fig.add_subplot(gs_left[2:4,2])
ax_lrn_crv = fig.add_subplot(gs_right[0:2,3-3])
ax_lrn_lg1 = fig.add_subplot(gs_right[0,  4-3])
ax_lrn_lg2 = fig.add_subplot(gs_right[1,  4-3])
ax_mlm_acc = fig.add_subplot(gs_right[2:4,3-3])
ax_mlm_lg1 = fig.add_subplot(gs_right[2,  4-3])
ax_mlm_lg2 = fig.add_subplot(gs_right[3,  4-3])

# Latent spaces
space_dir = 'results/report/spaces'
huma_huma = Data(hdf_filepath=f'{space_dir}/EM_kmer_k3.h5')
huma_arch = Data(hdf_filepath=f'{space_dir}/HUM_ARCH_kmer_k3.h5')
rnac_huma = Data(hdf_filepath=f'{space_dir}/RNAC_HUM_kmer_k3.h5')
rnac_arch = Data(hdf_filepath=f'{space_dir}/RNAC_ARCH_kmer_k3.h5')

marker_size = 1

data = [[huma_huma.df, huma_arch.df],
        [rnac_huma.df, rnac_arch.df]]
axs = [[ax_hum_hum, ax_hum_arc],[ax_arc_hum, ax_arc_arc]]
titles = ['Human model', 'RNAcentral model']
y_labels = ['Human data', 'ArchiveII data']
for i, data_row in enumerate(data):
    for j, dataset in enumerate(data_row):
        for label in dataset['label'].unique():
            axs[i][j].scatter('L0', 'L1', data=dataset[dataset['label']==label], 
                             s=marker_size, label=label, rasterized=True)
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            axs[i][0].set_title(titles[i], size=10)
            axs[0][j].set_ylabel(y_labels[j], size=10)

leg_axs = [ax_leg_hum, ax_leg_arc]
for i in range(2):
    leg_axs[i].legend(
        *axs[1][i].get_legend_handles_labels(), loc='center right', 
        markerscale=5, borderaxespad=1
    )
    leg_axs[i].axis('off')

# Learning curves
exp_names_df = pd.DataFrame({
    'database': ['none', 'human', 'RNAcentral']*2,
    'encoding method': ['K-mer (k=3)']*3 + ['CSE (k=9)']*3,
    'exp_name': [
        'CLSv2_SCR_kmer_k3_dm768_N12_bs8_lr1e-05_wd0_cl768_d0',
        'CLSv2_EM_FT_kmer_finetuned_k3_dm768_N12_dff3072_h12_bs8_lr1e-05_wd0_cl768_d0',
        'CLSv2_RNAC_FT_kmer_finetuned_k3_dm768_N12_dff3072_h12_bs8_lr1e-05_wd0_cl768_d0',
        'CLSv2_SCR_conv_nm768_sm9_dm768_N12_bs8_lr1e-05_wd0_cl768_d0',
        'CLSv2_EM_FT_conv_finetuned_nm768_sm9_dm768_N12_dff3072_h12_bs8_lr1e-05_wd0_cl768_d0',
        'CLSv2_RNAC_FT_conv_finetuned_nm768_sm9_dm768_N12_dff3072_h12_bs8_lr1e-05_wd0_cl768_d0',
    ]
})

metric = 'F1 (macro)'
colors = {'none': '#1F77B4', 'human': '#FF7F0E', 'RNAcentral': '#2CA02C'}
styles = {'K-mer (k=3)': '-', 'CSE (k=9)': ':'} # , 'BPE (vs=256)': ':'}

for i, row in exp_names_df.iterrows():
    if row['exp_name'] == 'None':
        continue
    history = pd.read_csv(
        f'results/report/pretraining/{row["exp_name"]}/history.csv'
    )
    ax_lrn_crv.plot(np.arange(1,101), f'{metric}|valid', styles[row['encoding method']], 
            data=history, color=colors[row['database']])


for db in colors:
    ax_lrn_lg1.bar(-1, -1, color=colors[db], label=db)
    ax_lrn_lg1.set_xlim(0,1)
for em in styles:
    ax_lrn_lg2.plot(-1, -1, styles[em], label=em, c='grey')
ax_lrn_crv.set_ylabel(metric)
ax_lrn_crv.set_xlabel('epoch')
# ax_lrn_crv.set_title(' ', size=10)

ax_lrn_lg1.legend(*[item[-3:] for item in ax_lrn_lg1.get_legend_handles_labels()], title='Pre-training data', loc='upper left')
ax_lrn_lg1.axis('off')
ax_lrn_lg2.legend(*[item[-2:] for item in ax_lrn_lg2.get_legend_handles_labels()], title='Encoding method',  loc='lower left')
ax_lrn_lg2.axis('off')

# MLM accuracy density plot
dir = 'results/report/mlm_accuracies/MLM_acc_'
rnac_rnac = Data(hdf_filepath=f'{dir}RNAC_RNAC_kmer_k3.h5')
rnac_pcrna = Data(hdf_filepath=f'{dir}RNAC_pcRNA_kmer_k3.h5')
rnac_ncrna = Data(hdf_filepath=f'{dir}RNAC_ncRNA_kmer_k3.h5')
huma_rnac = Data(hdf_filepath=f'{dir}HUM_RNAC_kmer_k3.h5')
huma_pcrna = Data(hdf_filepath=f'{dir}HUM_pcRNA_kmer_k3.h5')
huma_ncrna = Data(hdf_filepath=f'{dir}HUM_ncRNA_kmer_k3.h5')

colors = {'RNAcentral': '#1F77B4', 'Human pcRNA': '#FF7F0E', 'Human ncRNA': '#2CA02C'}
styles = {'Human': '-', 'RNAcentral': ':'}

for (train, test), data in zip(
    [("RNAcentral", 'RNAcentral'), ('RNAcentral', 'Human pcRNA'), 
     ('RNAcentral', 'Human ncRNA'), ('Human', 'RNAcentral'), 
     ('Human', 'Human pcRNA'), ('Human', 'Human ncRNA')], 
    [rnac_rnac, rnac_pcrna, rnac_ncrna, 
     huma_rnac, huma_pcrna, huma_ncrna]
):
    data.df = data.df[~data.df['Accuracy (MLM)'].isna()]
    print(train, test, data.df['Accuracy (MLM)'].mean())
    range = np.arange(0,1,0.01)
    kde = gaussian_kde(data.df['Accuracy (MLM)']).evaluate(range)
    ax_mlm_acc.plot(range, kde, styles[train], c=colors[test])
ax_mlm_acc.set_ylabel('density')
ax_mlm_acc.set_xlabel('MLM accuracy')

for train in styles:
    ax_mlm_lg1.plot(-1,-1,styles[train],c='grey',label=train)
ax_mlm_lg1.axis('off')
ax_mlm_lg1.legend(title="Pre-training data", loc='upper left')
for test in colors:
    ax_mlm_lg2.plot(-1,-1,c=colors[test], label=test)
ax_mlm_lg2.axis('off')
ax_mlm_lg2.legend(title="Validation data", loc='lower left')

fig.text(0.015, 0.935, 'A', fontsize=18)
# fig.text(0.360, 0.92, 'B', fontsize=18, fontweight='bold')
# fig.text(0.152, 0.43, 'C', fontsize=18, fontweight='bold')
# fig.text(0.360, 0.43, 'D', fontsize=18, fontweight='bold')
fig.text(0.550, 0.935, 'B', fontsize=18)
fig.text(0.550, 0.45, 'C', fontsize=18)

gs_left.tight_layout(fig, rect=[0,0,0.55,1], w_pad=0, h_pad=0.3)
gs_right.tight_layout(fig, rect=[0.54,0,1,1])
plt.savefig('pretraining.pdf')

# ENCODING METHODS -------------------------------------------------------------
methods = ['NUC', 'BPE (vs=256)',
           'k-mer (k=3)', 'k-mer (k=9)',
           'CSE (k=9)', 'CSE (k=10)']    

colors = {
    'pcRNA': '#1F77B4',
    'ncRNA': '#FF7F0E',
}

fig = plt.figure(figsize=(12,7.2))
gs_left = fig.add_gridspec(2,3)
gs_left.update(right=0.5, bottom=0.31, top=1)
gs_right = fig.add_gridspec(2, 1)
gs_right.update(left=0.5, bottom=0.31, top=1)
gs_bottom = fig.add_gridspec(1,6, width_ratios=[1,1,1,1,1,0.5])
gs_bottom.update(top=0.31)

# -- Latent spaces
tsne_axs = [fig.add_subplot(gs_left[int(i/3),i%3]) for i in range(len(methods))]
em_df = results_df[results_df['exp_name'].str.startswith('CLSv2_EM_PRB')]
em_df['encoding method'] = results_df['exp_name'].apply(get_em_name)
mapping = {'NUC': '', 'k-mer': 'kmer_k', 'BPE': 'bpe_vs', 'CSE': 'conv_sm'}

lg_ax = tsne_axs[-2]
lg_ax.scatter(0,0,label='pcRNA',s=1)
lg_ax.scatter(0,0,label='ncRNA',s=1)
lg_ax.legend(loc='lower center', markerscale=5, ncols=2, 
             bbox_to_anchor=(0.5,-0.3), columnspacing=0.1, handletextpad=0.1)

for i, method in enumerate(methods):
    ax = tsne_axs[i]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.97, 0.97,
        f'F1: {round(em_df[em_df["encoding method"]==method]["F1 (macro)|valid"].item(),2)}', 
        va='top', ha='right', transform=ax.transAxes,
    )
    ax.set_title(method)
    fn = f"{mapping[method.split(' ')[0]]}{method.split('=')[-1].strip(')')}.h5"
    data = Data(hdf_filepath=f'results/report/spaces/EM_{fn.lower()}').df
    for label in ['pcRNA', 'ncRNA']:
        ax.scatter('L0', 'L1', data=data[data['label']==label], s=1, alpha=0.25,
                   rasterized=True, color=colors[label])

# Scores
methods = ['NUC',   
           'k-mer (k=3)', 'k-mer (k=6)', 'k-mer (k=9)',
           'BPE (vs=256)', 'BPE (vs=1024)', 'BPE (vs=4096)',
           'CSE (k=3)', 'CSE (k=4)', 'CSE (k=6)', 'CSE (k=7)',
           'CSE (k=9)', 'CSE (k=10)'] # , 'CSE (k=13)']

colors = {'NUC': '#1F77B4', 'k-mer': '#FF7F0E', 'BPE': '#2CA02C', 
          'CSE': '#D62728'}

metric = 'F1 (macro)'

prb_ax = fig.add_subplot(gs_right[0])
ftn_ax = fig.add_subplot(gs_right[1])

ax = [prb_ax, ftn_ax]
for j, prb_or_ft in enumerate(['probed', 'fine-tuned']):
    sw = 'CLSv2_EM_PRB' if prb_or_ft == 'probed' else 'CLSv2_EM_FT'
    em_df = results_df[results_df['exp_name'].str.startswith(sw)]
    em_df['encoding method'] = results_df['exp_name'].apply(get_em_name)
    print(em_df[['encoding method', f'{metric}|valid']])
    for i, method in enumerate(methods):
        ax[j].bar([i], em_df[em_df['encoding method'] == method][f'{metric}|valid'],
                color=colors[method.split(' (')[0]])
    if j == 1:
        ax[j].set_xticks(np.arange(len(methods)), methods, rotation=90)
    else:
        ax[j].set_xticks([])
    ax[j].set_ylim(0.75, 1.0)
    ax[j].grid(axis='y')
    ax[j].set_axisbelow(True)
    ax[j].set_ylabel(metric)
    ax[j].set_title(prb_or_ft)

# Frameshift sensitivity
leg_ax = fig.add_subplot(gs_bottom[:,5])
axs = [fig.add_subplot(gs_bottom[0,i]) for i in range(5)]

methods = {
    # 'lncRNA-BERT (NUC)': 'frameshifts/FS_nuc.h5',
    'BPE (vs=256)': 'frameshifts/FS_bpe_vs256.h5',
    'K-mer (k=3)': 'frameshifts/FS_kmer_k3.h5',
    'CSE (k=9)': 'frameshifts/FS_conv_sm9.h5',
    'CSE (k=10)': 'frameshifts/FS_conv_sm10.h5',
    'Nucleotide Transformer': 'spaces/nt-v2.h5',
}

colors = {i:plt.rcParams['axes.prop_cycle'].by_key()['color'][i] 
          for i in range(10)}

for a, method in enumerate(methods):
    data = Data(hdf_filepath=f'results/report/{methods[method]}').df
    if method == 'Nucleotide Transformer':
        data['label'] = data['rf_label']
    axs[a].scatter('L0', 'L1', data=data[data['label']==-1], s=1, c='#BFBFBF',
                   rasterized=True)
    markers = {0:'X', 1:'^', 2:'o'}
    for i in range(10):
        for j in range(3):
            axs[a].scatter('L0', 'L1', c=colors[i], marker=markers[j],
                           data=data[(data['label']==i) & (data['rf']==j)], 
                           s=50, rasterized=True)
    axs[a].set_xticks([])
    axs[a].set_yticks([])
    axs[a].set_title(method)

leg_ax.scatter(-1,-1,c='black', marker='X', s=50, label='{1,4,7}')
leg_ax.scatter(-1,-1,c='black', marker='^', s=50, label='{2,5,8}')
leg_ax.scatter(-1,-1,c='black', marker='o', s=50, label='{3,6,9}')
leg_ax.set_xlim(0,1)
leg_ax.set_ylim(0,1)
leg_ax.axis('off')
leg_ax.legend(loc='center left', title='Removed (#nt)')

fig.text(0.015, (0.94*(1-0.31))+0.31,'A', fontsize=18)
fig.text(0.5,   (0.94*(1-0.31))+0.31,'B', fontsize=18)
fig.text(0.5,   (0.56*(1-0.31))+0.31,'C', fontsize=18)
fig.text(0.015, 0.31,                'D', fontsize=18)

gs_left.tight_layout(fig, rect=[0,0.31,0.5,0.98], w_pad=0.5)
gs_right.tight_layout(fig, rect=[0.5,0.31,1,1])
gs_bottom.tight_layout(fig, rect=[0,0,1,0.33])
plt.savefig('encoding_methods.pdf')
    
# NLM COMPARISON ---------------------------------------------------------------
methods = {
    'DNABERT-2': 'dnabert2',
    'BiRNA-BERT': 'birnabert',
    'Nucleotide Transformer': 'nt-v2',
    'RiNALMO': 'rinalmo',
    'GENA-LM': 'gena-lm',
    'lncRNA-BERT': 'EM_kmer_k3',
}

colors = {
    'pcRNA': '#1F77B4',
    'ncRNA': '#FF7F0E',
}

fig = plt.figure(figsize=(6,9.1))
gs = gridspec.GridSpec(4, 2, height_ratios=[4, 4, 4, 0.4])
leg_ax = fig.add_subplot(gs[3,:])
ax = [fig.add_subplot(gs[i,j]) for i in range(3) for j in range(2)]
leg_ax.scatter(0,0,label='pcRNA',s=1)
leg_ax.scatter(0,0,label='ncRNA',s=1)
leg_ax.legend(loc='center', markerscale=5, ncols=2, title='RNA type')
leg_ax.axis('off')
for i, method in enumerate(methods):
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_title(method)
    data = Data(hdf_filepath=f'results/report/spaces/{methods[method]}.h5').df
    x, y = (('L0', 'L1') if method in ['lncRNA-BERT', 'Nucleotide Transformer']
            else ('L1', 'L2'))
    for label in ['pcRNA', 'ncRNA']:
        ax[i].scatter(x, y, data=data[data['label']==label], s=1, alpha=0.25,
                   rasterized=True, label=label, c=colors[label])

plt.tight_layout()
plt.savefig('nlm_comparison_latent.pdf')