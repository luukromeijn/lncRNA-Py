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


results_dir = 'results/report'

metric_layout = [['F1 (macro)', 'Precision (pcRNA)', 'Precision (ncRNA)'],
                 ['Accuracy', 'Recall (pcRNA)', 'Recall (ncRNA)']]

methods = [
    'lncRNA-BERT (3-mer)', 
    'lncRNA-BERT (CSE k=9)', 
    'lncRNA-LR',
    'lncRNA-RF',
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


results = results.replace({'PredLnc-GFStack': 'PredLnc',
                           'lncFinder': 'LncFinder',
                           'GENCODE-RefSeq':'GENCODE/RefSeq'})

print(results.groupby('Dataset')[['F1 (macro)']].mean())

# --- TABLE ---
t_metrics = ['F1 (macro)', 'Precision (pcRNA)', 'Recall (pcRNA)', 
             'Precision (ncRNA)', 'Recall (ncRNA)', 'Accuracy']
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
all_data.to_csv('thesis_table.csv', float_format="%.3f")

fig, ax = plt.subplots(nrows=len(metric_layout)*len(datasets), ncols=len(metric_layout[0]), figsize=(12,12))
for b, dataset in enumerate(results['Dataset'].unique()):

    width = 1/(len(methods)+1)
    center = ((len(methods)-1) * width)/2
    c = len(metric_layout)*b
    for i, metric_row in enumerate(metric_layout):
        for j, metric in enumerate(metric_row):
            ax[c+i,j].grid(axis='y')
            ax[c+i,j].set_axisbelow(True)
            for a, method in enumerate(results['Method'].unique()):
                ax[c+i,j].bar(
                    np.arange(width*a, 1 + (width*a)), metric, width=0.8*width, 
                    data=results[(results['Method'] == method) & 
                                 (results['Dataset'] == dataset)], label=method
                )
            ax[c+i,j].set_xticks([])
            ax[c+i,j].set_ylabel(metric)
            ax[c+i,j].set_ylim(lims[dataset])
    ax[c,1].set_title(dataset, fontweight='bold')

handles, labels = ax[2,0].get_legend_handles_labels()
fig.legend(handles, labels, ncols=round(len(methods)/2), loc='upper center')
fig.tight_layout(rect=(0,0,1,0.94))
plt.savefig(f'comparison.pdf')

exit()

# # Some general stuff

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
    
# # Encoding methods comparison ------------------------------------------------
# # Scores
# methods = ['NUC',   
#            'k-mer (k=3)', 'k-mer (k=6)', 'k-mer (k=9)',
#            'BPE (vs=256)', 'BPE (vs=1024)', 'BPE (vs=4096)',
#            'CSE (k=3)', 'CSE (k=4)', 'CSE (k=6)', 'CSE (k=7)',
#            'CSE (k=9)', 'CSE (k=10)', 'CSE (k=13)']

# colors = {'NUC': '#1F77B4', 'k-mer': '#FF7F0E', 'BPE': '#2CA02C', 
#           'CSE': '#D62728'}

# metric = 'F1 (macro)'

# fig, ax = plt.subplots(1,2, figsize=(12,3.25))
# for j, prb_or_ft in enumerate(['probed', 'fine-tuned']):
#     sw = 'CLSv2_EM_PRB' if prb_or_ft == 'probed' else 'CLSv2_EM_FT'
#     em_df = results_df[results_df['exp_name'].str.startswith(sw)]
#     em_df['encoding method'] = results_df['exp_name'].apply(get_em_name)
#     print(em_df[['encoding method', f'{metric}|valid']])
#     for i, method in enumerate(methods):
#         ax[j].bar([i], em_df[em_df['encoding method'] == method][f'{metric}|valid'],
#                 color=colors[method.split(' (')[0]])
    
#     ax[j].set_xticks(np.arange(len(methods)), methods, rotation=90)
#     ax[j].set_ylim(0.75, 1.0)
#     ax[j].grid(axis='y')
#     ax[j].set_axisbelow(True)
#     ax[j].set_ylabel(metric)
#     ax[j].set_title(prb_or_ft)

# fig.tight_layout()
# plt.savefig('encoding_methods_scores.pdf')
# plt.show()

# # Spaces
# em_df = results_df[results_df['exp_name'].str.startswith('CLSv2_EM_PRB')]
# em_df['encoding method'] = results_df['exp_name'].apply(get_em_name)
# fig = plt.figure(figsize=(12,10))
# mapping = {'NUC': '', 'k-mer': 'kmer_k', 'BPE': 'bpe_vs', 'CSE': 'conv_sm'}
# for i, method in enumerate(methods):
#     a = 2 if method.startswith('CSE') else 1
#     ax = plt.subplot(4,4,i+a)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.text(0.97, 0.97,
#         f'F1: {round(em_df[em_df["encoding method"]==method]["F1 (macro)|valid"].item(),2)}', 
#         va='top', ha='right', transform=ax.transAxes,
#     )
#     ax.set_title(method)
#     fn = f"{mapping[method.split(' ')[0]]}{method.split('=')[-1].strip(')')}.h5"
#     data = Data(hdf_filepath=f'results/report/spaces/EM_{fn.lower()}').df
#     for label in ['pcRNA', 'ncRNA']:
#         ax.scatter('L0', 'L1', data=data[data['label']==label], s=3, alpha=0.25,
#                    rasterized=True)
# lg_ax = plt.subplot(4,4,16)
# lg_ax.scatter(-1, -1, label='pcRNA', s=3)
# lg_ax.scatter(-1, -1, label='ncRNA', s=3)
# lg_ax.set_ylim(0,0.1)
# lg_ax.set_xlim(0,0.1)
# lg_ax.axis('off')
# lg_ax.legend(loc='lower right', markerscale=10/3)
# plt.tight_layout()
# plt.savefig('encoding_methods_latent.pdf', dpi=300)

# # Frameshift sensitivity
# fig = plt.figure(figsize=(12,7))
# gs = gridspec.GridSpec(3, 4, height_ratios=[4, 4, 0.2])
# leg_ax = fig.add_subplot(gs[2,:])
# axs = [fig.add_subplot(gs[i,j]) for i in range(2) for j in range(4)]

# methods = {
#     'Nucleotide Transformer': 'spaces/nt-v2.h5',
#     'NUC': 'frameshifts/FS_nuc.h5',
#     'BPE (vs=256)': 'frameshifts/FS_bpe_vs256.h5',
#     'k-mer (k=3)': 'frameshifts/FS_kmer_k3.h5',
#     # 'k-mer (k=6)': 'frameshifts/FS_kmer_k6.h5',
#     'CSE (k=3)': 'frameshifts/FS_conv_sm3.h5',
#     'CSE (k=4)': 'frameshifts/FS_conv_sm4.h5',
#     'CSE (k=9)': 'frameshifts/FS_conv_sm9.h5',
#     'CSE (k=10)': 'frameshifts/FS_conv_sm10.h5',
# }

# colors = {i:plt.rcParams['axes.prop_cycle'].by_key()['color'][i] 
#           for i in range(10)}

# for a, method in enumerate(methods):
#     data = Data(hdf_filepath=f'results/report/{methods[method]}').df
#     if method == 'Nucleotide Transformer':
#         data['label'] = data['rf_label']
#     axs[a].scatter('L0', 'L1', data=data[data['label']==-1], s=1, c='#BFBFBF',
#                    rasterized=True)
#     markers = {0:'X', 1:'^', 2:'o'}
#     for i in range(10):
#         for j in range(3):
#             axs[a].scatter('L0', 'L1', c=colors[i], marker=markers[j],
#                            data=data[(data['label']==i) & (data['rf']==j)], 
#                            s=100, rasterized=True)
#     axs[a].set_xticks([])
#     axs[a].set_yticks([])
#     axs[a].set_title(method)

# leg_ax.scatter(-1,-1,c='black', marker='X', s=100, label='1/4/7')
# leg_ax.scatter(-1,-1,c='black', marker='^', s=100, label='2/5/8')
# leg_ax.scatter(-1,-1,c='black', marker='o', s=100, label='3/6/9')
# leg_ax.set_xlim(0,1)
# leg_ax.set_ylim(0,1)
# leg_ax.axis('off')
# leg_ax.legend(loc='center', ncols=3, title='# nucleotides removed')
# plt.tight_layout()
# plt.savefig('frameshift_sensitivity.pdf')

# # MLM accuracy
# plt.figure(figsize=(6,3))
# a, b = 0.35, 0.55
# range = np.arange(a,b,(b-a)/100)
# for k in [6,7]:
#     data = Data(hdf_filepath=
#                 f'{results_dir}/mlm_accuracies/MLM_valid_accuracy_CSE_sm{k}.h5')
#     print(data.df.groupby(by='label')['Accuracy (MLM)'].mean())
#     for label in ['pcRNA', 'ncRNA']:
#         kde = gaussian_kde(data.df[data.df['label']==label]['Accuracy (MLM)'])
#         kde = kde.evaluate(range)
#         plt.plot(range, kde, label=f'CSE (k={k}), {label}')
# plt.xlabel('MLM accuracy')
# plt.ylabel('density')
# plt.legend(loc='center left', bbox_to_anchor=(1.05,0.5))
# plt.tight_layout()
# plt.savefig('cse_mlm_accuracy.pdf')

# Motif visualization ----------------------------------------------------------
# kernels = np.load('results/report/CSE_sm9_kernels.npy')

# fig = plt.figure(figsize=(12,5))
# gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1], height_ratios=[4, 4, 0.2])
# leg_ax = fig.add_subplot(gs[2,:])
# axs = [fig.add_subplot(gs[i,j]) for i in range(2) for j in range(3)]

# labels = 'ACGT'
# for j in range(6):
#     for i in range(4):
#         # Calculate bottom coordinates (based on cumulative sum)
#         pos_case = np.where( # Sum all smaller values (to use as bottombase)
#             [(kernels[j][i] > kernels[j]) & (kernels[j] > 0)], 
#             kernels[j], 0
#         ).squeeze().sum(axis=0)
#         neg_case = np.where( # Sum all larger values (when negative)
#             [(kernels[j][i] < kernels[j]) & (kernels[j] < 0)], 
#             kernels[j], 0
#         ).squeeze().sum(axis=0)
#         # Apply either the positive or negative case based on value
#         bottom = np.where(kernels[j][i] >= 0, pos_case, neg_case)
#         # And do the actual plotting
#         fig = axs[j].bar(np.arange(1, kernels[j].shape[-1]+1), 
#                          kernels[j][i], bottom=bottom, label=labels[i])

#     # Making the plot nicer
#     axs[j].set_xticks(np.arange(1, kernels[j].shape[-1]+1))
#     axs[j].axhline(c='black')
#     if j >= 3:
#         axs[j].set_xlabel('Position')
#     if j % 3 == 0:
#         axs[j].set_ylabel('Kernel weight')

# leg_ax.legend(*axs[-1].get_legend_handles_labels(), loc='center', ncols=4)
# leg_ax.axis('off')
# plt.tight_layout()
# plt.savefig('visualize_kernels.pdf')
# plt.show()

# RNACentral comparison --------------------------------------------------------
# # Fine-tuning performance
# metric = 'F1 (macro)'
# exp_names_df = pd.DataFrame({
#     'database': ['none', 'human', 'RNAcentral']*3,
#     'encoding method': ['K-mer (k=3)']*3 + ['CSE (k=6)']*3 + ['CSE (k=9)']*3,
#     'exp_name': [
#         'CLSv2_SCR_kmer_k3_dm768_N12_bs8_lr1e-05_wd0_cl768_d0',
#         'CLSv2_EM_FT_kmer_finetuned_k3_dm768_N12_dff3072_h12_bs8_lr1e-05_wd0_cl768_d0',
#         'CLSv2_RNAC_FT_kmer_finetuned_k3_dm768_N12_dff3072_h12_bs8_lr1e-05_wd0_cl768_d0',
#         'CLSv2_SCR_conv_nm768_sm6_dm768_N12_bs8_lr1e-05_wd0_cl768_d0',
#         'CLSv2_EM_FT_conv_finetuned_nm768_sm6_dm768_N12_dff3072_h12_bs8_lr1e-05_wd0_cl768_d0',
#         'CLSv2_RNAC_FT_conv_finetuned_nm768_sm6_dm768_N12_dff3072_h12_bs8_lr1e-05_wd0_cl768_d0',
#         'CLSv2_SCR_conv_nm768_sm9_dm768_N12_bs8_lr1e-05_wd0_cl768_d0',
#         'CLSv2_EM_FT_conv_finetuned_nm768_sm9_dm768_N12_dff3072_h12_bs8_lr1e-05_wd0_cl768_d0',
#         'CLSv2_RNAC_FT_conv_finetuned_nm768_sm9_dm768_N12_dff3072_h12_bs8_lr1e-05_wd0_cl768_d0',
#     ]
# })


# fig = plt.figure(figsize=(10,3.05))
# gs = gridspec.GridSpec(2, 3, width_ratios=[4, 4, 1], height_ratios=[1, 1])
# ax1 = fig.add_subplot(gs[:, 1])
# ax2 = fig.add_subplot(gs[:, 0])
# ax_top = fig.add_subplot(gs[0, 2])
# ax_bot = fig.add_subplot(gs[1, 2])

# rnac_df = pd.merge(exp_names_df, results_df, 'left', 'exp_name')

# for i, db in enumerate(['none', 'human', 'RNAcentral']):
#     bar = ax1.bar(np.arange(i/4, 3, step=1), 
#             rnac_df[rnac_df['database']==db][f'{metric}|valid'], 1/4, label=db)
# # ax[0].legend(title='Pre-training data', ncols=3, loc='upper center', 
# #           bbox_to_anchor=(0.5, 1.05))
# ax1.set_xticks(np.arange(1/4, 3, 1), ['K-mer (k=3)', 'CSE (k=6)', 'CSE (k=9)'])
# ax1.set_ylabel(metric)
# ax1.set_ylim(0.6, 1)
# ax1.grid(axis='y')
# ax1.set_axisbelow(True)
# # plt.tight_layout()

# # Learning curves
# metric = 'F1 (macro)'
# colors = {'none': '#1F77B4', 'human': '#FF7F0E', 'RNAcentral': '#2CA02C'}
# styles = {'K-mer (k=3)': '-', 'CSE (k=6)': '--', 'CSE (k=9)': ':'}

# for i, row in exp_names_df.iterrows():
#     if row['exp_name'] == 'None':
#         continue
#     history = pd.read_csv(
#         f'results/report/pretraining/{row["exp_name"]}/history.csv'
#     )
#     ax2.plot(np.arange(1,101), f'{metric}|valid', styles[row['encoding method']], 
#             data=history, color=colors[row['database']])
#     # ax.plot(np.arange(100), f'{metric}|train', styles[row['encoding method']], 
#     #         data=history, color=colors[row['database']], alpha=0.25)

# ylim = ax2.get_ylim()
# xlim = ax2.get_xlim()
# for em in styles:
#     ax2.plot(-1, -1, styles[em], label=em, c='grey')
# ax2.set_ylim(0.6,1)
# ax2.set_xlim(xlim)
# ax2.set_ylabel(metric)
# ax2.set_xlabel('epoch')

# # ax_top = plt.subplot(2,3,3)
# ax_top.legend(*[item[-3:] for item in ax1.get_legend_handles_labels()], title='Pre-training data', bbox_to_anchor=(-0.8, 0.5), loc='center left')
# ax_top.axis('off')
# # ax_bot = plt.subplot(2,3,6)
# ax_bot.legend(*[item[-3:] for item in ax2.get_legend_handles_labels()], title='Encoding method',  bbox_to_anchor=(-0.8, 0.5), loc='center left')
# ax_bot.axis('off')
# plt.tight_layout()
# plt.savefig('pretraining_data.pdf')
# plt.show()



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
ax_mlm_acc = fig.add_subplot(gs_right[1:3,3-3])
ax_mlm_lg1 = fig.add_subplot(gs_right[1,  4-3])
ax_mlm_lg2 = fig.add_subplot(gs_right[2,  4-3])

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

gs_left.tight_layout(fig, rect=[0,0,0.55,1], w_pad=0, h_pad=0.3)
gs_right.tight_layout(fig, rect=[0.54,0,1,1])
plt.savefig('pretraining_thesis.pdf')
plt.show()

# # Latent space inspection ------------------------------------------------------
# seq_dir = 'data/sequences'

# # Loading the data
# data = Data([f'{seq_dir}/valid_gencode_pcrna.fasta',
#              f'{seq_dir}/valid_gencode_ncrna.fasta'], 
#             'results/report/spaces/LONG_kmer_k3.h5')

# # Feature plots
# mlm_acc = Data(hdf_filepath='results/report/MLM_valid_accuracy_LONG_kmer_k3.h5')
# data.df = pd.merge(data.df, mlm_acc.df[['id', 'Accuracy (MLM)']], on='id')
# data.calculate_feature(Length())
# for feature in ['length', 'Accuracy (MLM)']:
#     data.plot_feature_scatter('L0', 'L1', feature, figsize=(5,3.8), 
#                               filepath=f'3mer_latent_{feature.lower()}.pdf')
#     plt.close()
    
# # Calculating 6-mer freqs
# def calculate_distances(data):
#     X = data.df[KmerFreqs(6).name].to_numpy()
#     sum_squared = np.sum(X**2, axis=1)
#     distances = np.sqrt(sum_squared[:, None] + sum_squared[None, :] 
#                         -2*np.dot(X, X.T))
#     return distances.mean()

# data.calculate_feature(KmerFreqs(6))
# print("Total average distance:")
# print(calculate_distances(data))

# # Defining clusters
# def on_scroll(event, data, clusters):
#     xlim = event.inaxes.get_xlim()
#     ylim = event.inaxes.get_ylim()
#     data = copy.deepcopy(data)
#     data.filter_outliers('L0', xlim)
#     data.filter_outliers('L1', ylim)
#     dist = calculate_distances(data)
#     genes = get_gencode_gene_names(data)
#     clusters.append([xlim[0], xlim[1], ylim[0], ylim[1], dist, genes])
#     plt.gca().add_patch(
#         Rectangle((xlim[0],ylim[0]), xlim[1]-xlim[0],  ylim[1]-ylim[0], 
#                   edgecolor='black', facecolor='none')
#         )
#     plt.draw()

# # clusters = []
# # data.plot_feature_scatter('L0', 'L1')
# # plt.connect('scroll_event', lambda x: on_scroll(x, data, clusters))
# # plt.show()

# # clusters = pd.DataFrame(clusters, 
# #                         columns=['x0', 'x1', 'y0', 'y1', 'dist', 'genes'])
# # clusters.to_csv('clusters.csv')

# # Adding gProfiler information
# gprof_dir = 'results/report/3mer_enrichment'
# gprof_cols = ['term_name', 'adjusted_p_value']
# mf_repl = {'term_name': 'MF: term', 'adjusted_p_value': 'MF: P'}
# bp_repl = {'term_name': 'BP: term', 'adjusted_p_value': 'BP: P'}
# cc_repl = {'term_name': 'CC: term', 'adjusted_p_value': 'CC: P'}

# clusters = pd.read_csv('clusters.csv')
# data.plot_feature_scatter('L0', 'L1', figsize=(6,6))
# rows = []
# for i, row in clusters.iterrows():
#     try:
#         gea_data = pd.read_csv(f'{gprof_dir}/gProfiler_{i}.csv')
#         gea_data = gea_data.groupby(by='source', as_index=False).first()
#         mf = (gea_data[gea_data['source']=='GO:MF'][gprof_cols]
#               .rename(columns=mf_repl).reset_index(drop=True))
#         bp = (gea_data[gea_data['source']=='GO:BP'][gprof_cols]
#               .rename(columns=bp_repl).reset_index(drop=True))
#         cc = (gea_data[gea_data['source']=='GO:CC'][gprof_cols]
#               .rename(columns=cc_repl).reset_index(drop=True))
#         row_data = pd.concat((mf, bp, cc), axis=1)
#         row_data['ID'] = [i+1]
#     except FileNotFoundError:
#         row_data = pd.DataFrame([[np.nan]*6 + [i+1]], columns=[
#             'MF: term', 'MF: P', 'BP: term', 'BP: P', 'CC: term', 'CC: P', 'ID'
#         ])
#     rows.append(row_data)
#     plt.gca().add_patch(Rectangle((row['x0'], row['y0']), 
#                                   row['x1']-row['x0'], row['y1']-row['y0'], 
#                                   edgecolor='black', facecolor='none'))
#     plt.text(row['x0']+1, row['y1']-5, str(i+1), fontweight='bold')
# plt.savefig('3mer_latent.pdf')

# gprof_data = pd.concat(rows, axis=0, ignore_index=True)
# clusters['Unnamed: 0'] = clusters['Unnamed: 0'] + 1
# all = pd.merge(clusters, gprof_data, how='left', left_on='Unnamed: 0', 
#                right_on='ID')[['ID', 'dist', 'MF: term', 'MF: P', 'BP: term', 
#                                'BP: P', 'CC: term', 'CC: P']]
# all.to_csv('clusters_enriched.csv', index=False, float_format="%.2e")

# # NLM comparison ---------------------------------------------------------------
# methods = {
#     'DNABERT-2': 'dnabert2',
#     'Nucleotide Transformer': 'nt-v2',
#     'GENA-LM': 'gena-lm',
#     'BiRNA-BERT': 'birnabert',
#     'RiNALMO': 'rinalmo',
#     'lncRNA-BERT': 'LONG_kmer_k3',
# }

# fig, ax = plt.subplots(3,3,figsize=(12,8), height_ratios=[3.75,3.75,0.5])
# for i, method in enumerate(methods):
#     a, b = int(i/3),i % 3
#     ax[a,b].set_xticks([])
#     ax[a,b].set_yticks([])
#     ax[a,b].set_title(method)
#     data = Data(hdf_filepath=f'results/report/spaces/{methods[method]}.h5').df
#     x, y = (('L0', 'L1') if method in ['lncRNA-BERT', 'Nucleotide Transformer']
#             else ('L1', 'L2'))
#     for label in ['pcRNA', 'ncRNA']:
#         ax[a,b].scatter(x, y, data=data[data['label']==label], s=3, alpha=0.25,
#                    rasterized=True)
# lg_ax = ax[2,1]
# lg_ax.scatter(-1, -1, label='pcRNA', s=3)
# lg_ax.scatter(-1, -1, label='ncRNA', s=3)
# lg_ax.set_ylim(0,0.1)
# lg_ax.set_xlim(0,0.1)
# lg_ax.legend(loc='upper center', markerscale=10/3, ncols=2)
# for i in range(3):
#     ax[2,i].axis('off')
# plt.tight_layout()
# plt.savefig('nlm_comparison_latent.pdf')