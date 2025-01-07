from Bio import SeqIO

data_src_dir = "C:/Users/luukr/Downloads/rna"
data_tgt_dir = 'data/sequences'
filenames = [
    f'human.{i}.rna.fna' for i in range(1,14)
]
pcrna = []
ncrna = []
labels = set()

for filename in filenames:
    seqs = SeqIO.parse(f'{data_src_dir}/{filename}', 'fasta')
    for seq in seqs:
        label = seq[:-1].description.split(', ')[-1]
        if label == 'mRNA':
            pcrna.append(seq)
        elif label == 'long non-coding RNA':
            ncrna.append(seq)

print(len(pcrna), len(ncrna))
SeqIO.write(pcrna, f'{data_tgt_dir}/refseq225_human_pcrna.fasta', 'fasta')
SeqIO.write(ncrna, f'{data_tgt_dir}/refseq225_human_ncrna.fasta', 'fasta')