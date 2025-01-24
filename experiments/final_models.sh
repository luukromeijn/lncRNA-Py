# Script for (pre-)training the final models. Assuming the pre-training is 
# automatically interrupted after 7 days.

# lncRNA-BERT (3-mer tokenization) ---------------------------------------------
# Pre-training
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_human.fasta \
    sequences/valid_gencode.fasta \
    --epochs 10000 \
    --exp_prefix MLM_LONG \
    --encoding_method kmer \
    --k 3 \
    --data_dir data \
    --results_dir results

# Fine-tuning
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_LONG_FT \
    --encoding_method kmer \
    --k 3 \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_LONG_kmer_k3_dm768_N12_bs8_ws32000_cl768_d0

# lncRNA-BERT (CSE, k=9) -------------------------------------------------------
# Pre-training
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_human.fasta \
    sequences/valid_gencode.fasta \
    --epochs 10000 \
    --exp_prefix MLM_LONG \
    --encoding_method cse \
    --kernel_size 9 \
    --data_dir data \
    --results_dir results

# Fine-tuning
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_LONG_FT \
    --encoding_method cse \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_LONG_cse_nm768_sm9_dm768_N12_bs8_ws32000_cl768_d0