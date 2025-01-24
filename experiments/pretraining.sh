# Commands for comparing different pre-training settings.

# PRE-TRAINING -----------------------------------------------------------------
# --- Human data (same as encoding method comparison) ---
# 3-mer
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_human.fasta \
    sequences/valid_gencode.fasta \
    --exp_prefix MLM_EM \
    --encoding_method kmer \
    --k 3 \
    --data_dir data \
    --results_dir results

# CSE (kernel_size=9)
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_human.fasta \
    sequences/valid_gencode.fasta \
    --exp_prefix MLM_EM \
    --encoding_method cse \
    --kernel_size 9 \
    --data_dir data \
    --results_dir results

# --- RNAcentral data ---
# K-mer (k=3)
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_rnacentral.fasta \
    sequences/valid_gencode.fasta \
    --exp_prefix MLM_RNAC \
    --encoding_method 3mer \
    --k 3 \
    --data_dir data \
    --results_dir results

# CSE (kernel_size=9)
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_rnacentral.fasta \
    sequences/valid_gencode.fasta \
    --exp_prefix MLM_RNAC \
    --encoding_method cse \
    --kernel_size 9 \
    --data_dir data \
    --results_dir results

# FINE-TUNING ------------------------------------------------------------------
# --- Human data (same as encoding method comparison) ---
# 3-mer
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_FT \
    --encoding_method kmer \
    --k 3 \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_EM_kmer_k3_dm768_N12_bs8_ws32000_cl768_d0

# CSE (kernel_size=9)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_FT \
    --encoding_method cse \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_EM_cse_nm768_sm9_dm768_N12_bs8_ws32000_cl768_d0

# --- RNAcentral data ---
# K-mer (k=3)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_RNAC_FT \
    --encoding_method kmer \
    --k 3 \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_RNAC_3mer_k3_dm768_N12_bs8_ws32000_cl768_d0


# CSE (kernel_size=9)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_RNAC_FT \
    --encoding_method cse \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_RNAC_cse_nm768_sm9_dm768_N12_bs8_ws32000_cl768_d0


# --- From scratch (no pre-training) ---
# K-mer (k=3)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_SCR \
    --encoding_method kmer \
    --k 3 \
    --data_dir data \
    --results_dir results

# CSE (kernel_size=9)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_SCR \
    --encoding_method cse \
    --kernel_size 9 \
    --data_dir data \
    --results_dir results