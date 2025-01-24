# Commands for carrying out encoding method comparison.

# PRE-TRAINING -----------------------------------------------------------------
# NUC
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_human.fasta \
    sequences/valid_gencode.fasta \
    --exp_prefix MLM_EM \
    --encoding_method nuc \
    --data_dir data \
    --results_dir results

# 3-mer
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_human.fasta \
    sequences/valid_gencode.fasta \
    --exp_prefix MLM_EM \
    --encoding_method kmer \
    --k 3 \
    --data_dir data \
    --results_dir results

# 6-mer
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_human.fasta \
    sequences/valid_gencode.fasta \
    --exp_prefix MLM_EM \
    --encoding_method kmer \
    --k 6 \
    --data_dir data \
    --results_dir results

# 9-mer
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_human.fasta \
    sequences/valid_gencode.fasta \
    --exp_prefix MLM_EM \
    --encoding_method kmer \
    --k 9 \
    --data_dir data \
    --results_dir results

# BPE (vs=256)
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_human.fasta \
    sequences/valid_gencode.fasta \
    --exp_prefix MLM_EM \
    --encoding_method bpe \
    --bpe_file features/256.bpe \
    --data_dir data \
    --results_dir results

# BPE (vs=1024)
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_human.fasta \
    sequences/valid_gencode.fasta \
    --exp_prefix MLM_EM \
    --encoding_method bpe \
    --bpe_file features/1024.bpe \
    --data_dir data \
    --results_dir results

# BPE (vs=4096)
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_human.fasta \
    sequences/valid_gencode.fasta \
    --exp_prefix MLM_EM \
    --encoding_method bpe \
    --bpe_file features/4096.bpe \
    --data_dir data \
    --results_dir results

# CSE (kernel_size=3)
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_human.fasta \
    sequences/valid_gencode.fasta \
    --exp_prefix MLM_EM \
    --encoding_method cse \
    --kernel_size 3 \
    --data_dir data \
    --results_dir results

# CSE (kernel_size=4)
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_human.fasta \
    sequences/valid_gencode.fasta \
    --exp_prefix MLM_EM \
    --encoding_method cse \
    --kernel_size 4 \
    --data_dir data \
    --results_dir results

# CSE (kernel_size=6)
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_human.fasta \
    sequences/valid_gencode.fasta \
    --exp_prefix MLM_EM \
    --encoding_method cse \
    --kernel_size 6 \
    --data_dir data \
    --results_dir results

# CSE (kernel_size=7)
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_human.fasta \
    sequences/valid_gencode.fasta \
    --exp_prefix MLM_EM \
    --encoding_method cse \
    --kernel_size 7 \
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

# CSE (kernel_size=10)
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_human.fasta \
    sequences/valid_gencode.fasta \
    --exp_prefix MLM_EM \
    --encoding_method cse \
    --kernel_size 10 \
    --data_dir data \
    --results_dir results

# CSE (kernel_size=13)
python -m lncrnapy.scripts.pretrain \
    sequences/pretrain_human.fasta \
    sequences/valid_gencode.fasta \
    --exp_prefix MLM_EM \
    --encoding_method cse \
    --kernel_size 13 \
    --data_dir data \
    --results_dir results

# FINE-TUNING ------------------------------------------------------------------
# NUC
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_FT \
    --encoding_method nuc \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_EM_nuc_dm768_N12_bs8_ws32000_cl768_d0

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

# 6-mer
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_FT \
    --encoding_method kmer \
    --k 6 \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_EM_kmer_k6_dm768_N12_bs8_ws32000_cl768_d0

# 9-mer
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_FT \
    --encoding_method kmer \
    --k 9 \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_EM_kmer_k9_dm768_N12_bs8_ws32000_cl768_d0

# BPE (vs=256)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_FT \
    --encoding_method bpe \
    --bpe_file features/256.bpe \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_EM_bpe_vs256_dm768_N12_bs8_ws32000_cl768_d0

# BPE (vs=1024)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_FT \
    --encoding_method bpe \
    --bpe_file features/1024.bpe \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_EM_bpe_vs1024_dm768_N12_bs8_ws32000_cl768_d0

# BPE (vs=4096)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_FT \
    --encoding_method bpe \
    --bpe_file features/4096.bpe \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_EM_bpe_vs4096_dm768_N12_bs8_ws32000_cl768_d0

# CSE (kernel_size=3)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_FT \
    --encoding_method cse \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_EM_cse_nm768_sm3_dm768_N12_bs8_ws32000_cl768_d0

# CSE (kernel_size=4)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_FT \
    --encoding_method cse \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_EM_cse_nm768_sm4_dm768_N12_bs8_ws32000_cl768_d0

# CSE (kernel_size=6)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_FT \
    --encoding_method cse \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_EM_cse_nm768_sm6_dm768_N12_bs8_ws32000_cl768_d0

# CSE (kernel_size=7)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_FT \
    --encoding_method cse \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_EM_cse_nm768_sm7_dm768_N12_bs8_ws32000_cl768_d0

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

# CSE (kernel_size=10)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_FT \
    --encoding_method cse \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_EM_cse_nm768_sm10_dm768_N12_bs8_ws32000_cl768_d0

# CSE (kernel_size=13)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_FT \
    --encoding_method cse \
    --data_dir data \
    --results_dir results \
    --pretrained_model MLM_EM_cse_nm768_sm13_dm768_N12_bs8_ws32000_cl768_d0

# PROBING ----------------------------------------------------------------------
# NUC
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_PRB \
    --encoding_method nuc \
    --data_dir data \
    --results_dir results \
    --freeze_network \
    --learning_rate 0.0001 \
    --hidden_cls_layers 256 \
    --pretrained_model MLM_EM_nuc_dm768_N12_bs8_ws32000_cl768_d0

# 3-mer
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_PRB \
    --encoding_method kmer \
    --k 3 \
    --data_dir data \
    --results_dir results \
    --freeze_network \
    --learning_rate 0.0001 \
    --hidden_cls_layers 256 \
    --pretrained_model MLM_EM_kmer_k3_dm768_N12_bs8_ws32000_cl768_d0

# 6-mer
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_PRB \
    --encoding_method kmer \
    --k 6 \
    --data_dir data \
    --results_dir results \
    --freeze_network \
    --learning_rate 0.0001 \
    --hidden_cls_layers 256 \
    --pretrained_model MLM_EM_kmer_k6_dm768_N12_bs8_ws32000_cl768_d0

# 9-mer
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_PRB \
    --encoding_method kmer \
    --k 9 \
    --data_dir data \
    --results_dir results \
    --freeze_network \
    --learning_rate 0.0001 \
    --hidden_cls_layers 256 \
    --pretrained_model MLM_EM_kmer_k9_dm768_N12_bs8_ws32000_cl768_d0

# BPE (vs=256)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_PRB \
    --encoding_method bpe \
    --bpe_file features/256.bpe \
    --data_dir data \
    --results_dir results \
    --freeze_network \
    --learning_rate 0.0001 \
    --hidden_cls_layers 256 \
    --pretrained_model MLM_EM_bpe_vs256_dm768_N12_bs8_ws32000_cl768_d0

# BPE (vs=1024)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_PRB \
    --encoding_method bpe \
    --bpe_file features/1024.bpe \
    --data_dir data \
    --results_dir results \
    --freeze_network \
    --learning_rate 0.0001 \
    --hidden_cls_layers 256 \
    --pretrained_model MLM_EM_bpe_vs1024_dm768_N12_bs8_ws32000_cl768_d0

# BPE (vs=4096)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_PRB \
    --encoding_method bpe \
    --bpe_file features/4096.bpe \
    --data_dir data \
    --results_dir results \
    --freeze_network \
    --learning_rate 0.0001 \
    --hidden_cls_layers 256 \
    --pretrained_model MLM_EM_bpe_vs4096_dm768_N12_bs8_ws32000_cl768_d0

# CSE (kernel_size=3)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_PRB \
    --encoding_method cse \
    --data_dir data \
    --results_dir results \
    --freeze_network \
    --learning_rate 0.0001 \
    --hidden_cls_layers 256 \
    --pretrained_model MLM_EM_cse_nm768_sm3_dm768_N12_bs8_ws32000_cl768_d0

# CSE (kernel_size=4)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_PRB \
    --encoding_method cse \
    --data_dir data \
    --results_dir results \
    --freeze_network \
    --learning_rate 0.0001 \
    --hidden_cls_layers 256 \
    --pretrained_model MLM_EM_cse_nm768_sm4_dm768_N12_bs8_ws32000_cl768_d0

# CSE (kernel_size=6)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_PRB \
    --encoding_method cse \
    --data_dir data \
    --results_dir results \
    --freeze_network \
    --learning_rate 0.0001 \
    --hidden_cls_layers 256 \
    --pretrained_model MLM_EM_cse_nm768_sm6_dm768_N12_bs8_ws32000_cl768_d0

# CSE (kernel_size=7)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_PRB \
    --encoding_method cse \
    --data_dir data \
    --results_dir results \
    --freeze_network \
    --learning_rate 0.0001 \
    --hidden_cls_layers 256 \
    --pretrained_model MLM_EM_cse_nm768_sm7_dm768_N12_bs8_ws32000_cl768_d0

# CSE (kernel_size=9)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_PRB \
    --encoding_method cse \
    --data_dir data \
    --results_dir results \
    --freeze_network \
    --learning_rate 0.0001 \
    --hidden_cls_layers 256 \
    --pretrained_model MLM_EM_cse_nm768_sm9_dm768_N12_bs8_ws32000_cl768_d0

# CSE (kernel_size=10)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_PRB \
    --encoding_method cse \
    --data_dir data \
    --results_dir results \
    --freeze_network \
    --learning_rate 0.0001 \
    --hidden_cls_layers 256 \
    --pretrained_model MLM_EM_cse_nm768_sm10_dm768_N12_bs8_ws32000_cl768_d0

# CSE (kernel_size=13)
python -m lncrnapy.scripts.train \
    sequences/finetune_human_pcrna.fasta \
    sequences/finetune_human_ncrna.fasta \
    sequences/valid_human_pcrna.fasta \
    sequences/valid_human_ncrna.fasta \
    --exp_prefix CLS_EM_PRB \
    --encoding_method cse \
    --data_dir data \
    --results_dir results \
    --freeze_network \
    --learning_rate 0.0001 \
    --hidden_cls_layers 256 \
    --pretrained_model MLM_EM_cse_nm768_sm13_dm768_N12_bs8_ws32000_cl768_d0