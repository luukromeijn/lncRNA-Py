'''
The initial implementation of lncRNA-BERT had to be modified to allow upload
and storage on HuggingFace. 

Here's what to do if you wish to convert an 'old' model into a HuggingFace 
compatible format.

1) Rename lncrnapy/modules/wrappers.py with to new_wrappers.py
2) Rename lncrnapy/modules/bert.py to new_bert.py
3) Create new lncrnapy/modules/wrappers.py and bert.py
4) Populate these files with code from an earlier commit:
   https://github.com/luukromeijn/lncRNA-Py/tree/1601063722a750e204d204081aa532572aead8da/lncrnapy/modules
   When unpickling the old models, Python will now call upon these older files.
5) Run the script below. You might need to replace some filanames
6) After running the script, you can delete the older bert.py and wrappers.py 
   and revert new_bert.py and new_wrappers.py to their original names.

Note that we use some test sequences to verify that the predicted output stays the same.
'''

from lncrnapy.modules.new_bert import CSEBERT, BERT
from lncrnapy.modules.new_wrappers import MaskedConvModel, MaskedTokenModel, Classifier
from lncrnapy.data import Data
from lncrnapy.features import KmerTokenizer
import torch

# model_name, model_file = 'lncRNA-BERT-kmer-k3-pretrained', 'data/models/MLM_LONG_kmer_k3_dm768_N12_bs8_ws32000_cl768_d0.pt'
# model_name, model_file = 'lncRNA-BERT-kmer-k3-finetuned' , 'data/models/CLSv2_LONG_FT_kmer_finetuned_k3_dm768_N12_dff3072_h12_bs8_lr1e-05_wd0_cl768_d0.pt'
model_name, model_file = 'lncRNA-BERT-CSE-k9-pretrained' , 'data/models/MLM_LONG_conv_nm768_sm9_dm768_N12_bs8_ws32000_cl768_d0.pt'
# model_name, model_file = 'lncRNA-BERT-CSE-k9-finetuned'  , 'data/models/CLSv2_LONG_FT_conv_finetuned_nm768_sm9_dm768_N12_dff3072_h12_bs8_lr1e-05_wd0_cl768_d0.pt'

# Loading some test data (this will verify that our models behave the same after conversion)
data = Data('tests/data/pcrna.fasta').sample(N=2)
# K-mer
# kmers = KmerTokenizer(3)
# data.calculate_feature(kmers)
# data.set_tensor_features(kmers.name, torch.long)
# CSE
data.set_tensor_features('4D-DNA')

# Loading original model, save state dict
model = torch.load(model_file, map_location=torch.device('cpu'))
state_dict = model.state_dict()
print(model.predict(data)[0,0])

# Initializing new class
# base_arch = BERT(kmers.vocab_size) # K-mer
base_arch = CSEBERT(768, kernel_size=9) # CSE
# - For pretrained
# model = MaskedTokenModel(base_arch.config, base_arch) # K-mer
model = MaskedConvModel(base_arch.config, base_arch) # CSE
# - For finetuned
# model = Classifier(base_arch.config, base_arch)
# Loading state dict of original model
model.load_state_dict(state_dict)
print(model.predict(data)[0,0])

# save locally
model.save_pretrained(model_name)
# push to the hub
model.push_to_hub(f"luukromeijn/{model_name}")

# reload
# model = MaskedTokenModel.from_pretrained(f"luukromeijn/{model_name}") # pre-trained, k-mer
model = MaskedConvModel.from_pretrained(f"luukromeijn/{model_name}") # pre-trained, CSE
# model = Classifier.from_pretrained(f"luukromeijn/{model_name}") # classifier, k-mer
print(model.predict(data)[0,0])