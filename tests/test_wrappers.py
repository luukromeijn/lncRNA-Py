import pytest
import torch
from lncrnapy.modules import Classifier, MaskedTokenModel, Regressor
from lncrnapy.modules.wrappers import WrapperBase
from lncrnapy.modules import MycoAICNN, BERT
from lncrnapy.features import KmerFreqs, KmerTokenizer


def test_wrapper_base(data):
    data = data.sample(N=2)
    encoder = KmerTokenizer(3)
    data.calculate_feature(encoder)
    data.set_tensor_features(encoder.name, torch.long)
    base_arch = BERT(encoder.vocab_size, N=1)
    model = WrapperBase(base_arch.config, base_arch)
    model.predict(data)
    model.latent_space(data, pooling='CLS', dim_red=None)

def test_classifier_cnn(data):
    encoder = KmerFreqs(3)
    data.calculate_feature(encoder)
    data.set_tensor_features(encoder.name)
    base_arch = MycoAICNN()
    model = Classifier(base_arch.config, base_arch)
    model.predict(data)
    model.latent_space(data, dim_red=None)

def test_classifier_bert(data):
    encoder = KmerTokenizer(3)
    data.calculate_feature(encoder)
    data.set_tensor_features(encoder.name, torch.long)
    base_arch = BERT(encoder.vocab_size, N=1)
    model = Classifier(base_arch.config, base_arch)
    model.predict(data)
    model.latent_space(data, pooling='CLS', dim_red=None)

def test_classifier_return_logits(data):
    data = data.sample(N=2)
    encoder = KmerTokenizer(3)
    data.calculate_feature(encoder)
    data.set_tensor_features(encoder.name, torch.long)
    base_arch = BERT(encoder.vocab_size, N=1)
    model = Classifier(base_arch.config, base_arch)
    pred_probs = model.predict(data, return_logits=False)
    pred_logits = model.predict(data, return_logits=True)
    assert (torch.sigmoid(pred_logits) == pred_probs).sum() == len(data)

def test_mlm(data):
    data = data.sample(N=2)
    encoder = KmerTokenizer(3)
    data.calculate_feature(encoder)
    data.set_tensor_features(encoder.name, torch.long)
    base_arch = BERT(encoder.vocab_size, N=1)
    model = MaskedTokenModel(base_arch.config, base_arch)
    model.predict(data)
    model.latent_space(data, pooling='CLS', dim_red=None)

def test_regressor_cnn(data):
    encoder = KmerFreqs(3)
    data.calculate_feature(encoder)
    data.set_tensor_features(encoder.name)
    base_arch = MycoAICNN()
    model = Regressor(base_arch.config, base_arch)
    model.predict(data)
    model.latent_space(data, dim_red=None)

def test_regressor_bert(data):
    data = data.sample(N=2)
    encoder = KmerTokenizer(3)
    data.calculate_feature(encoder)
    data.set_tensor_features(encoder.name, torch.long)
    base_arch = BERT(encoder.vocab_size, N=1)
    model = Regressor(base_arch.config, base_arch)
    model.predict(data)
    model.latent_space(data, pooling='CLS', dim_red=None)

def test_latent_space_max_pooling(data):
    data = data.sample(N=2)
    encoder = KmerTokenizer(3)
    data.calculate_feature(encoder)
    data.set_tensor_features(encoder.name, torch.long)
    base_arch = BERT(encoder.vocab_size, N=1)
    model = MaskedTokenModel(base_arch.config, base_arch)
    model.latent_space(data, pooling='max', dim_red=None)

def test_latent_space_mean_pooling(data):
    data = data.sample(N=2)
    encoder = KmerTokenizer(3)
    data.calculate_feature(encoder)
    data.set_tensor_features(encoder.name, torch.long)
    base_arch = BERT(encoder.vocab_size, N=1)
    model = MaskedTokenModel(base_arch.config, base_arch)
    model.latent_space(data, pooling='mean', dim_red=None)