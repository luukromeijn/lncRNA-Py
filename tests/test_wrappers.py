import pytest
import torch
from rhythmnblues.modules import Classifier, MLM
from rhythmnblues.modules.wrappers import WrapperBase
from rhythmnblues.modules import MycoAICNN, BERT
from rhythmnblues.features import KmerFreqs, KmerTokenizer


def test_wrapper_base(data):
    encoder = KmerTokenizer(3)
    data.calculate_feature(encoder)
    data.set_tensor_features(encoder.name, torch.long)
    model = WrapperBase(BERT(encoder.vocab_size))
    model.predict(data)
    model.latent_space(data)

def test_classifier_cnn(data):
    encoder = KmerFreqs(3)
    data.calculate_feature(encoder)
    data.set_tensor_features(encoder.name)
    model = WrapperBase(MycoAICNN())
    model.predict(data)
    model.latent_space(data, CLS_only=False)

def test_classifier_bert(data):
    encoder = KmerTokenizer(3)
    data.calculate_feature(encoder)
    data.set_tensor_features(encoder.name, torch.long)
    model = Classifier(BERT(encoder.vocab_size))
    model.predict(data)
    model.latent_space(data)

def test_classifier_return_logits(data):
    encoder = KmerTokenizer(3)
    data.calculate_feature(encoder)
    data.set_tensor_features(encoder.name, torch.long)
    model = Classifier(BERT(encoder.vocab_size))
    pred_probs = model.predict(data, return_logits=False)
    pred_logits = model.predict(data, return_logits=True)
    assert (torch.sigmoid(pred_logits) == pred_probs).sum() == len(data)

def test_mlm(data):
    encoder = KmerTokenizer(3)
    data.calculate_feature(encoder)
    data.set_tensor_features(encoder.name, torch.long)
    model = MLM(BERT(encoder.vocab_size))
    model.predict(data)
    model.latent_space(data)