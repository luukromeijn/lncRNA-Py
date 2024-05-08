import pytest
from rhythmnblues.modules import Model
from rhythmnblues.modules import MycoAICNN
from rhythmnblues.features import KmerFreqs


def test_model(data):
    kmers = KmerFreqs(3)
    model = Model(MycoAICNN())
    data.calculate_feature(kmers)
    data.set_tensor_features(kmers.name)
    model.predict(data)