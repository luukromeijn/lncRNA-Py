import pytest
from rhythmnblues.modules import Classifier
from rhythmnblues.modules import MycoAICNN
from rhythmnblues.features import KmerFreqs


def test_model(data):
    kmers = KmerFreqs(3)
    model = Classifier(MycoAICNN())
    data.calculate_feature(kmers)
    data.set_tensor_features(kmers.name)
    model.predict(data)

# TODO more unittests required to ensure proper behaviour of WrapperBase