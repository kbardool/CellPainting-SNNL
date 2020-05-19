import pytest
import torch

from snnl.utils.data import load_dataset, create_dataloader


@pytest.mark.parametrize("name", ["mnist", "fashion_mnist", "emnist"])
def test_load_data(name):
    train_data, test_data = load_dataset(name)
    assert isinstance(train_data, torch.utils.data.Dataset)
    assert isinstance(test_data, torch.utils.data.Dataset)
    assert isinstance(train_data.data, torch.Tensor)
    assert isinstance(train_data.targets, torch.Tensor)
    assert isinstance(test_data.data, torch.Tensor)
    assert isinstance(test_data.targets, torch.Tensor)
    assert train_data.data.shape[1:] == (28, 28)
    assert test_data.data.shape[1:] == (28, 28)
