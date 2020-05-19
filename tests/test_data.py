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


@pytest.mark.parametrize("name", ["mnist", "fashion_mnist", "emnist"])
def test_create_dataloader(name):
    batch_size = 256
    train_data, test_data = load_dataset(name)
    train_loader = create_dataloader(train_data, batch_size=batch_size)
    test_loader = create_dataloader(test_data, batch_size=batch_size)
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)

    for batch_features, batch_labels in train_loader:
        assert isinstance(batch_features, torch.Tensor)
        assert isinstance(batch_labels, torch.Tensor)
        assert batch_features.size() == (256, 1, 28, 28)
        break

    for batch_features, batch_labels in test_loader:
        assert isinstance(batch_features, torch.Tensor)
        assert isinstance(batch_labels, torch.Tensor)
        assert batch_features.size() == (256, 1, 28, 28)
        break
