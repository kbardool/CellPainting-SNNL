import torch
from snnl.loss import pairwise_euclidean_distance
from snnl.loss import pairwise_cosine_distance

torch.manual_seed(42)


def test_pairwise_euclidean_distance():
    a = torch.rand((4, 2))
    b = torch.rand((4, 2))
    distance = pairwise_euclidean_distance(a, b)
    assert distance.size() == (4, 4)
    assert type(distance) is torch.Tensor

def test_pairwise_cosine_distance():
    a = torch.rand((4, 2))
    b = torch.rand((4, 2))
    distance = pairwise_cosine_distance(a, b)
    assert distance.size() == (4, 4)
    assert type(distance) is torch.Tensor

