import torch


def pairwise_euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Returns the pairwise Euclidean distance between matrices `a` and `b`.

    Parameters
    ----------
    a, b : torch.Tensor
        The input features.

    Returns
    -------
    torch.Tensor
        The pairwise Euclidean distance between matrices `a` and `b`.
    """
    batch_size_a = a.size()[0]
    batch_size_b = b.size()[0]
    squared_norm_a = torch.sum(torch.pow(a, 2), dim=1)
    squared_norm_a = torch.reshape(squared_norm_a, [1, batch_size_a])
    squared_norm_b = torch.sum(torch.pow(b, 2), dim=1)
    squared_norm_b = torch.reshape(squared_norm_b, [batch_size_b, 1])

    a = torch.t(a)
    inner_product = torch.matmul(b, a)

    tile_a = (
        squared_norm_a.view(-1, 1)
        .repeat(batch_size_b, 1)
        .view((batch_size_b * squared_norm_a.size()[0]), -1)
    )
    tile_b = (
        squared_norm_b.view(-1, 1)
        .repeat(1, batch_size_a)
        .view((batch_size_a * squared_norm_b.size()[0]), -1)
    )
    return tile_a + tile_b - 2.0 * inner_product

