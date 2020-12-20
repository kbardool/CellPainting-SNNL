# Soft Nearest Neighbor Loss

## Overview

In the context of this work, _entanglement_ is defined as to how close class-similar data points to each other compared to class-different data points. A low entanglement means that class-similar data points are much closer to each other than class-different data points. Having such a set of data points will render downstream tasks much easier to accomplish with an even better performance. To measure the entanglement of data points, Frosst et al. (2019) expanded the non-linear neighborhood components analysis (NCA) (Salakhutdinov and Hinton, 2007) objective by introducing the temperature factor _T_, and called this modified objective the _soft nearest neighbor loss_. In addition, this loss function is computed for each of the hidden layers of a deep neural network as opposed to non-linear NCA which was only computed on the latent code of an autoencoder network.

## Usage

## Results

| Model | MNIST (Average) | MNIST (Best) | Fashion-MNIST (Average) | Fashion-MNIST (Best) | KMNIST (Average) | KMNIST (Best) |
| ----- | --------------- | ------------ | ----------------------- | -------------------- | ---------------- | ------------- |


## References

- Frosst, Nicholas, Nicolas Papernot, and Geoffrey Hinton. "Analyzing and improving representations with the soft nearest neighbor loss." arXiv preprint arXiv:1902.01889 (2019).
