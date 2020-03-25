import torch


def accuracy(model, dataset):
    correct = 0
    total = 0

    with torch.no_grad():
        for (features, labels) in dataset:
            features = features.view(features.shape[0], -1)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total)
