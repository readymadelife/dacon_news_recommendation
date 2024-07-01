import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import numpy as np


def train(model, train_loader, optimizer, epoch, **kwargs):
    model.train()
    train_loss = 0
    for batch_idx, inputs in enumerate(train_loader):
        optimizer.zero_grad()

        _, loss = model(inputs, **kwargs)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    print(
        f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}"
    )


def eval(model, test_loader, **kwargs):
    model.eval()
    recall = []
    with torch.no_grad():
        for inputs, actual in test_loader:
            predict = model(inputs, calculate_loss=False).cpu()
            recall += recall_at_k(predict=predict, actual=actual)
    mean_recall = f"{(sum(recall) / len(recall)):.4f}"
    return mean_recall


def recall_at_k(predict, actual):
    recalls = []
    for idx in range(len(actual)):
        actual_set = set(actual[idx].cpu().numpy())
        top_k_predictions = np.argsort(-predict[idx].numpy())[:5]
        predict_set = set(top_k_predictions.tolist())

        if len(actual_set) > 0:
            recall = len(actual_set & predict_set) / len(actual_set)
            recalls.append(recall)
    return recalls
