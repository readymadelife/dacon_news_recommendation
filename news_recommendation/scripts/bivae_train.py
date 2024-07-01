import random
import numpy as np

import torch
import pandas as pd
from scipy.sparse import csr_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.optim as optim

from news_recommendation.models import bivae
from news_recommendation.utils import (
    train_test_split,
    dataset,
    early_stopping,
    augmentation,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train(model, train_loader, optimizer, epoch, loss_function, num_items):
    model.train()
    train_loss = 0
    for batch_idx, (user, item) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(user, item)
        loss = loss_function(recon_batch, item, mu, logvar, num_items)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]"
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}"
            )
        # print(
        #     f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}"
        # )


def recall_at_k(actual, predicted, k):
    recalls = []
    for i in range(len(actual)):
        actual_set = set(actual[i].cpu().numpy())
        predicted_avg = predicted[i].mean(axis=0)
        top_k_predictions = np.argsort(-predicted_avg.cpu().numpy())[:k]
        predicted_set = set(top_k_predictions.tolist())

        if len(actual_set) > 0:
            recall = len(actual_set & predicted_set) / len(actual_set)
            recalls.append(recall)
    return torch.tensor(recalls)


def test(model, test_loader, loss_function, num_items):
    model.eval()
    test_loss = 0
    recalls = []
    with torch.no_grad():
        for user, item, actual in test_loader:
            user = user.unsqueeze(1).expand(-1, num_items)
            recon, mu, logvar = model(user, item)
            test_loss += loss_function(recon, item, mu, logvar, num_items).item()
            recalls.append(recall_at_k(actual=actual, predicted=recon, k=5))
    test_loss /= len(test_loader.dataset)
    print(f"====> Test set loss: {test_loss:.4f}")

    recalls = torch.cat(recalls, dim=0)
    mean_recall = recalls.mean().item()
    return test_loss, mean_recall


def train_and_evaluate(
    model,
    epochs,
    train_loader,
    test_loader,
    optimizer,
    loss_function,
    scheduler,
    num_items,
    early_stopping=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(1, epochs + 1):
        train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            loss_function=loss_function,
            num_items=num_items,
        )
        test_loss, recall = test(
            model=model,
            test_loader=test_loader,
            loss_function=loss_function,
            num_items=num_items,
        )
        early_stopping(val_loss=test_loss, model=model)
        print(f"recall@5: {recall}")
        scheduler.step(test_loss)

        if early_stopping.early_stop:
            break


if __name__ == "__main__":

    BATCH_SIZE = 64
    EPOCHS = 100

    set_seed(42)
    data = pd.read_csv("news_recommendation/data/view_log.csv")

    user_encoder = dataset.load_label_encoder(
        pkl_path="news_recommendation/utils/user_encoder.pkl", data=data["userID"]
    )
    article_encoder = dataset.load_label_encoder(
        pkl_path="news_recommendation/utils/article_encoder.pkl", data=data["articleID"]
    )

    data["userID"] = user_encoder.transform(data["userID"])
    data["articleID"] = article_encoder.transform(data["articleID"])

    train_data, test_data, all_articles = (
        train_test_split.transfrom_and_split_test_and_train_data(data)
    )

    # train_data = augmentation.augmentate_data(
    #     data=train_data, target_label="articleID", all_items=all_articles
    # )

    test_data = train_test_split.split_y_from_data(data=test_data, label="articleID")

    train_data = train_test_split.inverse_transform_data(train_data, "articleID")

    num_users = train_data.userID.nunique()
    num_items = train_data.articleID.nunique()

    train_dataset = dataset.BiVAEDataset(
        train_data.userID.tolist(),
        train_data.articleID.tolist(),
        num_users,
        num_items,
        mode="train",
    )
    test_dataset = dataset.BiVAEDataset(
        test_data.userID.tolist(),
        test_data.y.tolist(),
        num_users=num_users,
        num_items=num_items,
        mode="valid",
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = bivae.BiVAE(
        num_users=num_users,
        num_items=num_items,
        user_dim=1024,
        item_dim=1024,
        latent_dim=256,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = bivae.vae_loss_function

    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=6)
    early_stopping = early_stopping.EarlyStopping(path="./checkpoint_bivae_no_aug.pkl")

    train_and_evaluate(
        model=model,
        epochs=EPOCHS,
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        optimizer=optimizer,
        loss_function=criterion,
        scheduler=scheduler,
        early_stopping=early_stopping,
        num_items=num_items,
    )
