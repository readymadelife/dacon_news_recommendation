import torch


class EarlyStopping:
    def __init__(self, patience=7, delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping couter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases"""
        if val_loss < self.val_loss_min:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving Model..."
            )
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
