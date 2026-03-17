import numpy as np


class EarlyStopping:
    """
    Stop training when the validation score does not improve for a given number of epochs.

    Args:
        patience (int): Number of epochs to wait without improvement.
        verbose (bool): Whether to print improvement / counter messages.
        delta (float): Minimum increase in validation score to qualify as an improvement.
    """

    def __init__(self, patience=10, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            if self.verbose:
                print(f"[EarlyStopping] Initial best score: {val_score:.6f}")
            return

        if val_score > self.best_score + self.delta:
            self.best_score = val_score
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] Score improved to {val_score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"[EarlyStopping] No improvement "
                    f"({self.counter}/{self.patience}) | "
                    f"best: {self.best_score:.6f}, current: {val_score:.6f}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("[EarlyStopping] Patience exceeded. Stopping training.")