from typing import Optional


class EarlyStopping:
    """
    Early stopping based on validation AUC.
    """

    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self.best_score: Optional[float] = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            return False

        if val_score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

        return self.early_stop
