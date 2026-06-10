import numpy as np
    

class EarlyStopper:
    """Very small early‑stopping helper.

    Args:
        patience: Number of *consecutive* epochs without an improvement on the
            monitored metric before stopping.
        min_delta: Minimal change to qualify as an improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float("inf")

    def __call__(self, value: float) -> bool:
        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience