"""Early stopping on a validation metric.

An epoch counts as an improvement only if it beats the running best by more than `min_delta`;
otherwise it increments the bad-epoch counter, and training stops once `patience` bad epochs
accumulate. The epoch at which training stops is the selected model.
"""


class EarlyStopper:
    def __init__(self, min_delta=0.001, patience=2):
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.best = None
        self.bad = 0

    def update(self, value):
        """Feed one epoch's validation metric. Returns True when training should stop."""
        value = float(value)
        if self.best is None or value > self.best + self.min_delta:
            self.best = value
            self.bad = 0
        else:
            self.bad += 1
        return self.bad >= self.patience

    def replay(self, values):
        """Rebuild the counter state from a resumed run's history. Returns the stop flag."""
        stop = False
        for v in values:
            stop = self.update(v)
        return stop

    def status(self):
        b = "-inf" if self.best is None else f"{self.best:.6f}"
        return f"best={b} bad={self.bad}/{self.patience} min_delta={self.min_delta}"
