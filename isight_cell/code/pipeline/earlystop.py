"""Early stopping on a validation metric: min_delta / patience / keep-current.

This is the rule declared for the released staining model (see docs/CODE_AUDIT.md §8):

    criterion   validation macro-F1  (mean of the per-cell intensity and location F1)
    min_delta   0.001   (0.1%)
    patience    2
    semantics   keep-current

**keep-current** means the epoch that exhausts the patience budget is the selected model --
not the best-so-far. This is Keras' `restore_best_weights=False` convention, and it is the
convention the released checkpoint follows; `restore_best` would select a different epoch.
An epoch counts as an improvement only if it beats the running best by more than `min_delta`,
so a marginal gain still increments the bad-epoch counter.
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
