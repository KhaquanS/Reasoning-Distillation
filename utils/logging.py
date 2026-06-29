import csv
from pathlib import Path


class AveragedCSVLogger:
    """
    Writes averaged scalar metrics to CSV at a target number of rows per epoch.
    """
    def __init__(self, log_path, steps_per_epoch, target_rows_per_epoch=1000, append=False):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.steps_per_epoch = steps_per_epoch
        self.target_rows_per_epoch = max(1, int(target_rows_per_epoch))
        self.log_interval = max(1, -(-steps_per_epoch // self.target_rows_per_epoch))
        self._loss_sum = 0.0
        self._num_steps = 0

        mode = "a" if append and self.log_path.exists() else "w"
        self._file = self.log_path.open(mode, newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "epoch",
                "step_start",
                "step_end",
                "num_steps",
                "avg_loss",
                "lr",
            ],
        )
        if mode == "w":
            self._writer.writeheader()
            self._file.flush()

    def add(self, epoch, step, loss, lr, force=False):
        self._loss_sum += float(loss)
        self._num_steps += 1

        if self._num_steps < self.log_interval and not force:
            return

        step_end = step + 1
        step_start = step_end - self._num_steps + 1
        self._writer.writerow({
            "epoch": epoch + 1,
            "step_start": step_start,
            "step_end": step_end,
            "num_steps": self._num_steps,
            "avg_loss": self._loss_sum / self._num_steps,
            "lr": lr,
        })
        self._file.flush()
        self._loss_sum = 0.0
        self._num_steps = 0

    def close(self):
        self._file.close()
