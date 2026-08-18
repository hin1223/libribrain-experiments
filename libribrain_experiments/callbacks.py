import torch
from pytorch_lightning import Callback
from libribrain_experiments.utils import run_validation


class _ConditionedEvalModule:
    """Wraps a module needing FiLM conditioning c=1/sqrt(n_eval) so it
    exposes the plain module(x) interface run_validation() expects."""

    def __init__(self, module, c):
        self.module = module
        self.c = c

    def eval(self):
        self.module.eval()

    @property
    def device(self):
        return self.module.device

    def __call__(self, x):
        return self.module(x, self.c.expand(x.size(0), -1))


class TestMetricsCallback(Callback):
    """Evaluates on the test set at the end of every validation epoch,
    logging under a "test_" prefix (test_loss, test_f1_macro, etc.)
    alongside the existing per-epoch val_* metrics.

    Deliberately does not touch Trainer.fit()'s val_dataloaders argument
    or any validation_step — this is a separate hook, so val_loss-based
    checkpoint selection (ModelCheckpoint(monitor="val_loss")) and the
    existing per-epoch val_* logging are completely unaffected. Reuses
    the same run_validation() utility (naive baselines, per-class
    metrics, etc.) that the post-training val/test evaluation uses, for
    consistency, instead of a stripped-down loss/f1 pair.
    """

    def __init__(self, test_loader, labels, samples_per_class=None, baseline_only=False):
        self.test_loader = test_loader
        self.labels = labels
        self.samples_per_class = samples_per_class
        self.baseline_only = baseline_only

    def _eval_iterable(self, pl_module):
        if self.baseline_only:
            yield from self.test_loader
        else:
            for student_x, _, y in self.test_loader:
                if hasattr(pl_module, "_average_student"):
                    student_x = pl_module._average_student(student_x, pl_module.n_eval)
                yield [student_x, y]

    def on_validation_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module, "_conditioning"):
            c = pl_module._conditioning(pl_module.n_eval)
            eval_module = _ConditionedEvalModule(pl_module, c)
        else:
            eval_module = pl_module

        with torch.no_grad():
            result, *_ = run_validation(
                self._eval_iterable(pl_module), eval_module, self.labels,
                samples_per_class=self.samples_per_class, prefix="test")
        cm_key = next(k for k in result if k.endswith("_cm"))
        del result[cm_key]
        for key, value in result.items():
            pl_module.log(key, value, add_dataloader_idx=False)
