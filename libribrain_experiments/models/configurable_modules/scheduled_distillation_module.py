import torch
import torch.nn.functional as F
from .stochastic_distillation_module import StochasticDistillationModule


class ScheduledDistillationModule(StochasticDistillationModule):
    """Distillation with linearly scheduled student averaging.

    At epoch 0 the student sees n_max averaged trials (same SNR as teacher).
    Each epoch, N decreases linearly until reaching n_min at the final epoch.
    Evaluation is fixed at n_eval.
    """

    def _scheduled_n(self) -> int:
        max_epochs = self.trainer.max_epochs
        epoch = self.current_epoch
        n = self.n_max - epoch * (self.n_max - self.n_min) / (max_epochs - 1)
        return int(round(n))

    def training_step(self, batch, batch_idx):
        raw_student, teacher_x, y = batch[0], batch[1], batch[2]

        n = self._scheduled_n()
        student_x = self._average_student(raw_student, n)
        c = self._conditioning(n).expand(student_x.size(0), -1)

        with torch.no_grad():
            teacher_logits = self.teacher(teacher_x)
        student_logits = self(student_x, c)

        ce_loss = self.loss_fn(student_logits, y)
        T = self.temperature
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean',
        ) * (T ** 2)
        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        self.log('train_loss', loss)
        self.log('train_kd_loss', kd_loss)
        self.log('train_ce_loss', ce_loss)
        self.log('train_n', float(n))
        self.log('train_f1_macro', self.f1_macro(student_logits, y))
        self.log('train_bal_acc', self.balanced_accuracy(student_logits, y))
        return loss
