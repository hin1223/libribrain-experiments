import torch
import torch.nn.functional as F
from .distillation_module import DistillationModule
from libribrain_experiments.stochastic_averaging import sample_n, average_trials


class StochasticDistillationModule(DistillationModule):
    """Distillation with stochastic student averaging and FiLM SNR conditioning.

    During training, N is sampled per batch from a noise-std-uniform distribution
    with mode at n_min. The conditioning signal c = 1/sqrt(N) is passed to any
    FiLM layers in the model.

    At eval/test time N is fixed at n_eval (default = n_min).
    """

    def __init__(self, model_config, n_classes, optimizer_config, loss_config,
                 teacher_checkpoint_path, temperature=2.0, alpha=0.5,
                 n_min=50, n_max=100, n_eval=50, channels_per_sample=306):
        super().__init__(
            model_config, n_classes, optimizer_config, loss_config,
            teacher_checkpoint_path, temperature=temperature, alpha=alpha,
        )
        self.n_min = n_min
        self.n_max = n_max
        self.n_eval = n_eval
        self.channels_per_sample = channels_per_sample

    def _conditioning(self, n: int) -> torch.Tensor:
        """c = 1/sqrt(N), shape (1, 1) — broadcast over batch in FiLM."""
        return torch.tensor([[1.0 / (n ** 0.5)]], dtype=torch.float32, device=self.device)

    def _average_student(self, raw_trials, n):
        """Average n randomly selected trials from raw_trials."""
        return average_trials(raw_trials, n, self.channels_per_sample)

    def training_step(self, batch, batch_idx):
        raw_student, teacher_x, y = batch[0], batch[1], batch[2]

        n = int(sample_n(1, self.n_min, self.n_max)[0])
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

    def validation_step(self, batch, batch_idx):
        raw_student, teacher_x, y = batch[0], batch[1], batch[2]

        student_x = self._average_student(raw_student, self.n_eval)
        c = self._conditioning(self.n_eval).expand(student_x.size(0), -1)
        student_logits = self(student_x, c)

        loss = self.loss_fn(student_logits, y)
        self.log('val_loss', loss)
        self.log('val_f1_macro', self.f1_macro(student_logits, y))
        self.log('val_bal_acc', self.balanced_accuracy(student_logits, y))
        return loss
