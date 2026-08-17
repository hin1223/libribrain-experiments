import torch
import torch.nn.functional as F
from .distillation_module import DistillationModule
from libribrain_experiments.stochastic_averaging import (
    sample_n, sample_n_inverted, sample_n_uniform, average_trials,
)


class StochasticDistillationModule(DistillationModule):
    """Distillation with stochastic student averaging and FiLM SNR conditioning.

    During training, N is sampled per batch from a noise-std-uniform distribution
    with mode at n_min. The conditioning signal c = 1/sqrt(N) is passed to any
    FiLM layers in the model.

    At eval/test time N is fixed at n_eval (default = n_min).
    """

    def __init__(self, model_config, n_classes, optimizer_config, loss_config,
                 teacher_checkpoint_path, temperature=2.0, alpha=0.5,
                 n_min=50, n_max=100, n_eval=50, channels_per_sample=306,
                 snr_weighted_kd=False, deterministic_cycling=False,
                 teacher_confidence_gated_kd=False, sampling_mode="default"):
        super().__init__(
            model_config, n_classes, optimizer_config, loss_config,
            teacher_checkpoint_path, temperature=temperature, alpha=alpha,
        )
        self.save_hyperparameters()
        self.n_min = n_min
        self.n_max = n_max
        self.n_eval = n_eval
        self.channels_per_sample = channels_per_sample
        self.snr_weighted_kd = snr_weighted_kd
        self.deterministic_cycling = deterministic_cycling
        self.teacher_confidence_gated_kd = teacher_confidence_gated_kd
        self.sampling_mode = sampling_mode

    def _conditioning(self, n: int) -> torch.Tensor:
        """c = 1/sqrt(N), shape (1, 1) — broadcast over batch in FiLM."""
        return torch.tensor([[1.0 / (n ** 0.5)]], dtype=torch.float32, device=self.device)

    def _average_student(self, raw_trials, n):
        """Average n randomly selected trials from raw_trials."""
        return average_trials(raw_trials, n, self.channels_per_sample)

    def _effective_alpha(self, n: int) -> float:
        """Scale alpha by how close n is to n_max (the teacher's fixed view).

        At n=n_min (noisiest, most mismatched with the teacher's constant
        high-SNR target) this is pure CE; at n=n_max (student input matches
        what the teacher saw) it gets the full nominal alpha weight. No-op
        (returns self.alpha unchanged) unless snr_weighted_kd is enabled.
        """
        if not self.snr_weighted_kd:
            return self.alpha
        snr_weight = (n - self.n_min) / (self.n_max - self.n_min)
        return self.alpha * snr_weight

    def _sample_n(self) -> int:
        """Pick this step's averaging count.

        deterministic_cycling (if set) takes priority: a deterministic
        sawtooth sweep through n_min..n_max indexed by global_step — same
        range of SNR variation as random sampling, but predictable rather
        than random, so the same training step always sees the same n.

        Otherwise, sampling_mode selects the random distribution:
          - "default": mode at n_min, tail toward n_max (original behavior).
          - "inverted": mirror of default — mode at n_max (tending toward
            the teacher's fixed high-SNR view) instead of n_min.
          - "uniform": flat over [n_min, n_max], no mode bias either way.
        """
        if self.deterministic_cycling:
            span = self.n_max - self.n_min
            return self.n_min if span <= 0 else self.n_min + (self.global_step % (span + 1))
        if self.sampling_mode == "inverted":
            return int(sample_n_inverted(1, self.n_min, self.n_max)[0])
        if self.sampling_mode == "uniform":
            return int(sample_n_uniform(1, self.n_min, self.n_max)[0])
        return int(sample_n(1, self.n_min, self.n_max)[0])

    def _kd_loss(self, student_logits, teacher_logits):
        """Per-example KD loss, optionally gated by the teacher's own
        prediction confidence (max softmax prob) — trust the teacher's soft
        labels less on examples it isn't confident about, rather than
        weighting every example uniformly. Equivalent to the original
        reduction='batchmean' formula when teacher_confidence_gated_kd is
        off.
        """
        T = self.temperature
        per_example_kd = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='none',
        ).sum(dim=1) * (T ** 2)
        if self.teacher_confidence_gated_kd:
            confidence = F.softmax(teacher_logits, dim=1).max(dim=1).values
            return (per_example_kd * confidence).mean()
        return per_example_kd.mean()

    def training_step(self, batch, batch_idx):
        raw_student, teacher_x, y = batch[0], batch[1], batch[2]

        n = self._sample_n()
        student_x = self._average_student(raw_student, n)
        c = self._conditioning(n).expand(student_x.size(0), -1)

        with torch.no_grad():
            teacher_logits = self.teacher(teacher_x)
        student_logits = self(student_x, c)

        ce_loss = self.loss_fn(student_logits, y)
        kd_loss = self._kd_loss(student_logits, teacher_logits)
        effective_alpha = self._effective_alpha(n)
        loss = effective_alpha * kd_loss + (1 - effective_alpha) * ce_loss

        self.log('train_loss', loss)
        self.log('train_kd_loss', kd_loss)
        self.log('train_ce_loss', ce_loss)
        self.log('train_n', float(n))
        self.log('train_effective_alpha', effective_alpha)
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
