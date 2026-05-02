import torch
import torch.nn.functional as F
from .classification_module import ClassificationModule


class DistillationModule(ClassificationModule):
    def __init__(self, model_config, n_classes, optimizer_config, loss_config,
                 teacher_checkpoint_path, temperature=2.0, alpha=0.5):
        super().__init__(model_config, n_classes, optimizer_config, loss_config)
        self.save_hyperparameters()
        teacher = ClassificationModule.load_from_checkpoint(teacher_checkpoint_path)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha

    def training_step(self, batch, batch_idx):
        student_x, teacher_x, y = batch[0], batch[1], batch[2]
        with torch.no_grad():
            teacher_logits = self.teacher(teacher_x)
        student_logits = self(student_x)

        ce_loss = self.loss_fn(student_logits, y)
        T = self.temperature
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T ** 2)
        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        self.log('train_loss', loss)
        self.log('train_kd_loss', kd_loss)
        self.log('train_ce_loss', ce_loss)
        self.log('train_f1_macro', self.f1_macro(student_logits, y))
        self.log('train_bal_acc', self.balanced_accuracy(student_logits, y))
        return loss

    def validation_step(self, batch, batch_idx):
        student_x, teacher_x, y = batch[0], batch[1], batch[2]
        student_logits = self(student_x)
        loss = self.loss_fn(student_logits, y)
        self.log('val_loss', loss)
        self.log('val_f1_macro', self.f1_macro(student_logits, y))
        self.log('val_bal_acc', self.balanced_accuracy(student_logits, y))
        return loss
