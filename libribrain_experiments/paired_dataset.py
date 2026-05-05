import torch
from torch.utils.data import Dataset
from pnpl.datasets.grouped_dataset import GroupedDataset


class StudentOnlyDataset(Dataset):
    """Wraps PairedGroupedDataset and returns (student_x, label) for baseline training."""

    def __init__(self, paired_dataset):
        self.paired = paired_dataset

    def __len__(self):
        return len(self.paired)

    def __getitem__(self, idx):
        student_x, _, label = self.paired[idx]
        return [student_x, label]


class PairedGroupedDataset(Dataset):
    """Returns (student_x, teacher_x, label) aligned pairs.

    Both student and teacher draw from the same n_teacher-sized group of same-label
    samples. teacher_x contains all n_teacher samples concatenated (shape: n_teacher*C, T).
    student_x is the first n_student samples from that group (shape: n_student*C, T).
    """

    def __init__(self, original_dataset, n_teacher=100, n_student=50):
        assert n_teacher % n_student == 0, "n_teacher must be divisible by n_student"
        self.n_teacher = n_teacher
        self.n_student = n_student
        self.teacher_grouped = GroupedDataset(
            original_dataset,
            grouped_samples=n_teacher,
            average_grouped_samples=False,
            drop_remaining=True,
        )
        self.channels_per_sample = original_dataset[0][0].shape[0]

    def __len__(self):
        return len(self.teacher_grouped)

    def __getitem__(self, idx):
        teacher_x, label = self.teacher_grouped[idx]
        student_x = teacher_x[:self.n_student * self.channels_per_sample]
        return student_x, teacher_x, label
