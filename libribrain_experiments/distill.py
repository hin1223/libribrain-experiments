from argparse import ArgumentParser
import itertools
import yaml
import wandb
import pytorch_lightning as lightning
import numpy as np
import torch
import time
import os

from libribrain_experiments.hpo import runs_configs_from_search_space, load_search_space, update_config_for_single_run
from libribrain_experiments.utils import (
    get_dataset_partition_from_config, adapt_config_to_data,
    run_validation, log_results, get_label_counts
)
from libribrain_experiments.paired_dataset import PairedGroupedDataset, StudentOnlyDataset
from libribrain_experiments.models.configurable_modules.distillation_module import DistillationModule
from libribrain_experiments.models.configurable_modules.classification_module import ClassificationModule
from libribrain_experiments.utils import run_training
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.accelerators import find_usable_cuda_devices


def get_paired_datasets_from_config(data_config):
    datasets_config = data_config["datasets"]
    n_teacher = data_config["general"]["n_teacher"]
    n_student = data_config["general"]["n_student"]

    train_raw = get_dataset_partition_from_config(datasets_config["train"])
    train_channel_means = train_raw.datasets[0].channel_means
    train_channel_stds = train_raw.datasets[0].channel_stds
    train_labels = train_raw.datasets[0].labels_sorted

    train_dataset = PairedGroupedDataset(train_raw, n_teacher=n_teacher, n_student=n_student)

    val_raw = get_dataset_partition_from_config(
        datasets_config["val"], train_channel_means, train_channel_stds)
    val_dataset = PairedGroupedDataset(val_raw, n_teacher=n_teacher, n_student=n_student)

    test_dataset = None
    if "test" in datasets_config:
        test_raw = get_dataset_partition_from_config(
            datasets_config["test"], train_channel_means, train_channel_stds)
        test_dataset = PairedGroupedDataset(test_raw, n_teacher=n_teacher, n_student=n_student)

    return train_dataset, val_dataset, test_dataset, train_labels


def run_distillation(train_loader, val_loader, config):
    distill_config = config["distillation"]
    module = DistillationModule(
        model_config=config["model"],
        n_classes=config["_n_classes"],
        optimizer_config=config["optimizer"],
        loss_config=config["loss"],
        teacher_checkpoint_path=distill_config["teacher_checkpoint_path"],
        temperature=distill_config.get("temperature", 2.0),
        alpha=distill_config.get("alpha", 0.5),
    )

    logger = False
    if config["general"]["wandb"]:
        logger = WandbLogger()

    os.makedirs(config["general"]["checkpoint_path"], exist_ok=True)
    best_metric = config["general"].get("best_model_metrics", "val_loss")
    mode = "min" if best_metric == "val_loss" else "max"
    checkpoint_cb = ModelCheckpoint(
        dirpath=config["general"]["checkpoint_path"],
        monitor=best_metric,
        mode=mode,
        save_top_k=1,
        save_last=True,
        filename="best-" + best_metric + "-" + config["general"]["run_name"] + "-{epoch:02d}-{val_f1_macro:.4f}",
    )

    trainer = Trainer(
        logger=logger,
        accelerator="auto",
        log_every_n_steps=1,
        callbacks=[checkpoint_cb],
        **config["trainer"],
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_module = DistillationModule.load_from_checkpoint(checkpoint_cb.best_model_path)
    return trainer, best_module, module


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    search_space = load_search_space(args.search_space)
    run_configs = runs_configs_from_search_space(search_space)
    if args.run_index is None:
        args.run_index = np.random.randint(0, len(run_configs))
    config = update_config_for_single_run(config, run_configs[args.run_index])
    print("Running config index:", args.run_index, run_configs[args.run_index])

    run_name = (args.run_name or "distill") + "-hpo-" + str(args.run_index)
    config["general"]["run_name"] = run_name

    if config["general"]["wandb"]:
        if args.project_name is None:
            raise ValueError("Please provide --project-name for wandb logging")
        wandb.init(project=args.project_name, name=run_name)
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_f1_macro", summary="max")
        wandb.define_metric("val_bal_acc", summary="max")

    lightning.seed_everything(config["general"]["seed"])

    paired_train, paired_val, paired_test, labels = get_paired_datasets_from_config(config["data"])

    # set n_groups="auto" → n_student in the model config
    for layer in config["model"]:
        layer_name = list(layer.keys())[0]
        layer_dict = layer[layer_name]
        if layer_dict and layer_dict.get("n_groups") == "auto":
            layer_dict["n_groups"] = config["data"]["general"]["n_student"]

    config["_n_classes"] = len(labels)

    if args.baseline_only:
        train_dataset = StudentOnlyDataset(paired_train)
        val_dataset = StudentOnlyDataset(paired_val)
        print("BASELINE MODE — TRAIN SIZE:", len(train_dataset), "  VAL SIZE:", len(val_dataset))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, **config["data"]["dataloader"])
        val_loader = torch.utils.data.DataLoader(
            val_dataset, **config["data"]["dataloader"])
        adapt_config_to_data(config, train_loader, labels)
        _, best_module, module = run_training(
            train_loader, val_loader, config, len(labels))
    else:
        train_dataset = paired_train
        val_dataset = paired_val
        print("DISTILLATION MODE — TRAIN SIZE:", len(train_dataset), "  VAL SIZE:", len(val_dataset))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, **config["data"]["dataloader"])
        val_loader = torch.utils.data.DataLoader(
            val_dataset, **config["data"]["dataloader"])
        _, best_module, module = run_distillation(train_loader, val_loader, config)

    del module

    if torch.cuda.is_available():
        device = find_usable_cuda_devices()[0]
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    best_module = best_module.to(device)

    samples_per_class = get_label_counts(train_loader, len(labels))

    def student_loader(loader):
        if args.baseline_only:
            yield from loader
        else:
            for student_x, _, y in loader:
                yield [student_x, y]

    result, y, preds, logits = run_validation(
        student_loader(val_loader), best_module, labels, samples_per_class=samples_per_class)
    log_results(result, y, preds, logits, config["general"]["output_path"], "val-best-" + run_name)

    if paired_test is not None:
        test_dataset = StudentOnlyDataset(paired_test) if args.baseline_only else paired_test
        test_loader = torch.utils.data.DataLoader(
            test_dataset, **config["data"]["dataloader"])
        result, y, preds, logits = run_validation(
            student_loader(test_loader), best_module, labels, samples_per_class=samples_per_class)
        log_results(result, y, preds, logits, config["general"]["output_path"], "test-best-" + run_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--search-space", type=str, required=True)
    parser.add_argument("--run-index", type=int)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--project-name", type=str, default="libribrain-experiments")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Train baseline (CE only) on the same data as the distillation student")
    args = parser.parse_args()
    main(args)
