"""Evaluate a trained checkpoint (fixed averaging level, e.g. 50avg) at other
test-time averaging levels.

AverageGroups has no learned parameters, so the same trained conv layers can
be evaluated at any averaging level: pool `n_pool` raw trials per group,
manually average the first `level` of them, and feed that directly past the
checkpoint's AverageGroups layer (patched to n_groups=1, i.e. identity).
"""
from argparse import ArgumentParser
import json
import os
import yaml
import torch
from pnpl.datasets.grouped_dataset import GroupedDataset

from libribrain_experiments.utils import get_dataset_partition_from_config
from libribrain_experiments.models.average_groups import AverageGroups
from libribrain_experiments.models.configurable_modules.classification_module import ClassificationModule
from libribrain_experiments.models.configurable_modules.distillation_module import DistillationModule

MODULES = {
    "classification": ClassificationModule,
    "distillation": DistillationModule,
}


def evaluate_at_level(model, loader, level, channels_per_sample, device):
    model.f1_macro.reset()
    model.balanced_accuracy.reset()
    with torch.no_grad():
        for raw_x, y in loader:
            raw_x = raw_x.to(device)
            y = y.to(device)
            n_pool = raw_x.size(1) // channels_per_sample
            averaged_x = raw_x.view(
                raw_x.size(0), n_pool, channels_per_sample, raw_x.size(2)
            )[:, :level].mean(dim=1)
            logits = model(averaged_x)
            model.f1_macro.update(logits, y)
            model.balanced_accuracy.update(logits, y)
    return {
        "f1_macro": model.f1_macro.compute().item(),
        "bal_acc": model.balanced_accuracy.compute().item(),
    }


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    datasets_config = config["data"]["datasets"]
    train_raw = get_dataset_partition_from_config(datasets_config["train"])
    train_means = train_raw.datasets[0].channel_means
    train_stds = train_raw.datasets[0].channel_stds

    split_raw = get_dataset_partition_from_config(
        datasets_config[args.split], train_means, train_stds)
    channels_per_sample = split_raw[0][0].shape[0]

    pooled = GroupedDataset(
        split_raw, grouped_samples=args.n_pool,
        average_grouped_samples=False, drop_remaining=True)
    loader = torch.utils.data.DataLoader(pooled, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")

    ModuleClass = MODULES[args.module]
    model = ModuleClass.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model = model.to(device)

    if not isinstance(model.modules_list[0], AverageGroups):
        raise ValueError(
            "Expected the checkpoint's first model layer to be AverageGroups")
    model.modules_list[0].n_groups = 1  # averaging done manually per level above

    levels = [int(l) for l in args.levels.split(",")]
    results = {}
    for level in levels:
        if level > args.n_pool:
            raise ValueError(
                f"Level {level} exceeds --n-pool {args.n_pool}; increase --n-pool")
        results[level] = evaluate_at_level(
            model, loader, level, channels_per_sample, device)
        print(f"level={level}: {results[level]}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Base config yaml matching the checkpoint's training run (for data paths/standardization)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--module", choices=list(MODULES.keys()), required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--levels", default="10,20,30,40,50,75,100")
    parser.add_argument("--n-pool", type=int, default=100,
                        help="Raw trials pooled per group; must be >= max level")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    main(args)
