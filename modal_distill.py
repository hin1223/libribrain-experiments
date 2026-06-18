import modal
from argparse import Namespace

WANDB_PROJECT = "libribrain-experiments"
DATA_PATH = "/vol/libribrain-data"
RESULTS_PATH = "/vol/results"
CHECKPOINTS_PATH = "/vol/checkpoints"

app = modal.App("libribrain-distill")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install([
        "torch==2.11.0",
        "torchvision==0.26.0",
        "torchaudio==2.11.0",
        "pytorch-lightning==2.6.1",
        "lightning",
        "wandb",
        "pnpl",
        "numpy",
        "scikit-learn",
        "h5py",
        "mne",
        "mne-bids",
        "matplotlib",
        "pyyaml",
    ])
    .add_local_dir(".", remote_path="/app", copy=True, ignore=["wandb/**", "__pycache__/**", "*.egg-info/**"])
    .run_commands("pip install -e /app -q")
)

volume = modal.Volume.from_name("libribrain-vol", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=3600 * 14,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
    max_containers=10,
    retries=10,
    memory=65536,
)
def run_distill(run_index: int, baseline_only: bool = False, alpha_override: float = None,
                config_name: str = "student-50avg"):
    import sys, os, yaml
    sys.path.insert(0, "/app")
    os.chdir("/app")

    # Load and patch config in memory — no file writes, safe for parallel runs
    with open(f"configs/phoneme/{config_name}/base-config.yaml") as f:
        config = yaml.safe_load(f)

    for split in ["train", "val", "test"]:
        if split in config["data"]["datasets"]:
            for ds in config["data"]["datasets"][split]:
                for ds_cfg in ds.values():
                    ds_cfg["data_path"] = DATA_PATH
                    ds_cfg["preload_files"] = True

    config["general"]["output_path"] = f"{RESULTS_PATH}/{config_name}"
    config["general"]["checkpoint_path"] = f"{CHECKPOINTS_PATH}/{config_name}"
    config["distillation"]["teacher_checkpoint_path"] = "/vol/teacher-checkpoint.ckpt"

    if alpha_override is not None:
        config["distillation"]["alpha"] = alpha_override

    # Write to a per-run temp config to avoid race conditions
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        tmp_config_path = f.name

    alpha_tag = f"-a{int(alpha_override * 10):02d}" if alpha_override is not None else ""
    if baseline_only:
        run_name = f"baseline-50avg{alpha_tag}"
    else:
        run_name = f"{config_name}{alpha_tag}"

    from libribrain_experiments.distill import main
    args = Namespace(
        config=tmp_config_path,
        search_space=f"configs/phoneme/{config_name}/search-space.yaml",
        run_name=run_name,
        run_index=run_index,
        project_name=WANDB_PROJECT,
        baseline_only=baseline_only,
        alpha_override=alpha_override,
        temperature_override=None,
    )
    main(args)


@app.function(
    image=image,
    timeout=86400,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
)
def run_sequential(jobs: list, config_name: str = "student-50avg"):
    for run_index, alpha in jobs:
        run_distill.remote(run_index, alpha_override=alpha, config_name=config_name)


@app.local_entrypoint()
def main():
    run_sequential.spawn(
        [(i, None) for i in range(5, 15)],
        config_name="student-50avg-stochastic",
    )
