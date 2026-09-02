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
        # pinned to match the ARC libribrain conda env exactly (checked via
        # pip freeze there) so Modal and ARC runs use identical library versions
        "torch==2.12.1",
        "torchvision==0.27.1",
        "torchaudio==2.11.0",
        "pytorch-lightning==2.6.5",
        "lightning==2.6.5",
        "wandb==0.27.2",
        "pnpl==0.1.1",
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
    with open(f"configs/phoneme/{config_name}/base-config-arc.yaml") as f:
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
        run_name = f"{config_name}-abo{alpha_tag}"
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
        track_test_per_epoch=False,
    )
    main(args)


@app.function(
    image=image,
    timeout=86400,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
)
def run_sequential(jobs: list, config_name: str = "student-50avg", baseline_only: bool = False):
    for run_index, alpha in jobs:
        run_distill.remote(run_index, alpha_override=alpha, config_name=config_name, baseline_only=baseline_only)


@app.function(
    image=image,
    gpu="L4",
    timeout=3600 * 14,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
    max_containers=9,
    retries=10,
    memory=65536,
)
def run_baseline_traintest(run_index: int, config_name: str, test_level: int = 50):
    """Train a plain baseline-Xavg (native averaging, hpo.py) config, then
    evaluate the resulting checkpoint at test_level via evaluate_averaging.py —
    the Modal equivalent of run_arc_traintest.sh."""
    import sys, os, yaml, glob, tempfile
    sys.path.insert(0, "/app")
    os.chdir("/app")

    with open(f"configs/phoneme/{config_name}/base-config-arc.yaml") as f:
        config = yaml.safe_load(f)

    for split in ["train", "val", "test"]:
        if split in config["data"]["datasets"]:
            for ds in config["data"]["datasets"][split]:
                for ds_cfg in ds.values():
                    ds_cfg["data_path"] = DATA_PATH
                    ds_cfg["preload_files"] = True

    config["general"]["output_path"] = f"{RESULTS_PATH}/{config_name}"
    config["general"]["checkpoint_path"] = f"{CHECKPOINTS_PATH}/{config_name}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        tmp_config_path = f.name

    from libribrain_experiments.hpo import main as hpo_main
    hpo_args = Namespace(
        config=tmp_config_path,
        search_space=f"configs/phoneme/{config_name}/search-space.yaml",
        run_name=config_name,
        run_index=run_index,
        project_name=WANDB_PROJECT,
        track_test_per_epoch=False,
    )
    hpo_main(hpo_args)

    run_name = f"{config_name}-hpo-{run_index}"
    ckpt_dir = f"{CHECKPOINTS_PATH}/{config_name}/{run_name}"
    ckpts = glob.glob(f"{ckpt_dir}/best-*.ckpt")
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    from libribrain_experiments.evaluate_averaging import main as eval_main
    eval_args = Namespace(
        config=tmp_config_path,
        checkpoint=ckpts[0],
        module="classification",
        split="test",
        levels=str(test_level),
        n_pool=100,
        batch_size=8,
        output=f"{RESULTS_PATH}/{config_name}/{run_name}-test{test_level}.json",
    )
    eval_main(eval_args)


@app.local_entrypoint()
def main():
    # baseline CE (step-matched, 63 steps/epoch) for student-50avg, seeds 0-9, temp=2.0
    run_sequential.spawn(
        [(i, None) for i in [1, 6, 11, 16, 21, 26, 31, 36, 41, 46]],
        config_name="student-50avg",
        baseline_only=True,
    )


@app.local_entrypoint()
def traintest50():
    # train baseline-{5,15,40}avg (seeds 0-2), evaluate test-time at level=50 — 9 jobs, parallel across GPUs
    # Each job spawned independently (not .starmap()) so one job's failure/preemption
    # can never cascade into cancelling its siblings via a shared blocking coordinator.
    jobs = [(seed, f"baseline-{level}avg") for level in [5, 15, 40] for seed in [0, 1, 2]]
    for seed, config_name in jobs:
        run_baseline_traintest.spawn(seed, config_name)


@app.local_entrypoint()
def traintest50seed1():
    # train baseline-{5,15,40,60}avg, seed 1 only, evaluate test-time at level=50 — 4 jobs
    jobs = [(1, f"baseline-{level}avg") for level in [5, 15, 40, 60]]
    for seed, config_name in jobs:
        run_baseline_traintest.spawn(seed, config_name)


@app.local_entrypoint()
def traintest50_20_30_85():
    # train baseline-{20,30,85}avg, seeds 0-1, evaluate test-time at level=50 — 6 jobs
    jobs = [(seed, f"baseline-{level}avg") for level in [20, 30, 85] for seed in [0, 1]]
    for seed, config_name in jobs:
        run_baseline_traintest.spawn(seed, config_name)


@app.local_entrypoint()
def traintest50_10():
    # train baseline-10avg, seeds 0-1, evaluate test-time at level=50 — 2 jobs
    for seed in [0, 1]:
        run_baseline_traintest.spawn(seed, "baseline-10avg")


@app.local_entrypoint()
def traintest50_2030_extra():
    # extra seeds for baseline-20avg (seeds 2-3) and baseline-30avg (seed 2), test on 50 — 3 jobs
    # prioritized to strengthen the "peak at 25" claim (20/30 flank the apparent peak)
    jobs = [(2, "baseline-20avg"), (3, "baseline-20avg"), (2, "baseline-30avg")]
    for seed, config_name in jobs:
        run_baseline_traintest.spawn(seed, config_name)


@app.local_entrypoint()
def traintest50_85_extra():
    # extra seed for baseline-85avg (seed 2), test on 50 — 1 job
    run_baseline_traintest.spawn(2, "baseline-85avg")


@app.local_entrypoint()
def traintest50_10_extra():
    # extra seed for baseline-10avg (seed 2), test on 50 — 1 job
    run_baseline_traintest.spawn(2, "baseline-10avg")


@app.local_entrypoint()
def traintest50_30_85_balance():
    # one more seed each for baseline-30avg and baseline-85avg (seed 3), test on 50 — 2 jobs
    # brings both up to n=4, matching baseline-20avg
    jobs = [(3, "baseline-30avg"), (3, "baseline-85avg")]
    for seed, config_name in jobs:
        run_baseline_traintest.spawn(seed, config_name)


@app.local_entrypoint()
def baseline50seeds6to9():
    # step-matched CE baseline (63 steps/epoch, --baseline-only) for student-50avg, seeds 6-9, temp=2.0 — 4 jobs, parallel
    jobs = [(run_index, True, None, "student-50avg") for run_index in [31, 36, 41, 46]]
    for run_index, baseline_only, alpha_override, config_name in jobs:
        run_distill.spawn(run_index, baseline_only=baseline_only, alpha_override=alpha_override, config_name=config_name)


@app.local_entrypoint()
def baseline50seeds3to5():
    # step-matched CE baseline (63 steps/epoch, --baseline-only) for student-50avg, seeds 3-5, temp=2.0 — 3 jobs, parallel
    jobs = [(run_index, True, None, "student-50avg") for run_index in [16, 21, 26]]
    for run_index, baseline_only, alpha_override, config_name in jobs:
        run_distill.spawn(run_index, baseline_only=baseline_only, alpha_override=alpha_override, config_name=config_name)


@app.function(
    image=image,
    gpu="L4",
    timeout=1800,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
)
def run_eval_timing(config_name: str, run_index: int, test_level: int, log_wandb: bool = False):
    """Times a single evaluate_averaging.py call against an already-trained
    checkpoint — the same code path as ARC's run_arc_matrix.sh array job —
    to check whether its 15-minute time budget is realistic."""
    import sys, os, glob, json, time, tempfile, yaml
    sys.path.insert(0, "/app")
    os.chdir("/app")

    with open(f"configs/phoneme/{config_name}/base-config-arc.yaml") as f:
        config = yaml.safe_load(f)
    for split in ["train", "val", "test"]:
        if split in config["data"]["datasets"]:
            for ds in config["data"]["datasets"][split]:
                for ds_cfg in ds.values():
                    ds_cfg["data_path"] = DATA_PATH
                    ds_cfg["preload_files"] = True
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        tmp_config_path = f.name

    run_name = f"{config_name}-hpo-{run_index}"
    ckpt_dir = f"{CHECKPOINTS_PATH}/{config_name}/{run_name}"
    ckpts = glob.glob(f"{ckpt_dir}/best-*.ckpt")
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    out_path = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False).name

    from libribrain_experiments.evaluate_averaging import main as eval_main
    t0 = time.time()
    eval_args = Namespace(
        config=tmp_config_path,
        checkpoint=ckpts[0],
        module="classification",
        split="test",
        levels=str(test_level),
        n_pool=200,
        batch_size=8,
        output=out_path,
    )
    eval_main(eval_args)
    elapsed = time.time() - t0
    with open(out_path) as f:
        results = json.load(f)
    print(f"EVAL TIMING: {elapsed:.1f}s ({elapsed / 60:.2f} min) for {run_name} @ test level {test_level}")
    print(f"RESULTS: {results}")

    if log_wandb:
        import wandb
        wandb.init(project=WANDB_PROJECT, name=f"{run_name}-test{test_level}-timing")
        wandb.log({
            "test_level": test_level,
            "elapsed_seconds": elapsed,
            "f1_macro": results[str(test_level)]["f1_macro"],
            "bal_acc": results[str(test_level)]["bal_acc"],
        })
        wandb.finish()


@app.local_entrypoint()
def timing_test():
    # single timing probe: baseline-85avg-hpo-0 (already trained), evaluated at test level 50
    run_eval_timing.remote("baseline-85avg", 0, 50, log_wandb=False)


@app.local_entrypoint()
def student95_seed5_triplet():
    # student-95avg, seed 5 (first seed not covered by ARC's seeds-0-4 batch),
    # across all 3 categories — 3 jobs, parallel (default L4; this Modal
    # client version doesn't support with_options() for a per-call GPU override)
    run_distill.spawn(5, baseline_only=True, alpha_override=None, config_name="student-95avg")
    run_distill.spawn(5, baseline_only=False, alpha_override=0.5, config_name="student-95avg")
    run_distill.spawn(26, baseline_only=False, alpha_override=0.6, config_name="student-95avg-scheduled")


@app.local_entrypoint()
def batch_80_90_triplets_and_20avg_extra():
    # student-80avg + student-90avg triplets (baseline/KD/scheduled-FiLM) at
    # seed 5, extending the seed-5-across-all-levels comparison — 6 jobs.
    # Plus one more seed (4) for baseline-20avg's train-X-test-50 sweep: its
    # current n=3 mean (0.4691) narrowly beats 25avg (0.4635), which would
    # flip the "peak at 25" story to "peak at 20" — worth confirming. 7 jobs total.
    for level in [80, 90]:
        run_distill.spawn(5, baseline_only=True, alpha_override=None, config_name=f"student-{level}avg")
        run_distill.spawn(5, baseline_only=False, alpha_override=0.5, config_name=f"student-{level}avg")
        run_distill.spawn(26, baseline_only=False, alpha_override=0.6, config_name=f"student-{level}avg-scheduled")
    run_baseline_traintest.spawn(4, "baseline-20avg")


@app.local_entrypoint()
def student85_seed5_triplet():
    # student-85avg, seed 5 — completes the seed-5-across-all-levels triplet
    # set (80/85/90/95), which previously had every level except this one — 3 jobs
    run_distill.spawn(5, baseline_only=True, alpha_override=None, config_name="student-85avg")
    run_distill.spawn(5, baseline_only=False, alpha_override=0.5, config_name="student-85avg")
    run_distill.spawn(26, baseline_only=False, alpha_override=0.6, config_name="student-85avg-scheduled")


@app.local_entrypoint()
def scheduled_seed6_85_90_95():
    # scheduled-KD-FiLM only, seed 6, at 85/90/95avg — 3 jobs. ARC's seeds-0-4
    # batch appears to be processing levels in submission order (80avg first),
    # so 85/90/95 are likely still queued; this builds seed depth ahead of
    # that on the category most worth deepening (scheduled-a05 was the more
    # robust of the two significant 75avg KD hits, not outlier-driven).
    for level in [85, 90, 95]:
        run_distill.spawn(31, baseline_only=False, alpha_override=0.6, config_name=f"student-{level}avg-scheduled")


@app.local_entrypoint()
def timing_test_wandb():
    # same probe, but also logs the result + elapsed time to wandb
    run_eval_timing.remote("baseline-85avg", 0, 50, log_wandb=True)


@app.local_entrypoint()
def thin_levels_extra_seeds():
    # reinforce the thinnest rows in the full transfer matrix (20/30/85/200avg,
    # all currently n=3-4) — 5 jobs. 20avg gets 2 (it's the thinnest, n=3
    # after excluding the ambiguous hpo-2 checkpoint); 30/85/200avg get 1 each.
    # Trains + evaluates at test=50 only for now; sync new checkpoints to ARC
    # afterward and rerun the matrix sweep there to fill in all 16 test levels.
    run_baseline_traintest.spawn(4, "baseline-20avg")
    run_baseline_traintest.spawn(5, "baseline-20avg")
    run_baseline_traintest.spawn(4, "baseline-30avg")
    run_baseline_traintest.spawn(4, "baseline-85avg")
    run_baseline_traintest.spawn(4, "baseline-200avg")


@app.local_entrypoint()
def seed7_95_triplet_90_baseline_kd():
    # 95avg gets a second full same-seed triplet (seed 7) -- still the most
    # theoretically decisive level (closest to the teacher's 100avg) and
    # currently only has n=1 per category. 90avg gets baseline+KD only
    # (its scheduled category already got the seed-6 priority boost). 5 jobs.
    run_distill.spawn(7, baseline_only=True, alpha_override=None, config_name="student-95avg")
    run_distill.spawn(7, baseline_only=False, alpha_override=0.5, config_name="student-95avg")
    run_distill.spawn(36, baseline_only=False, alpha_override=0.6, config_name="student-95avg-scheduled")
    run_distill.spawn(7, baseline_only=True, alpha_override=None, config_name="student-90avg")
    run_distill.spawn(7, baseline_only=False, alpha_override=0.5, config_name="student-90avg")
