import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import math
import re
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
import easydict
from dataloaders.legacy_online_degradation import LegacyOnlineDegradationDataset
from dataloaders.use_simulation import USESimulationSEMambaDataset
from model.stfts import mag_phase_stft, mag_phase_istft
from model.discriminator import MultiScaleSubbandCQTDiscriminator, MultiResolutionDiscriminator, \
    feature_loss, generator_loss, discriminator_loss
from model.loss import phase_losses, MultiScaleMelSpectrogramLoss
from utils import (
    load_config,
    load_ckpts,
    load_optimizer_states,
    prune_step_checkpoints,
    save_best_step_checkpoints,
    save_checkpoint,
    build_env,
    load_json_file,
    scan_checkpoint,
)
import random
import torchaudio
import torch.distributed as dist
from metrics.evaluation import load_modules, compute_val_metrics
from model.semambapp import SEMambapp
sys.path.append("/scratch/work/lil14/use_simulation_pipeline/scripts")
from validation_avqi_gap import run_validation_avqi_metrics


def log_validation_avqi_metrics(generator, hps, device, global_step):
    sampling_rate = hps["stft_cfg"]["sampling_rate"]
    generator.eval()

    @torch.inference_mode()
    def enhance_one(path):
        audio, source_rate = torchaudio.load(path)
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if source_rate != sampling_rate:
            audio = torchaudio.functional.resample(audio, source_rate, sampling_rate)
        audio = audio.squeeze(0).to(device)
        scale = 0.9 / audio.abs().max().clamp_min(1e-9)
        normalized = (audio * scale).unsqueeze(0)
        mag, pha, _ = mag_phase_stft(
            normalized,
            hps["stft_cfg"]["n_fft"],
            hps["stft_cfg"]["hop_size"],
            hps["stft_cfg"]["win_size"],
            hps["model_cfg"]["compress_factor"],
        )
        enhanced_mag, enhanced_pha, _ = generator(mag, pha)
        enhanced = mag_phase_istft(
            enhanced_mag,
            enhanced_pha,
            hps["stft_cfg"]["n_fft"],
            hps["stft_cfg"]["hop_size"],
            hps["stft_cfg"]["win_size"],
            hps["model_cfg"]["compress_factor"],
        )
        return (enhanced / scale).squeeze().cpu().numpy(), sampling_rate

    metrics = run_validation_avqi_metrics("semambapp", global_step, enhance_one)
    generator.train()
    return metrics


def save_best_avqi_gap_checkpoints(generator, discs, optims, exp_path, epoch, global_step, n_gpus, gap):
    generator_module = generator.module if n_gpus > 1 else generator
    best_gap = getattr(generator_module, "_best_avqi_gap_to_clean", float("inf"))
    if gap < 0 or gap >= best_gap:
        return

    mssbcqtd, mrd = discs
    optim_g, optim_d = optims
    generator_module._best_avqi_gap_to_clean = gap
    generator_state = {"generator": generator_module.state_dict()}
    optimizer_state = {
        "mssbcqtd": (mssbcqtd.module if n_gpus > 1 else mssbcqtd).state_dict(),
        "mrd": (mrd.module if n_gpus > 1 else mrd).state_dict(),
        "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(),
        "steps": global_step,
        "epoch": epoch,
        "best_avqi_gap_to_clean": gap,
    }
    save_best_step_checkpoints(
        exp_path,
        global_step,
        generator_state,
        optimizer_state,
        generator_prefix="avqi_gap_ln_g_",
        optimizer_prefix="avqi_gap_ln_do_",
    )


def save_best_guarded_checkpoints(generator, discs, optims, exp_path, epoch, global_step, n_gpus, val_loss):
    generator_module = generator.module if n_gpus > 1 else generator
    latest_gap = getattr(generator_module, "_latest_avqi_gap_to_clean", None)
    best_loss = getattr(generator_module, "_best_guarded_val_loss", float("inf"))
    if latest_gap is None or latest_gap < 0 or val_loss >= best_loss:
        return

    mssbcqtd, mrd = discs
    optim_g, optim_d = optims
    generator_module._best_guarded_val_loss = val_loss
    generator_state = {"generator": generator_module.state_dict()}
    optimizer_state = {
        "mssbcqtd": (mssbcqtd.module if n_gpus > 1 else mssbcqtd).state_dict(),
        "mrd": (mrd.module if n_gpus > 1 else mrd).state_dict(),
        "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(),
        "steps": global_step,
        "epoch": epoch,
        "best_avqi_gap_to_clean": getattr(generator_module, "_best_avqi_gap_to_clean", float("inf")),
        "latest_avqi_gap_to_clean": latest_gap,
        "best_guarded_val_loss": val_loss,
    }
    save_best_step_checkpoints(
        exp_path,
        global_step,
        generator_state,
        optimizer_state,
        generator_prefix="guarded_ln_g_",
        optimizer_prefix="guarded_ln_do_",
    )


os.environ['MASTER_ADDR'] = 'localhost'
torch.backends.cudnn.benchmark = True
steps = 0


WANDB_STANDARD_TAG = "wandb_standard_v1"


def scalar_value(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += parameter.grad.detach().norm(2).item() ** 2
    return math.sqrt(total)


def add_metric_window(metric_sums, metrics):
    for name, value in metrics.items():
        metric_sums[name] = metric_sums.get(name, 0.0) + scalar_value(value)


def average_metric_window(metric_sums, count):
    if count <= 0:
        return {}
    return {name: value / count for name, value in metric_sums.items()}


def name_token(value, default="na"):
    text = str(value if value not in (None, "") else default).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or default


def build_wandb_identity(repo_name, model_name, dataset_type, experiment, change, timestamp):
    run_name = "__".join(
        [
            name_token(timestamp),
            name_token(repo_name),
            name_token(model_name),
            name_token(dataset_type),
            name_token(change or experiment),
        ]
    )
    group = "__".join(
        [
            name_token(repo_name),
            name_token(model_name),
            name_token(dataset_type),
            name_token(experiment),
        ]
    )
    return run_name, group


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def setup_schedulers(optimizers, cfg, last_epoch):
    """Set up learning rate schedulers."""
    optim_g, optim_d = optimizers
    lr_decay = cfg['training_cfg']['lr_decay']

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay, last_epoch=last_epoch)

    return scheduler_g, scheduler_d

def load_ckpts_spec(args, device, prefix):
    """Load checkpoints if available."""
    if os.path.isdir(args.exp_path):
        cp_g = scan_checkpoint(args.exp_path, f'{prefix}g_')
        cp_do = scan_checkpoint(args.exp_path, f'{prefix}do_')
        if cp_g is None or cp_do is None:
            return None, None, 0, -1
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        return state_dict_g, state_dict_do, state_dict_do['steps'] + 1, state_dict_do['epoch']
    return None, None, 0, -1


def load_pretrained_if_requested(args, hps, device):
    """Load a base checkpoint for fine-tuning when no local resume checkpoint exists."""
    fine_tune_cfg = hps.get("fine_tuning_cfg", {})
    if not args.fine_tuning:
        return None, None

    generator_path = fine_tune_cfg.get("pretrained_generator_checkpoint", "")
    discriminator_path = fine_tune_cfg.get("pretrained_discriminator_checkpoint", "")

    if not generator_path:
        raise ValueError(
            "fine_tuning_cfg.enabled is true, but pretrained_generator_checkpoint is empty "
            "and no local resume checkpoint was found."
        )

    state_dict_g = load_checkpoint(generator_path, device) if generator_path else None
    state_dict_do = load_checkpoint(discriminator_path, device) if discriminator_path else None

    if state_dict_g is not None and args.fine_tuning:
        print(f"Starting fine-tuning from pretrained generator: {generator_path}")
    if state_dict_do is not None and args.fine_tuning:
        print(f"Starting fine-tuning from pretrained discriminators: {discriminator_path}")

    return state_dict_g, state_dict_do


def build_runtime_args(config_path, hps):
    data_cfg = hps.get("data_cfg", {})
    experiment_cfg = hps.get("experiment_cfg", {})

    a = easydict.EasyDict({
        "config": config_path,
        "dataset_type": data_cfg.get("dataset_type", "use_simulation_fixed"),
        "use_simulation_root": data_cfg.get("use_simulation_root", "/scratch/work/lil14/USE_simulation"),
        "train_pair_manifest": data_cfg.get("train_pair_manifest", ""),
        "valid_pair_manifest": data_cfg.get("valid_pair_manifest", ""),
        "clean_train_json": data_cfg.get("clean_train_json", ""),
        "noise_train_json": data_cfg.get("noise_train_json", ""),
        "rir_train_json": data_cfg.get("rir_train_json", ""),
        "clean_valid_json": data_cfg.get("clean_valid_json", ""),
        "degraded_valid_json": data_cfg.get("degraded_valid_json", ""),
        "stdout_interval": experiment_cfg.get("stdout_interval", 1250),
        "checkpoint_interval": experiment_cfg.get("checkpoint_interval_steps", experiment_cfg.get("checkpoint_interval", 5000)),
        "summary_interval": experiment_cfg.get("wandb_log_interval_steps", experiment_cfg.get("summary_interval", 1250)),
        "validation_interval": experiment_cfg.get("validation_interval_steps", experiment_cfg.get("validation_interval", 5000)),
        "max_steps": int(experiment_cfg.get("max_steps", 0) or 0),
        "exp_path": experiment_cfg.get("exp_path", "exp"),
        "fine_tuning": bool(hps.get("fine_tuning_cfg", {}).get("enabled", False)),
        "experiment_name": experiment_cfg.get("experiment_name", "train_semambapp"),
        "use_wandb": bool(experiment_cfg.get("use_wandb", True)),
        "wandb_project": experiment_cfg.get("wandb_project", "semambapp"),
        "wandb_entity": experiment_cfg.get("wandb_entity", None),
        "wandb_mode": experiment_cfg.get("wandb_mode", "online"),
        "wandb_tags": list(dict.fromkeys(experiment_cfg.get("wandb_tags", []) + [WANDB_STANDARD_TAG])),
    })
    a.exp_path = os.path.join(a.exp_path, a.experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = hps.get("model_name", "semambapp")
    experiment = hps.get("experiment", a.experiment_name)
    change = experiment_cfg.get("wandb_change") or hps.get("wandb_change") or hps.get("change") or experiment
    a.wandb_run_name, a.wandb_group = build_wandb_identity(
        hps.get("repo_name", "semambapp"),
        model_name,
        a.dataset_type,
        experiment,
        change,
        timestamp,
    )
    a.model_name = model_name
    a.wandb_change = change
    return a


def build_dataset(hps, a, mode, device):
    if a.dataset_type == "use_simulation_fixed":
        manifest = a.train_pair_manifest if mode == "Train" else a.valid_pair_manifest
        if not manifest:
            raise ValueError(f"{mode} pair manifest is required for use_simulation_fixed dataset.")
        return USESimulationSEMambaDataset(
            hps,
            pair_manifest=manifest,
            use_simulation_root=a.use_simulation_root,
            mode=mode,
            random_start=(mode == "Train"),
            normalize=True,
            seed=hps["env_setting"]["seed"],
        )

    if a.dataset_type == "legacy_online_degradation":
        required = {
            "clean_train_json": a.clean_train_json,
            "noise_train_json": a.noise_train_json,
            "rir_train_json": a.rir_train_json,
            "clean_valid_json": a.clean_valid_json,
            "degraded_valid_json": a.degraded_valid_json,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                "Legacy online degradation requires data_cfg fields: "
                + ", ".join(missing)
            )
        return LegacyOnlineDegradationDataset(
            hps,
            clean_json=a.clean_train_json,
            noise_json=a.noise_train_json,
            rir_json=a.rir_train_json,
            clean_valid_json=a.clean_valid_json,
            degraded_valid_json=a.degraded_valid_json,
            use_simulation_root=a.use_simulation_root,
            mode=mode,
            seed=hps["env_setting"]["seed"],
        )

    raise ValueError(
        f"Unsupported dataset_type={a.dataset_type!r}. "
        "Expected use_simulation_fixed or legacy_online_degradation."
    )


def run(rank, n_gpus, a, hps):


    global steps

    # Initialize distributed training if using multiple GPUs
    if n_gpus > 1:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    
    torch.manual_seed(hps["env_setting"]["seed"])
    torch.cuda.set_device(rank)
    device = torch.device('cuda:{:d}'.format(rank))

    # Collecting filelists for training and validation

    # Training dataset configuration
    trainset = build_dataset(hps, a, "Train", device)
    
    train_sampler = DistributedSampler(trainset, rank = rank) if n_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=hps["env_setting"]["num_workers"], shuffle=False,
                              sampler=train_sampler,
                              batch_size=hps["training_cfg"]["batch_size"], pin_memory=True, drop_last=True)

    # Validation dataset configuration
    validation_loader = None
    if rank == 0:
        validset = build_dataset(hps, a, "Validation", device)
        if len(validset) > 0:
            validation_loader = DataLoader(
                validset,
                num_workers=1,
                shuffle=False,
                sampler=None,
                batch_size=1,
                pin_memory=True,
                drop_last=True,
            )
        else:
            print("Validation disabled: no paired validation files were provided.")

        # Initialize Weights & Biases logging
        if a.use_wandb:
            os.environ.setdefault("WANDB_DIR", os.path.join(os.getcwd(), "runs", "wandb"))
            os.environ.setdefault("WANDB_CACHE_DIR", os.path.join(os.getcwd(), "runs", "wandb-cache"))
            os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
            os.makedirs(os.environ["WANDB_CACHE_DIR"], exist_ok=True)
            wandb.init(
                project=a.wandb_project,
                entity=a.wandb_entity,
                name=a.wandb_run_name,
                group=a.wandb_group,
                resume="allow",
                mode=a.wandb_mode,
                tags=a.wandb_tags,
                config={
                    **hps,
                    "model_name": a.model_name,
                    "wandb_change": a.wandb_change,
                    "wandb_group": a.wandb_group,
                },
            )
            wandb.define_metric("charts/global_step", overwrite=True)
            wandb.define_metric("*", step_metric="charts/global_step", step_sync=True, overwrite=True)
            wandb.define_metric("trainer/global_step", hidden=True, overwrite=True)
            wandb.define_metric("charts/epoch", step_metric="charts/global_step", overwrite=True)
            wandb.define_metric("train/*", step_metric="charts/global_step", overwrite=True)
            wandb.define_metric("val/*", step_metric="charts/global_step", overwrite=True)
            wandb.define_metric("val_avqi/*", step_metric="charts/global_step", overwrite=True)
            wandb.define_metric("charts/*", step_metric="charts/global_step", overwrite=True)

    # Initializing modules
    univsemamba = SEMambapp(hps).to(device)
    mssbcqtd = MultiScaleSubbandCQTDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)

    fn_mel_loss_multiscale = MultiScaleMelSpectrogramLoss(
                sampling_rate=hps["stft_cfg"]["sampling_rate"]
            ).to(device) 
    if validation_loader is not None or rank != 0:
        mrstft = load_modules(hps, device)
    else:
        mrstft = None
    # Print model parameter counts
    if rank == 0:
        print('Number of Parameters for SEMambapp:', get_param_num(univsemamba))
        print("Number of Parameters for MSSBCQTD:  ", get_param_num(mssbcqtd))
        print("Number of Parameters for MRD:  ", get_param_num(mrd))





    state_dict_g, state_dict_do, steps, last_epoch = load_ckpts_spec(a, device, prefix='ln_')
    resume_from_local_checkpoint = state_dict_g is not None
    if state_dict_g is None:
        state_dict_g, state_dict_do = load_pretrained_if_requested(a, hps, device)
    if state_dict_g is not None:
        univsemamba.load_state_dict(state_dict_g.get('generator', state_dict_g), strict=False)
        if state_dict_do is not None and 'mssbcqtd' in state_dict_do and 'mrd' in state_dict_do:
            mssbcqtd.load_state_dict(state_dict_do['mssbcqtd'], strict=False)
            mrd.load_state_dict(state_dict_do['mrd'], strict=False)
    univsemamba._best_avqi_gap_to_clean = (
        float(state_dict_do.get("best_avqi_gap_to_clean", float("inf")))
        if state_dict_do is not None
        else float("inf")
    )
    univsemamba._latest_avqi_gap_to_clean = (
        float(state_dict_do["latest_avqi_gap_to_clean"])
        if state_dict_do is not None and "latest_avqi_gap_to_clean" in state_dict_do
        else None
    )
    univsemamba._best_guarded_val_loss = (
        float(state_dict_do.get("best_guarded_val_loss", float("inf")))
        if state_dict_do is not None
        else float("inf")
    )

    optim_g = torch.optim.AdamW(univsemamba.parameters(), hps["training_cfg"]["learning_rate"], betas=[hps["training_cfg"]["adam_b1"], hps["training_cfg"]["adam_b2"]])
    optim_d = torch.optim.AdamW(itertools.chain(mrd.parameters(), mssbcqtd.parameters()),
                                hps["training_cfg"]["learning_rate"], betas=[hps["training_cfg"]["adam_b1"], hps["training_cfg"]["adam_b2"]])

    # Load optimizer states
    if resume_from_local_checkpoint and state_dict_do is not None:
        print("Loading Optimizer States...")
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g, scheduler_d = setup_schedulers([optim_g, optim_d], hps, last_epoch)



    # Setup distributed data parallel if using multiple GPUs
    if n_gpus > 1:
        univsemamba = DDP(univsemamba, device_ids=[rank])
        mssbcqtd = DDP(mssbcqtd, device_ids=[rank]).to(device)
        mrd = DDP(mrd, device_ids=[rank]).to(device)

    # Set models to training mode
    univsemamba.train()
    mssbcqtd.train()
    mrd.train()

    # Main training loop
    total_epochs = hps["training_cfg"]["training_epochs"]
    for epoch in range(max(0, last_epoch), total_epochs):
        start = time.time()
        
        if rank == 0:
            print("Epoch: {:d}".format(epoch))
            print('Learning Rate : {:.6f}'.format(optim_g.param_groups[0]['lr']))
            train(a, rank, epoch, hps, univsemamba, [mssbcqtd, mrd], [fn_mel_loss_multiscale, mrstft], [optim_g, optim_d],
                     [scheduler_g, scheduler_d], [train_loader, validation_loader], n_gpus, device)
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))
        else:
            train(a, rank, epoch, hps, univsemamba, [mssbcqtd, mrd], [fn_mel_loss_multiscale, mrstft], [optim_g, optim_d],
                     [scheduler_g, scheduler_d], [train_loader, None], n_gpus, device)

def train(a, rank, epoch, hps, nets, discs, aux, optims, schedulers, loaders, n_gpus, device=None):

    generator = nets
    mssbcqtd, mrd = discs
    fn_mel_loss_multiscale, mrstft = aux
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, validation_loader = loaders

    global steps

    # Set epoch for distributed sampler
    if n_gpus > 1:
        train_loader.sampler.set_epoch(epoch)

    # Set models to training mode
    generator.train()
    mssbcqtd.train()
    mrd.train()
    train_metric_sums = {}
    train_metric_count = 0
    train_window_elapsed_sec = 0.0
    samples_seen = 0
    train_window_samples = 0
    optimizer_metric_sums = {}
    optimizer_metric_count = 0
    summary_interval = max(1, int(a.summary_interval))
    validation_interval = max(1, int(a.validation_interval))
    avqi_interval = int(hps["experiment_cfg"].get("avqi_validation_interval_steps", 1000) or 0)
    gradient_accumulation_steps = max(1, int(hps["training_cfg"].get("gradient_accumulation_steps", 1)))
    discriminator_parameters = list(itertools.chain(mssbcqtd.parameters(), mrd.parameters()))
    optim_d.zero_grad()
    optim_g.zero_grad()

    # Training loop over batches
    for i, batch in enumerate(train_loader):
        if i % gradient_accumulation_steps == 0:
            # Keep every optimizer step at the configured effective batch size.
            if len(train_loader) - i < gradient_accumulation_steps:
                break
            if rank == 0:
                optimizer_step_start = time.time()
        clean_audio, clean_mag, clean_pha, clean_com, _, noisy_mag, noisy_pha = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
        del _
        clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
        clean_mag = torch.autograd.Variable(clean_mag.to(device, non_blocking=True))
        clean_pha = torch.autograd.Variable(clean_pha.to(device, non_blocking=True))
        clean_com = torch.autograd.Variable(clean_com.to(device, non_blocking=True))
        noisy_mag = torch.autograd.Variable(noisy_mag.to(device, non_blocking=True))
        noisy_pha = torch.autograd.Variable(noisy_pha.to(device, non_blocking=True))

        mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

        audio_g = mag_phase_istft(mag_g, pha_g, hps["stft_cfg"]["n_fft"], hps["stft_cfg"]["hop_size"], hps["stft_cfg"]["win_size"], hps["model_cfg"]["compress_factor"])

        # Discriminator
        # ------------------------------------------------------- #
        y_dq_hat_r, y_dq_hat_g, _, _ = mssbcqtd(clean_audio.unsqueeze(1), audio_g.unsqueeze(1).detach())
        loss_disc_q, losses_disc_q_r, losses_disc_q_g = discriminator_loss(
            y_dq_hat_r, y_dq_hat_g
        )
        # MRD
        y_dr_hat_r, y_dr_hat_g, _, _ = mrd(clean_audio.unsqueeze(1), audio_g.unsqueeze(1).detach())
        loss_disc_r, losses_disc_r_r, losses_disc_r_g = discriminator_loss(
            y_dr_hat_r, y_dr_hat_g
        )

        loss_disc_all = loss_disc_q + loss_disc_r
        
        (loss_disc_all / gradient_accumulation_steps).backward()
        # ------------------------------------------------------- #
        
        # Generator
        # ------------------------------------------------------- #
        for parameter in discriminator_parameters:
            parameter.requires_grad_(False)

        y_dq_hat_r, y_dq_hat_g, fmap_q_r, fmap_q_g = mssbcqtd(clean_audio.unsqueeze(1), audio_g.unsqueeze(1))
        loss_fm_q = feature_loss(fmap_q_r, fmap_q_g)
        loss_gen_q, losses_gen_q = generator_loss(y_dq_hat_g)

        # MRD loss
        y_dr_hat_r, y_dr_hat_g, fmap_r_r, fmap_r_g = mrd(clean_audio.unsqueeze(1), audio_g.unsqueeze(1))
        loss_fm_r = feature_loss(fmap_r_r, fmap_r_g)
        loss_gen_r, losses_gen_r = generator_loss(y_dr_hat_g)

        adv_g_loss = loss_gen_q + loss_gen_r
        fm_g_loss = loss_fm_q + loss_fm_r
        # Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/train.py
        # L2 Magnitude Loss
        loss_mag = F.mse_loss(clean_mag, mag_g)
        # Anti-wrapping Phase Loss
        loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, hps)
        loss_pha = loss_ip + loss_gd + loss_iaf
        # L2 Complex Loss
        loss_com = F.mse_loss(clean_com, com_g) * 2
        # Time Loss
        loss_time = F.l1_loss(clean_audio, audio_g)

        # Consistency Loss
        _, _, rec_com = mag_phase_stft(audio_g, hps["stft_cfg"]["n_fft"], hps["stft_cfg"]["hop_size"], hps["stft_cfg"]["win_size"], hps["model_cfg"]["compress_factor"], addeps=True)
        loss_con = F.mse_loss(com_g, rec_com) * 2


        mel_loss = fn_mel_loss_multiscale(clean_audio.unsqueeze(1), audio_g.unsqueeze(1))


        loss_gen_all = (
            adv_g_loss * hps['training_cfg']['loss']['adv_g'] +
            fm_g_loss * hps['training_cfg']['loss']['fm_g'] +
            mel_loss * hps['training_cfg']['loss']['mel'] +
            loss_mag * hps['training_cfg']['loss']['magnitude'] +
            loss_pha * hps['training_cfg']['loss']['phase'] +
            loss_com * hps['training_cfg']['loss']['complex'] +
            loss_time * hps['training_cfg']['loss'].get('time', 0.0) +
            loss_con * hps['training_cfg']['loss']['consistancy']
        )

        (loss_gen_all / gradient_accumulation_steps).backward()
        for parameter in discriminator_parameters:
            parameter.requires_grad_(True)

        if rank == 0:
            with torch.no_grad():
                optimizer_metrics = {
                    "loss": loss_gen_all,
                    "loss_generator": loss_gen_all,
                    "loss_adv_g": adv_g_loss,
                    "loss_discriminator": loss_disc_all,
                    "loss_feature_matching": fm_g_loss,
                    "loss_magnitude": F.mse_loss(clean_mag, mag_g),
                    "loss_phase": loss_pha,
                    "loss_complex": F.mse_loss(clean_com, com_g),
                    "loss_time": loss_time,
                    "loss_consistency": F.mse_loss(com_g, rec_com),
                    "loss_mel": mel_loss,
                }
                add_metric_window(optimizer_metric_sums, optimizer_metrics)
                optimizer_metric_count += 1

        if (i + 1) % gradient_accumulation_steps != 0:
            continue

        grad_norm_d = grad_norm(discriminator_parameters)
        grad_norm_g = grad_norm(generator.parameters())
        optim_d.step()
        optim_d.zero_grad()
        optim_g.step()
        optim_g.zero_grad()
        

        if rank == 0:
                global_step = steps + 1
                optimizer_step_elapsed_sec = time.time() - optimizer_step_start
                with torch.no_grad():
                    train_metrics = average_metric_window(optimizer_metric_sums, optimizer_metric_count)
                    train_metrics["grad_norm_g"] = grad_norm_g
                    train_metrics["grad_norm_d"] = grad_norm_d
                    add_metric_window(train_metric_sums, train_metrics)
                    train_metric_count += 1
                    train_window_elapsed_sec += optimizer_step_elapsed_sec
                    optimizer_step_samples = clean_audio.size(0) * n_gpus * gradient_accumulation_steps
                    samples_seen += optimizer_step_samples
                    train_window_samples += optimizer_step_samples
                    optimizer_metric_sums = {}
                    optimizer_metric_count = 0

                # STDOUT logging
                if global_step % a.stdout_interval == 0:
                    print(
                        'Steps : {:d}, Gen Loss: {:4.3f}, Disc Loss: {:4.3f}, adv_g_loss Loss: {:4.3f}, '
                        'fm_g_loss: {:4.3f}, Mag Loss: {:4.3f}, Pha Loss: {:4.3f}, Com Loss: {:4.3f}, Time Loss: {:4.3f}, Mel Loss: {:4.3f}, Cons Loss: {:4.3f}, s/b : {:4.3f}'.format(
                            global_step,
                            scalar_value(train_metrics["loss_generator"]),
                            scalar_value(train_metrics["loss_discriminator"]),
                            scalar_value(train_metrics["loss_adv_g"]),
                            scalar_value(train_metrics["loss_feature_matching"]),
                            scalar_value(train_metrics["loss_magnitude"]),
                            scalar_value(train_metrics["loss_phase"]),
                            scalar_value(train_metrics["loss_complex"]),
                            scalar_value(train_metrics["loss_time"]),
                            scalar_value(train_metrics["loss_mel"]),
                            scalar_value(train_metrics["loss_consistency"]),
                            optimizer_step_elapsed_sec,
                        )
                    )

                # Checkpointing
                if global_step % a.checkpoint_interval == 0:
                    generator_state = {
                        'generator': (generator.module if n_gpus > 1 else generator).state_dict()
                    }
                    optimizer_state = {
                        'mssbcqtd': (mssbcqtd.module if n_gpus > 1 else mssbcqtd).state_dict(),
                        'mrd': (mrd.module if n_gpus > 1 else mrd).state_dict(),
                        'optim_g': optim_g.state_dict(),
                        'optim_d': optim_d.state_dict(),
                        'steps': global_step,
                        'epoch': epoch,
                        'best_avqi_gap_to_clean': getattr(
                            generator.module if n_gpus > 1 else generator,
                            "_best_avqi_gap_to_clean",
                            float("inf"),
                        ),
                        'latest_avqi_gap_to_clean': getattr(
                            generator.module if n_gpus > 1 else generator,
                            "_latest_avqi_gap_to_clean",
                            None,
                        ),
                        'best_guarded_val_loss': getattr(
                            generator.module if n_gpus > 1 else generator,
                            "_best_guarded_val_loss",
                            float("inf"),
                        ),
                    }
                    exp_name = f"{a.exp_path}/ln_g_{global_step:08d}.pth"
                    save_checkpoint(
                        exp_name,
                        generator_state
                    )
                    exp_name = f"{a.exp_path}/ln_do_{global_step:08d}.pth"

                    save_checkpoint(
                        exp_name,
                        optimizer_state
                    )
                    prune_step_checkpoints(a.exp_path, keep=3, prefixes=("ln_g_", "ln_do_"))

                if avqi_interval > 0 and rank == 0 and global_step % avqi_interval == 0:
                    avqi_metrics = log_validation_avqi_metrics(generator, hps, device, global_step)
                    gap = avqi_metrics["avqi_gap_to_clean"]
                    (generator.module if n_gpus > 1 else generator)._latest_avqi_gap_to_clean = gap
                    print(
                        "Steps : {:d}, AVQI gap to clean: {:4.3f}".format(
                            global_step,
                            gap,
                        )
                    )
                    if a.use_wandb:
                        avqi_log = {"charts/global_step": global_step}
                        for name, value in avqi_metrics.items():
                            avqi_log[f"val_avqi/{name}"] = value
                        wandb.log(
                            avqi_log,
                            step=global_step,
                        )
                    save_best_avqi_gap_checkpoints(
                        generator, [mssbcqtd, mrd], [optim_g, optim_d], a.exp_path, epoch, global_step, n_gpus, gap
                    )

                # Tensorboard summary logging
                if a.use_wandb and global_step % summary_interval == 0:
                    train_log = {
                        "charts/epoch": epoch + 1,
                        "charts/global_step": global_step,
                        "charts/lr": optim_g.param_groups[0]['lr'],
                        "charts/samples_per_sec": train_window_samples / train_window_elapsed_sec,
                    }
                    averaged_train_metrics = average_metric_window(train_metric_sums, train_metric_count)
                    if "grad_norm_g" in averaged_train_metrics:
                        train_log["charts/grad_norm"] = averaged_train_metrics["grad_norm_g"]
                    for name, value in averaged_train_metrics.items():
                        if name.startswith("grad_norm"):
                            train_log[f"charts/{name}"] = value
                        else:
                            train_log[f"train/{name}"] = value
                    wandb.log(train_log, step=global_step)
                    train_metric_sums = {}
                    train_metric_count = 0
                    train_window_elapsed_sec = 0.0
                    train_window_samples = 0

                # If NaN happend in training period, RaiseError
                if torch.isnan(loss_gen_all).any():
                    raise ValueError("NaN values found in loss_gen_all")

                # Validation
                if validation_loader is not None and global_step % validation_interval == 0:
                    print("Validation Started...")
                    validation_start_time = time.time()
                    generator.eval()
                    torch.cuda.empty_cache()
                    audios_r, audios_g = [], []
                    val_mag_err_tot = 0
                    val_pha_err_tot = 0
                    val_com_err_tot = 0
                    val_time_err_tot = 0
                    val_mrstft_score = 0
                    with torch.no_grad():
                        validation_dataset = getattr(validation_loader, "dataset", None)
                        for j, batch in enumerate(validation_loader):
                            clean_audio, clean_mag, clean_pha, clean_com, noisy_audio, noisy_mag, noisy_pha = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
                            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
                            noisy_audio = torch.autograd.Variable(noisy_audio.to(device, non_blocking=True))
                            clean_mag = torch.autograd.Variable(clean_mag.to(device, non_blocking=True))
                            clean_pha = torch.autograd.Variable(clean_pha.to(device, non_blocking=True))
                            clean_com = torch.autograd.Variable(clean_com.to(device, non_blocking=True))

                            mag_g, pha_g, com_g = generator(noisy_mag.to(device), noisy_pha.to(device))

                            audio_g = mag_phase_istft(mag_g, pha_g, hps["stft_cfg"]["n_fft"], hps["stft_cfg"]["hop_size"], hps["stft_cfg"]["win_size"], hps["model_cfg"]["compress_factor"])
                            
                            #audios_r += torch.split(clean_audio, 1, dim=0) # [1, T] * B
                            #audios_g += torch.split(audio_g, 1, dim=0)

                            val_mag_err_tot += F.mse_loss(clean_mag, mag_g).item()
                            val_ip_err, val_gd_err, val_iaf_err = phase_losses(clean_pha, pha_g, hps)
                            val_pha_err_tot += (val_ip_err + val_gd_err + val_iaf_err).item()
                            val_com_err_tot += F.mse_loss(clean_com, com_g).item()

                            # Trim audio_g to fit clean_audio length
                            if audio_g.size(1) > clean_audio.size(1):
                                audio_g = audio_g[:, :clean_audio.size(1)]
                            elif audio_g.size(1) < clean_audio.size(1):
                                clean_audio = clean_audio[:, :audio_g.size(1)]
                            val_time_err_tot += F.l1_loss(clean_audio, audio_g).item()

                            score_metrics = compute_val_metrics(mrstft, clean_audio, audio_g)

                            val_mrstft_score += score_metrics["mrstft_score"].item()

                        val_mag_err = val_mag_err_tot / (j+1)
                        val_pha_err = val_pha_err_tot / (j+1)
                        val_com_err = val_com_err_tot / (j+1)
                        val_time_err = val_time_err_tot / (j+1)
                        val_mrstft_score = val_mrstft_score / (j+1)
                        loss_cfg = hps["training_cfg"]["loss"]
                        val_loss = (
                            val_mag_err * loss_cfg.get("magnitude", 0.0)
                            + val_pha_err * loss_cfg.get("phase", 0.0)
                            + val_com_err * loss_cfg.get("complex", 0.0)
                            + val_time_err * loss_cfg.get("time", 0.0)
                        )

                        print('Steps : {:d}, MRSTFT Score: {:4.3f}, Mag Loss: {:4.3f}, Pha Loss: {:4.3f}, Com Loss: {:4.3f}, Time Loss: {:4.3f}, s/b : {:4.3f}'.
                                format(global_step, val_mrstft_score, val_mag_err, val_pha_err, val_com_err, val_time_err, time.time() - validation_start_time))

                        if a.use_wandb:
                            validation_log = {
                                "charts/epoch": epoch + 1,
                                "charts/global_step": global_step,
                                "val/loss": val_loss,
                                "val/mrstft": val_mrstft_score,
                                "val/loss_magnitude": val_mag_err,
                                "val/loss_phase": val_pha_err,
                                "val/loss_time": val_time_err,
                                "val/loss_complex": val_com_err,
                            }
                            wandb.log(validation_log, step=global_step)
                        save_best_guarded_checkpoints(
                            generator,
                            [mssbcqtd, mrd],
                            [optim_g, optim_d],
                            a.exp_path,
                            epoch,
                            global_step,
                            n_gpus,
                            val_loss,
                        )

                    generator.train()

        completed_step = steps + 1
        if a.max_steps > 0 and completed_step >= a.max_steps:
            steps = completed_step
            if n_gpus > 1:
                dist.barrier()
            return

        steps = completed_step

    scheduler_g.step()
    scheduler_d.step()




def main():

    print('Initializing Training Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/semambapp_default.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    config_path = os.path.expanduser(os.path.expandvars(args.config))

    # Ensure CUDA availability for SEMamba++ training
    assert torch.cuda.is_available(), "SEMamba++ training requires CUDA."

    # Load SEMamba++ configuration
    hps = load_config(config_path)
    a = build_runtime_args(config_path, hps)

    # Setup multi-GPU SEMamba++ training
    n_gpus = torch.cuda.device_count()
    hps["training_cfg"]["batch_size"] = hps["training_cfg"]["batch_size"] // n_gpus  # Divide batch size by number of GPUs
    print("The number of GPUs used for SEMamba++ training is:", n_gpus)
    print("SEMamba++ batch size per GPU is set to:", hps["training_cfg"]["batch_size"])

    port = 50000 + random.randint(0, 100)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    hps["env_setting"]["num_gpus"] = n_gpus
    
    # Launch SEMamba++ training
    if n_gpus > 1:
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, a, hps,))
    else:
        run(0, n_gpus, a, hps)

if __name__ == "__main__":
    main()
