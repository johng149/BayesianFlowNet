from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.scheduler import AcceleratedScheduler
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.checkpointing.checkpointing import load_checkpoint, save_checkpoint
from src.datasets.dataset_helper import CollateOutput
from src.inference.conditional import half_callback_maker
from src.inference.generate import inference
from src.schedule.base import Scheduler
from src.training.loss import loss


class TrainingContext:
    def __init__(
        self,
        save_file_name: str,
        accelerator: Accelerator,
        model: Module,
        scheduler: Scheduler,
        optim: Optimizer | None,
        train_loader: DataLoader,
        test_loader: DataLoader | None = None,
        lr_scheduler: LRScheduler | None = None,
        target_epochs: int = 100,
        seen_epochs: int = 0,
        test_every: int = 125,
        save_every: int = 512,
        test_inference_steps: int = 100,
        save_dir: str | Path = "./checkpoints",
        metadata_save_file_name: str = "metadata.json",
        aux_weight: float = 0.03,  # weight for auxiliary loss
        grad_clip_norm: float | None = 1.0,
    ):
        assert (
            grad_clip_norm is None or grad_clip_norm > 0.0
        ), "grad_clip_norm must be positive or None"
        assert aux_weight >= 0.0, "aux_weight must be non-negative"
        assert test_every > 0, "test_every must be positive"
        assert save_every > 0, "save_every must be positive"
        assert target_epochs > 0, "target_epochs must be positive"
        assert test_inference_steps > 0, "test_inference_steps must be positive"

        # note that currently scheduler is not directly saved in the checkpoint, if you
        # want the scheduler to be saved (say, if you are using parameterized schedulers),
        # you need to include it in the model as a submodule.
        # actually, you might want to use the accelerator's `register_for_checkpointing` function
        # instead. For example, if the schedule is a EMA parameterized scheduler, saving and loading
        # in distributed setting (which would happen if the scheduler was a submodule of the model)
        # would be trickier.
        self.save_file_name = save_file_name
        self.accelerator = accelerator
        self.model = model
        self.scheduler = scheduler
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.target_epochs = target_epochs
        self.seen_epochs = seen_epochs
        self.test_every = test_every
        self.save_every = save_every
        self.test_inference_steps = test_inference_steps
        self.save_dir = Path(save_dir)
        self.metadata_save_file_name = metadata_save_file_name
        self.aux_weight = aux_weight
        self.grad_clip_norm = grad_clip_norm

        self.train_loader, self.test_loader = self.accelerator.prepare(
            self.train_loader, self.test_loader
        )

        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader) if self.test_loader else None

        self.accelerator.init_trackers(save_file_name)

    def get_train_data(self) -> CollateOutput:
        try:
            batch = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            batch = next(self.train_iter)
        return batch

    def get_test_data(self) -> CollateOutput | None:
        if self.test_loader is None:
            return None
        try:
            assert self.test_iter is not None
            batch = next(self.test_iter)
        except StopIteration:
            self.test_iter = iter(self.test_loader)
            batch = next(self.test_iter)
        return batch

    def log(self, tag: str, value: float, step: int):
        """
        Logs a scalar value to TensorBoard. It will appear in a graph named `tag`
        """
        self.accelerator.log({tag: value}, step=step)

    def save(self, epoch: int):
        checkpoint_path = self.save_dir / self.save_file_name
        save_checkpoint(
            accelerator=self.accelerator,
            path=checkpoint_path,
            seen_epochs=epoch + 1,
        )

    def load(self, ignore_missing: bool = True):
        checkpoint_path = self.save_dir / self.save_file_name
        self.model, self.optim, self.seen_epochs, self.lr_scheduler = load_checkpoint(
            model=self.model,
            optim=self.optim,
            lr_scheduler=self.lr_scheduler,
            accelerator=self.accelerator,
            path=checkpoint_path,
            ignore_missing=ignore_missing,
        )

    def get_seen_epochs(self) -> int:
        return self.seen_epochs if self.seen_epochs is not None else 0

    def end_training(self):
        self.accelerator.end_training()

    def get_optimizer(self) -> Optimizer:
        # use only for when actually training. During inference you might not have an optimizer
        assert self.optim is not None
        return self.optim

    def is_fsdp_environment(self) -> bool:
        return (
            hasattr(self.accelerator.state, "fsdp_plugin")
            and self.accelerator.state.fsdp_plugin is not None
        )

    def get_lr_scheduler(self) -> AcceleratedScheduler | None:
        assert (
            isinstance(self.lr_scheduler, AcceleratedScheduler)
            or self.lr_scheduler is None
        ), "By the time you call this, the lr_scheduler should have been wrapped by the accelerator due to the load from checkpoint function"
        return self.lr_scheduler


def train_step(context: TrainingContext, current_epoch: int):
    batch: CollateOutput = context.get_train_data()
    model = context.model
    model.train()
    optim = context.get_optimizer()
    lr_scheduler = context.get_lr_scheduler()

    model_input = batch["model_input"]
    t = batch["t"]
    mask = batch["mask"]
    ground_truth = batch["ground_truth"]
    scheduler_output = batch["scheduler_output"]
    doc_ids = batch["document_id"]
    loss_context = {
        "scheduler_output": scheduler_output,
        "target": ground_truth,
        "mask": mask,
    }
    prediction, l = model(model_input, t, mask, doc_ids, loss_context)
    optim.zero_grad()
    context.accelerator.backward(l)
    if context.grad_clip_norm is not None and context.accelerator.sync_gradients:
        context.accelerator.clip_grad_norm_(model.parameters(), context.grad_clip_norm)
    optim.step()
    if lr_scheduler is not None:
        lr_scheduler.step(metrics=l)
    context.log("train/loss", l.item(), current_epoch)


def test_step(context: TrainingContext, current_epoch: int):
    batch: CollateOutput | None = context.get_test_data()
    if batch is None:
        return
    model = context.model
    scheduler = context.scheduler
    model.eval()

    model_input = batch["model_input"]
    t = batch["t"]
    ground_truth = batch["ground_truth"]
    scheduler_output = batch["scheduler_output"]
    mask = batch["mask"]
    doc_ids = batch["document_id"]

    batch_size, seq_len, K = model_input.shape

    steps = context.test_inference_steps

    inference_result = inference(
        model=model,
        scheduler=scheduler,
        num_steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        K=K,
        mask=mask,
        masked_input=model_input,
        doc_ids=doc_ids,
        device=model_input.device,
        dtype=model_input.dtype,
    )

    # predicted = masker(inference_result)
    # target = masker(ground_truth)

    # we only care about accuracy for masked positions
    match = inference_result.argmax(dim=-1) == ground_truth.argmax(dim=-1)

    accuracy = match[mask].float().mean().item()
    loss_context = {
        "scheduler_output": scheduler_output,
        "target": ground_truth,
        "mask": mask,
    }
    prediction, l = model(model_input, t, mask, doc_ids, loss_context)

    context.log("test/loss", l.item(), current_epoch)
    context.log("test/accuracy", accuracy, current_epoch)


def train(context: TrainingContext):
    total_epochs = context.target_epochs
    starting_epoch = context.get_seen_epochs()

    if starting_epoch >= total_epochs:
        print("Training already completed.")
        return

    pbar = tqdm(range(starting_epoch, total_epochs), desc="Training", unit="epoch")
    try:
        for epoch in pbar:
            train_step(context, epoch + 1)

            if (epoch + 1) % context.test_every == 0:
                # note we don't check to do this only on the main process, because for whatever
                # reason, if we are in FSDP mode and we try to run only on the main process,
                # the script just hangs
                test_step(context, epoch + 1)

            if (epoch + 1) % context.save_every == 0:
                context.save(epoch + 1)
        context.save(epoch + 1)
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        context.save(epoch + 1)
    finally:
        context.end_training()
