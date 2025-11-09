from pathlib import Path

import torch
from accelerate import Accelerator
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import Optimizer
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
        target_epochs: int = 100,
        seen_epochs: int = 0,
        test_every: int = 125,
        save_every: int = 512,
        test_inference_steps: int = 100,
        save_dir: str | Path = "./checkpoints",
        metadata_save_file_name: str = "metadata.json",
    ):
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
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.target_epochs = target_epochs
        self.seen_epochs = seen_epochs
        self.test_every = test_every
        self.save_every = save_every
        self.test_inference_steps = test_inference_steps
        self.save_dir = Path(save_dir)
        self.metadata_save_file_name = metadata_save_file_name

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
        self.model, self.optim, self.seen_epochs = load_checkpoint(
            model=self.model,
            optim=self.optim,
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


def train_step(context: TrainingContext, current_epoch: int):
    batch: CollateOutput = context.get_train_data()
    model = context.model
    model.train()
    optim = context.get_optimizer()

    model_input = batch["model_input"]
    t = batch["t"]
    ground_truth = batch["ground_truth"]
    scheduler_output = batch["scheduler_output"]
    prediction = model(model_input, t)
    optim.zero_grad()
    l = loss(scheduler_output, ground_truth, prediction)
    context.accelerator.backward(l)
    optim.step()
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

    batch_size, seq_len, K = model_input.shape

    callback, masker = half_callback_maker(ground_truth)

    steps = context.test_inference_steps

    inference_result = inference(
        model,
        scheduler,
        steps,
        batch_size,
        seq_len,
        K,
        model_input.device,
        model_input.dtype,
        conditioning_callback=callback,
    )

    predicted = masker(inference_result)
    target = masker(ground_truth)
    accuracy = (predicted.argmax(dim=-1) == target.argmax(dim=-1)).float().mean().item()

    prediction = model(model_input, t)
    l = loss(scheduler_output, ground_truth, prediction)

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
