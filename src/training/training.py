import torch
from accelerate import Accelerator
from tqdm.auto import tqdm

from src.datasets.discrete_helper import beta_t, sample_t, theta, y_distribution
from src.inference.discrete_inference import bayesian_inference, dis_t
from src.training.checkpoint import CheckpointManager
from src.training.discrete_loss import loss


class TrainingContext:
    def __init__(
        self,
        model,
        optimizer,
        train_dl,
        epochs,
        accelerator: Accelerator,
        checkpoint_manager: CheckpointManager,
        save_dir: str,
        starting_epoch: int = 0,
        save_every: int = 10_000,
        test_every: int = 3_000,
        test_inference_steps: int = 100,
        test_dl=None,
        test_step_fn=None,  # overwrite for mocking
        train_step_fn=None,  # overwrite for mocking
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.epochs = epochs
        self.accelerator = accelerator
        self.checkpoint_manager = checkpoint_manager
        self.save_dir = save_dir
        self.starting_epoch = starting_epoch
        self.save_every = save_every
        self.test_every = test_every
        self.test_inference_steps = test_inference_steps
        self.test_dl = test_dl
        self.test_step_fn = test_step_fn
        self.train_step_fn = train_step_fn

        self.train_iter = None
        self.test_iter = None

    def train_data(self):
        if self.train_iter is None:
            self.train_iter = iter(self.train_dl)
        try:
            data = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_dl)
            data = next(self.train_iter)
        return data

    def test_data(self):
        if self.test_dl is None:
            return None
        if self.test_iter is None:
            self.test_iter = iter(self.test_dl)
        try:
            data = next(self.test_iter)
        except StopIteration:
            self.test_iter = iter(self.test_dl)
            data = next(self.test_iter)
        return data

    def save_checkpoint(self, epoch, always_save=False):
        if always_save or epoch % self.save_every == 0:
            self.checkpoint_manager.save(self.save_dir, epoch)

    def end_training(self):
        self.accelerator.end_training()


def train_step(context: TrainingContext, pbar, current_epoch):
    model = context.model
    optimizer = context.optimizer
    accelerator = context.accelerator
    model.train()
    data = context.train_data()
    x = data["x"]
    t = data["t"]
    beta_1 = data["beta_1"]
    model_input = data["theta"]  # batch_size, seq_len, K
    output = model(model_input, t)
    l = loss(beta_1, t, x, model_output_logits=output)
    optimizer.zero_grad()
    accelerator.backward(l)
    optimizer.step()
    pbar.set_postfix({"loss": l.item()}, step=current_epoch)
    accelerator.log({"train/loss": l.item()}, step=current_epoch)


def test_step(context: TrainingContext, pbar, current_epoch):
    with torch.no_grad():
        model = context.model
        accelerator = context.accelerator
        test_data = context.test_data()
        if test_data is None:
            return
        model.eval()
        x = test_data["x"]
        t = test_data["t"]
        beta_1 = test_data["beta_1"]
        model_input = test_data["theta"]  # batch_size, seq_len, K
        batch_size, seq_len, K = model_input.shape

        total_iterations = torch.ones_like(t) * context.test_inference_steps

        # the first half of the sequence will be kept the other half will be replaced with model output
        # this will serve as a janky conditional generation test
        indices = torch.arange(seq_len, device=model_input.device)
        lower_half = indices < (seq_len // 2)
        mask = lower_half.unsqueeze(0).unsqueeze(-1)  # for broadcasting

        # the model_input we are given might not be produced by beta at t = 0, so we need to use
        # x to recreate a new model_input
        beta_0 = beta_t(beta_1, t * 0)
        model_input_acc = theta(y_distribution(beta_0, K, x))

        # since x is currently one-hot, it won't work well when used as logits for the categorical distribution
        # used by bayesian_inference in its sampling step, so we need to make all zeros -inf values
        x_zero = x.clone().float()
        x_zero[x_zero == 0] = float("-inf")

        for i in range(1, context.test_inference_steps + 1):
            current_iteration = torch.ones_like(t) * i
            t_curr = dis_t(current_iteration, total_iterations)
            output = model(model_input_acc, t_curr)
            output = torch.where(mask, x_zero, output)
            model_input_acc = bayesian_inference(
                model_input_acc, output, current_iteration, total_iterations, beta_1
            )

        # now for each position, get the most likely token. However, we only care about the
        # positions where the mask is False (which is the sequence the model is generating)
        expanded_mask = mask.expand(batch_size, -1, K)
        predicted = model_input_acc[~expanded_mask].view(batch_size, -1, K)
        target = x[~expanded_mask].view(batch_size, -1, K)
        accuracy = (
            (predicted.argmax(dim=-1) == target.argmax(dim=-1)).float().mean().item()
        )

        # as for the original model_input, we just calculate the loss as usual
        output = model(model_input, t)
        l = loss(beta_1, t, x, model_output_logits=output)

        accelerator.log(
            {"test/loss": l.item(), "test/accuracy": accuracy}, step=current_epoch
        )


def train_discrete_model(context: TrainingContext):
    start_epoch = context.starting_epoch
    end_epoch = start_epoch + context.epochs
    pbar = tqdm(range(start_epoch, end_epoch), desc="Training Discrete Model")
    train_fn = context.train_step_fn or train_step
    test_fn = context.test_step_fn or test_step
    try:
        for epoch in pbar:
            train_fn(context, pbar, epoch)
            if epoch % context.test_every == 0:
                test_fn(context, pbar, epoch)
            context.save_checkpoint(epoch)
        context.save_checkpoint(epoch, always_save=True)
        context.end_training()
    except KeyboardInterrupt:
        print("Training interrupted. Exiting...")
        context.save_checkpoint(epoch, always_save=True)
        context.end_training()
    # finally:
    #     context.end_training()
    #     # we don't save here because if it was a runtime error, the model may be in a bad state
