import os
import random

import torch
from accelerate import Accelerator
from matplotlib import pyplot as plt
from safetensors.torch import load_file

from src.datasets.collate_fn import collate_fn_maker
from src.datasets.shakespeare.shakespeare import ShakespeareDataset
from src.inference.discrete_inference import bayesian_inference, dis_t
from src.nn.layers.learnable_schedule import LearnableBetaScheduleNI
from src.nn.models.discrete_model import DiscreteModel
from src.tokenizers.byt5.byt5_tokenizer import ByT5Tokenizer as Tokenizer
from src.training.checkpoint import CheckpointManager, CheckpointMetadata
from src.training.training import train_discrete_model

accelerator = Accelerator(log_with="tensorboard", project_dir="./runs")
tokenizer = Tokenizer()
max_seq_len = 168
batch_size = 64 * 2
folds = 8
effective_batch_size = batch_size // folds

collate_fn = collate_fn_maker(
    tokenizer=tokenizer, max_masks=3, min_masks=1, max_fill=0.95, min_fill=0.05
)

train_ds = ShakespeareDataset(tokenizer=tokenizer, max_length=max_seq_len, folds=folds)
test_ds = ShakespeareDataset(
    tokenizer=tokenizer, max_length=max_seq_len, folds=1, train=False
)
train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=effective_batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=3 if os.name != "nt" else 0,
)
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=effective_batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=3 if os.name != "nt" else 0,
)

model_kwargs = {
    "max_seq_len": max_seq_len,
    "K": tokenizer.vocab_size(),
    "hidden_dim": 512,
    "num_heads": 8,
    "layers": 5,
    # beta_1 from https://arxiv.org/html/2407.20294v2 equation 5
    "reference_beta_1": 20.4054 / tokenizer.vocab_size(),
    "learner_weight": 1.0,
    "freeze_body": False,
}
model = DiscreteModel(**model_kwargs)

# count number of model parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of model parameters: {num_params}")

grad_clip_norm = None
skip_schedule_optim = False

optimizer_kwargs = {
    "body_optim_kwargs": {"lr": 1e-4},
    "schedule_optim_kwargs": {"lr": 5e-5},
}
body_opt = torch.optim.Adam(
    model.body.parameters(),
    **optimizer_kwargs["body_optim_kwargs"],  # pyright: ignore[reportArgumentType]
)
schedule_opt = torch.optim.Adam(
    model.learnable_beta.parameters(),
    **optimizer_kwargs["schedule_optim_kwargs"],  # pyright: ignore[reportArgumentType]
)

metadata = CheckpointMetadata(
    model_kwargs=model_kwargs,
    optimizer_kwargs=optimizer_kwargs,
    is_fsdp=hasattr(accelerator.state, "fsdp_plugin")
    and accelerator.state.fsdp_plugin is not None,
    num_accelerators=accelerator.num_processes,
    grad_clip_norm=grad_clip_norm,
)

accelerator.init_trackers(
    "shakespeare_byt5_learnt2",
)

checkpoint_dir = "./checkpoint/shakespeare_byt5_learnt2"
checkpoint_manager = CheckpointManager()
checkpoint_manager.prepare(model, body_opt, schedule_opt, accelerator, metadata)
checkpoint_manager.load(checkpoint_dir, error_if_not_exists=False)

model, body_opt, schedule_opt = (
    checkpoint_manager.model,
    checkpoint_manager.body_optimizer,
    checkpoint_manager.schedule_optimizer,
)
train_dl, test_dl = accelerator.prepare(train_dl, test_dl)

assert model is not None
assert isinstance(model, DiscreteModel)

# check that all parameters in model body is frozen
# for param in model.body.parameters():
#     assert not param.requires_grad

epochs = 1_300_000

train_discrete_model(
    model,
    body_opt,
    schedule_opt,
    train_dl,
    starting_epoch=checkpoint_manager.current_epoch,
    epochs=epochs,
    accelerator=accelerator,
    checkpoint_manager=checkpoint_manager,
    save_dir=checkpoint_dir,
    grad_clip_norm=grad_clip_norm,
    save_every=35_000,
    folds=folds,
    test_every=4_000,
    test_dl=test_dl,
    test_dl_inference_steps=100,
    variance_loss_strength=0.8,
    divergence_loss_strength=0.6,
    skip_schedule_optim=skip_schedule_optim,
)

schedule = model.learnable_beta
assert isinstance(schedule, LearnableBetaScheduleNI)

sample_idx = random.randint(0, len(train_ds) - 1)
raw_sample = train_ds[sample_idx]

# dummy to make folds appear to be 1 in collate_fn, if we did not do this, then collate_fn
# would duplicate the model inputs `folds` times, which is fine during training but not for this
# demonstration inference script
raw_sample["t"] = torch.tensor([0.0])
collated_sample = collate_fn([raw_sample])

enc = collated_sample["encoder_input"].to(accelerator.device)
model_input = collated_sample["target"].to(accelerator.device)

n = 190

for i in range(1, n + 1):
    if i % 10 == 0:
        tokenizer.decode(model_input.squeeze(0))
    cur_it = torch.tensor([i], device=accelerator.device)
    total_it = torch.tensor([n], device=accelerator.device)
    t = dis_t(cur_it, total_it).to(accelerator.device)

    model_output_logits, _ = model.forward(enc, model_input, t)
    model_input = bayesian_inference(
        model_input,
        model_output_logits,
        cur_it,
        total_it,
        schedule,
        tokenizer.vocab_size(),
    )

print("Final model result:")
print(tokenizer.decode(model_input.squeeze(0)))
