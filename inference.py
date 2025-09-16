import os
import random

import torch
from accelerate import Accelerator
from matplotlib import pyplot as plt
from safetensors.torch import load_file
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.datasets.collate_fn import collate_fn_maker
from src.datasets.shakespeare.shakespeare import ShakespeareDataset
from src.inference.discrete_inference import bayesian_inference, dis_t
from src.nn.layers.learnable_schedule import LearnableBetaScheduleNI
from src.nn.models.discrete_model import DiscreteModel
from src.tokenizers.byt5.byt5_tokenizer import ByT5Tokenizer as Tokenizer
from src.training.checkpoint import CheckpointManager, CheckpointMetadata
from src.training.training import train_discrete_model

accelerator = Accelerator(project_dir="./runs/shakespeare")
tokenizer = Tokenizer()
max_seq_len = 168
test_ds = ShakespeareDataset(
    tokenizer=tokenizer, max_length=max_seq_len, folds=1, train=False
)

collate_fn = collate_fn_maker(
    tokenizer=tokenizer, max_masks=3, min_masks=1, max_fill=0.95, min_fill=0.05
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

optimizer_kwargs = {
    "lr": 3e-5,
}
body_opt = torch.optim.Adam(
    model.body.parameters(), **optimizer_kwargs  # pyright: ignore[reportArgumentType]
)
schedule_opt = torch.optim.Adam(
    model.learnable_beta.parameters(),
    **optimizer_kwargs,  # pyright: ignore[reportArgumentType]
)

metadata = CheckpointMetadata(
    model_kwargs=model_kwargs,
    optimizer_kwargs=optimizer_kwargs,
    is_fsdp=hasattr(accelerator.state, "fsdp_plugin")
    and accelerator.state.fsdp_plugin is not None,
    num_accelerators=accelerator.num_processes,
)

checkpoint_dir = "./checkpoint/shakespeare_byt5_baseline"
checkpoint_manager = CheckpointManager()
print("Preparing model...")
checkpoint_manager.prepare(model, body_opt, schedule_opt, accelerator, metadata)
print("Starting checkpoint loading process...")
checkpoint_manager.load(checkpoint_dir, error_if_not_exists=True)
print("Finished loading checkpoint")

model, opt = checkpoint_manager.model, checkpoint_manager.body_optimizer

assert model is not None
assert isinstance(model, DiscreteModel)

schedule: LearnableBetaScheduleNI = model.learnable_beta

assert isinstance(schedule, LearnableBetaScheduleNI)

quit_loop = False
while not quit_loop:
    torch.manual_seed(0)
    sample_idx = random.randint(0, len(test_ds) - 1)
    raw_sample = test_ds[sample_idx]
    collated_sample = collate_fn([raw_sample])

    enc = collated_sample["encoder_input"].to(accelerator.device)
    model_input = collated_sample["target"].to(accelerator.device)
    expected = tokenizer.decode(model_input.squeeze(0), None)
    expected_combined = tokenizer.decode(model_input.squeeze(0), enc.squeeze(0))

    try:
        n = int(input("Enter number of iterations (e.g., 100): "))
    except ValueError:
        print("Invalid input. Using default value of 250.")
        n = 250

    for i in tqdm(range(1, n + 1)):
        if i % 10 == 0:
            tqdm.write(tokenizer.decode(model_input.squeeze(0)))
        cur_it = torch.tensor([i], device=accelerator.device)
        total_it = torch.tensor([n], device=accelerator.device)
        t = dis_t(cur_it, total_it).to(accelerator.device)

        model_output, _ = model.forward(enc, model_input, t)
        model_input = bayesian_inference(
            model_input,
            model_output,
            cur_it,
            total_it,
            schedule,
            tokenizer.vocab_size(),
        )

    tqdm.write("Final model result:")
    tqdm.write(f"Final model output: {tokenizer.decode(model_input.squeeze(0), None)}")
    tqdm.write(f"Expected output: {expected}")
    tqdm.write(
        f"Encoder input: {tokenizer.decode(F.one_hot(enc.squeeze(0), num_classes=tokenizer.vocab_size()).float(), None)}"
    )
    tqdm.write(f"Combined: {tokenizer.decode(model_input.squeeze(0), enc.squeeze(0))}")
    tqdm.write(f"Expected combined: {expected_combined}")

    user_input = input("Continue? (y/n): ")
    if user_input.lower() != "y":
        quit_loop = True
