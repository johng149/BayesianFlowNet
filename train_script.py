import os

import torch
from accelerate import Accelerator
from matplotlib import pyplot as plt
from safetensors.torch import load_file

from src.datasets.discrete_helper import collate_fn
from src.datasets.shakespeare.shakespeare import ShakespeareDataset
from src.inference.discrete_inference import bayesian_inference, dis_t
from src.nn.models.discrete_model import DiscreteModel
from src.tokenizers.ascii.ascii_tokenizer import ASCIITokenizer as Tokenizer
from src.training.checkpoint import CheckpointManager, CheckpointMetadata
from src.training.training import train_discrete_model

accelerator = Accelerator(log_with="tensorboard", project_dir="./runs")
tokenizer = Tokenizer()
max_seq_len = 32
train_ds = ShakespeareDataset(tokenizer=tokenizer, max_length=max_seq_len, beta_1=0.5)
train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=3 if os.name != "nt" else 0,
)


model_kwargs = {
    "max_seq_len": max_seq_len,
    "K": tokenizer.vocab_size(),
    "hidden_dim": 512,
    "num_heads": 8,
    "layers": 5,
}
model = DiscreteModel(**model_kwargs)

grad_clip_norm = 1.0

optimizer_kwargs = {
    "lr": 1e-5,
}
opt = torch.optim.Adam(
    model.parameters(), **optimizer_kwargs  # pyright: ignore[reportArgumentType]
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
    "shakespeare_chonky_silu_xavier_1e-5_beta05_ASCIITokenizer_big_data",
)

checkpoint_dir = (
    "./checkpoint/shakespeare_chonky_silu_xavier_1e-5_beta05_ASCIITokenizer_big_data"
)
checkpoint_manager = CheckpointManager()
checkpoint_manager.prepare(model, opt, accelerator, metadata)
checkpoint_manager.load(checkpoint_dir, error_if_not_exists=False)

model, opt = checkpoint_manager.model, checkpoint_manager.optimizer
train_dl = accelerator.prepare(train_dl)

epochs = 550_000

train_discrete_model(
    model,
    opt,
    train_dl,
    starting_epoch=checkpoint_manager.current_epoch,
    epochs=epochs,
    accelerator=accelerator,
    checkpoint_manager=checkpoint_manager,
    save_dir=checkpoint_dir,
    grad_clip_norm=grad_clip_norm,
    save_every=2400,
)

model_input = torch.normal(0, 1, (1, max_seq_len, tokenizer.vocab_size())).to(
    accelerator.device
)
model_input = torch.softmax(model_input, dim=-1) * 2 - 1

n = 190

for i in range(1, n + 1):
    if i % 10 == 0:
        tokenizer.decode(model_input.squeeze(0))
    cur_it = torch.tensor([i], device=accelerator.device)
    total_it = torch.tensor([n], device=accelerator.device)
    t = dis_t(cur_it, total_it).to(accelerator.device)
    dis_beta_1 = torch.ones_like(t, device=accelerator.device) * 0.5

    current_model_input = model_input.clone()

    model_output = model.forward(  # pyright: ignore[reportOptionalMemberAccess]
        model_input, t
    )
    model_input = bayesian_inference(
        model_input, model_output, cur_it, total_it, dis_beta_1
    )

print("Final model result:")
print(tokenizer.decode(model_input.squeeze(0)))
