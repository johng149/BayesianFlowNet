import random
from tracemalloc import start

import numpy as np
import torch
from accelerate import Accelerator

from src.datasets.discrete_helper import collate_fn
from src.datasets.shakespeare.shakespeare import ShakespeareDataset
from src.nn.models.discrete_model import DiscreteModel
from src.optimizers.c_adamw import AdamW as Opt
from src.tokenizers.byt5.byt5_tokenizer import ByT5Tokenizer as Tokenizer
from src.training.checkpoint import CheckpointManager, CheckpointMetadata
from src.training.training import TrainingContext, train_discrete_model

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

accelerator = Accelerator(log_with="tensorboard", project_dir="./runs")
print(f"Using device: {accelerator.device}")
print(f"Num processes: {accelerator.num_processes}")
print(
    f"Using fsdp: {hasattr(accelerator.state, 'fsdp_plugin') and accelerator.state.fsdp_plugin is not None}"
)
tokenizer = Tokenizer()
max_seq_len = 56
batch_size = 256
train_ds = ShakespeareDataset(tokenizer=tokenizer, max_length=max_seq_len)
train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=3
)
test_ds = ShakespeareDataset(tokenizer=tokenizer, max_length=max_seq_len, train=False)
test_dl = torch.utils.data.DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=3
)


model_kwargs = {
    "max_seq_len": max_seq_len,
    "K": tokenizer.vocab_size(),
    "hidden_dim": 256,
    "num_heads": 8,
    "layers": 3,
}
model = DiscreteModel(**model_kwargs)

print(
    f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters"
)

optimizer_kwargs = {
    "lr": 1e-5,
    "weight_decay": 0.01,
}
opt = Opt(model.parameters(), **optimizer_kwargs)  # pyright: ignore[reportArgumentType]

metadata = CheckpointMetadata(
    model_kwargs=model_kwargs,
    optimizer_kwargs=optimizer_kwargs,
    is_fsdp=hasattr(accelerator.state, "fsdp_plugin")
    and accelerator.state.fsdp_plugin is not None,
    num_accelerators=accelerator.num_processes,
)

checkpoint_name = "shakespeare_full_c_adamw"

accelerator.init_trackers(checkpoint_name)

checkpoint_dir = f"./checkpoint/{checkpoint_name}"

checkpoint_manager = CheckpointManager()
checkpoint_manager.prepare(model, opt, accelerator, metadata)
checkpoint_manager.load(checkpoint_dir, error_if_not_exists=False)
start_epoch = (
    checkpoint_manager.metadata.current_epoch if checkpoint_manager.metadata else 0
)

model, opt = checkpoint_manager.model, checkpoint_manager.optimizer
train_dl, test_dl = accelerator.prepare(train_dl, test_dl)

assert model is not None

print(f"Starting epoch: {start_epoch}")

epochs = 2_300_000

train_context = TrainingContext(
    model=model,
    optimizer=opt,
    train_dl=train_dl,
    epochs=epochs,
    accelerator=accelerator,
    checkpoint_manager=checkpoint_manager,
    save_dir=checkpoint_dir,
    starting_epoch=start_epoch,
    save_every=32000,
    test_every=4000,
    test_inference_steps=100,
    test_dl=test_dl,
)

train_discrete_model(train_context)
