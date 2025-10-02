from tracemalloc import start

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin
from torch.utils.data import DataLoader, Dataset

from src.datasets.discrete_helper import collate_fn
from src.datasets.shakespeare.shakespeare import ShakespeareDataset
from src.inference.discrete_inference import bayesian_inference, dis_t

# from src.nn.models.discrete_model import DiscreteModel
from src.tokenizers.byt5.byt5_tokenizer import ByT5Tokenizer as Tokenizer
from src.training.checkpoint import CheckpointManager, CheckpointMetadata
from src.training.training import TrainingContext, train_discrete_model

accelerator = Accelerator(log_with="tensorboard", project_dir="./runs")

tokenizer = Tokenizer()
max_seq_len = 6
batch_size = 2
train_ds = ShakespeareDataset(tokenizer=tokenizer, max_length=max_seq_len)
train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
test_ds = ShakespeareDataset(tokenizer=tokenizer, max_length=max_seq_len, train=False)
test_dl = torch.utils.data.DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)


class DiscreteModel(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        K: int,
        hidden_dim: int,
        num_heads: int,
        layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisble by num_heads"
        self.emb = nn.Parameter(torch.randn(K, hidden_dim))
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, hidden_dim))
        self.time_vec = nn.Parameter(torch.randn(1, hidden_dim))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    hidden_dim,
                    num_heads,
                    hidden_dim * 4,
                    dropout,
                    batch_first=True,
                    bias=False,
                )
                for i in range(layers)
            ]
        )
        self.classifier = nn.Parameter(torch.randn(hidden_dim, K))

    def token_emb(self, x):
        return x @ self.emb

    def positional_emb(self, x):
        return x + self.pos_emb[: x.shape[1]]

    def time_emb(self, x, t):
        assert t.ndim == 1, "time vector `t` should be vector of length batch_size"
        # we need to first unsqueeze t to get it from shape (batch_size,)
        # to (batch_size, 1) so it is compatible with the time_vec's (1, hidden_dim)
        # the result is (batch_size, hidden_dim) however the x is
        # (batch_size, seq_len, hidden_dim) so we need a second unsqueeze
        return (t.unsqueeze(-1) @ self.time_vec).unsqueeze(-2) + x

    def forward(self, x, t):
        x = self.token_emb(x)
        x = self.positional_emb(x)
        x = self.time_emb(x, t)
        for i, l in enumerate(self.layers):
            x = l(x)
        return x @ self.classifier


learning_rate = 0.01

model_kwargs = {
    "max_seq_len": max_seq_len,
    "K": tokenizer.vocab_size(),
    "hidden_dim": 4,
    "num_heads": 2,
}

# Create model, loss function, and optimizer
model = DiscreteModel(**model_kwargs)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Prepare everything with accelerator
model, optimizer = accelerator.prepare(model, optimizer)
dataloader = accelerator.prepare(train_dl)

emb = model.emb
print(f"Model emb type: {type(emb)}, emb data type: {type(emb.data)}")

# Training loop
model.train()

accelerator.print(f"Starting training on {accelerator.device}")
accelerator.print(f"Dataset size: {len(train_ds)}, Batch size: {batch_size}")
accelerator.print(f"Model: {model}")
accelerator.print("-" * 50)

for batch_idx, (inputs) in enumerate(dataloader):
    # Forward pass
    outputs = model(inputs["theta"], inputs["t"])
    loss = outputs.sum()  # Dummy loss for illustration

    # Backward pass
    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()

    break

# Final evaluation
accelerator.print("-" * 50)
accelerator.print("Training completed!")

# Wait for all processes
accelerator.wait_for_everyone()
