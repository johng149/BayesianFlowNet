import torch

# from src.tokenizers.byt5.byt5_tokenizer import ByT5Tokenizer as Tokenizer
from src.tokenizers.ascii.ascii_tokenizer import ASCIITokenizer as Tokenizer
from src.datasets.shakespeare.shakespeare import ShakespeareDataset
from src.datasets.discrete_helper import collate_fn
from src.nn.models.discrete_model import DiscreteModel
from src.training.training import train_discrete_model
from matplotlib import pyplot as plt
from src.inference.discrete_inference import dis_t, bayesian_inference
from accelerate import Accelerator
from src.training.checkpoint import CheckpointMetadata, CheckpointManager
from safetensors.torch import load_file
import os

accelerator = Accelerator(log_with="trackio", project_dir="./runs/shakespeare")
tokenizer = Tokenizer()
max_seq_len = 32

model_kwargs = {
    "max_seq_len": max_seq_len,
    "K": tokenizer.vocab_size(),
    "hidden_dim": 512,
    "num_heads": 8,
    "layers": 5,
}
model = DiscreteModel(**model_kwargs)

optimizer_kwargs = {
    "lr": 3e-5,
}
opt = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

metadata = CheckpointMetadata(
    model_kwargs=model_kwargs,
    optimizer_kwargs=optimizer_kwargs,
    is_fsdp=hasattr(accelerator.state, "fsdp_plugin")
    and accelerator.state.fsdp_plugin is not None,
    num_accelerators=accelerator.num_processes,
)

checkpoint_dir = (
    "./checkpoint/shakespeare_chonky_silu_xavier_1e-5_beta05_ASCIITokenizer_big_data"
)
checkpoint_manager = CheckpointManager()
checkpoint_manager.prepare(model, opt, accelerator, metadata)
checkpoint_manager.load(checkpoint_dir, error_if_not_exists=False)

model, opt = checkpoint_manager.model, checkpoint_manager.optimizer

quit_loop = False
while not quit_loop:
    model_input = (
        torch.normal(0, 1, (1, max_seq_len, tokenizer.vocab_size())).to(
            accelerator.device
        )
        + torch.randn(1, max_seq_len, tokenizer.vocab_size()).to(accelerator.device)
        * 0.1
    )
    model_input = torch.softmax(model_input, dim=-1) * 2 - 1

    n = 190

    for i in range(1, n + 1):
        if i % 10 == 0:
            tokenizer.decode(model_input.squeeze(0))
        cur_it = torch.tensor([i], device=accelerator.device)
        total_it = torch.tensor([n], device=accelerator.device)
        t = dis_t(cur_it, total_it).to(accelerator.device)
        dis_beta_1 = torch.ones_like(t, device=accelerator.device) * 4

        current_model_input = model_input.clone()

        model_output = model.forward(model_input, t)
        model_input = bayesian_inference(
            model_input, model_output, cur_it, total_it, dis_beta_1
        )

    print("Final model result:")
    print(tokenizer.decode(model_input.squeeze(0)))

    user_input = input("Continue? (y/n): ")
    if user_input.lower() != "y":
        quit_loop = True
